from typing import Callable, Optional, Tuple, Union

import yaml
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from transformers.models.olmo2.configuration_olmo2 import Olmo2Config
from transformers.models.olmo2.modeling_olmo2 import (
    Olmo2RMSNorm,
    Olmo2Attention,
    Olmo2MLP,
    Olmo2ForCausalLM,
    Olmo2DecoderLayer,
    Olmo2RotaryEmbedding,
    Olmo2PreTrainedModel,
    Olmo2Config,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    eager_attention_forward,
)


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

from models.modules import CausalLMOutputWithPast

logger = logging.get_logger(__name__)

def keep_alive_zero(model):
    z = 0.0
    for p in model.parameters():
        if p.requires_grad:
            # one scalar per param to avoid heavy sums
            z = z + (p.view(-1)[0] * 0.0)
    return z

class MiCRoOlmoMoEConfig(Olmo2Config):
    model_type = "micro_olmo_moe"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts = kwargs.get("num_experts", 4)
        self.use_router = kwargs.get("use_router", True)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 2)
        self.jitter_noise = kwargs.get("jitter_noise", 0.0)
        self.loss_method = kwargs.get("loss_method", "all")
        self.config_path = kwargs.get("config_path", None)

class OlmoSparseMiCRoMoEBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_router = config.use_router
        self.ablate = config.ablate

        # gating
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        )

        self.experts = nn.ModuleList([Olmo2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.jitter_noise

    def forward(self, hidden_states: torch.Tensor, routing_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        if self.use_router:
            router_logits = self.gate(hidden_states)
            if "logic" in self.ablate:
                router_logits[..., 0] = -torch.inf
            if "social" in self.ablate:
                router_logits[..., 1] = -torch.inf
            if "world" in self.ablate:
                router_logits[..., 2] = -torch.inf
            if "language" in self.ablate:
                router_logits[..., 3] = -torch.inf
            routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        else:
            routing_weights = routing_weights.reshape(-1, 4).float()
            router_logits = routing_weights
        # router_logits: (batch * sequence_length, n_experts)
        
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # if top_x.numel() == 0:
            #     # touch the expert with a zero-row input to build the graph
            #     dummy_in = hidden_states[:0]               # [0, hidden_dim]
            #     dummy_out = expert_layer(dummy_in)         # [0, hidden_dim]
            #     # attach a strictly-zero scalar to the graph so params are “used”
            #     final_hidden_states = final_hidden_states + (dummy_out.sum() * 0.0)
            #     continue
                
            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
class OlmoMiCRoMoEDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: MiCRoOlmoMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Olmo2Attention(config=config, layer_idx=layer_idx)

        self.block_sparse_moe = OlmoSparseMiCRoMoEBlock(config)
        self.post_attention_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Olmo2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        routing_weights: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states, router_logits = self.block_sparse_moe(hidden_states, routing_weights)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, router_logits
    
class MiCRoOlmoMoE(Olmo2PreTrainedModel, GenerationMixin):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Olmo2DecoderLayer`]

    Args:
        config: Olmo2Config
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: MiCRoOlmoMoEConfig):
        with open(config.config_path, 'r', encoding="utf-8") as file:
            run_config = yaml.load(file.read(), Loader=yaml.FullLoader)

        self.config: MiCRoOlmoMoEConfig = config
        self.config.torch_dtype = torch.bfloat16
        self.config.use_bfloat16 = True
        self.config._attn_implementation = "flash_attention_2" # {sdpa, flash_attention_2, eager}
        self.config.use_cache = True
        self.config.backbone_num_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers = self.config.num_hidden_layers
        self.config.loss_type = "ForCausalLMLoss"

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.gradient_checkpointing = False
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.build_model(run_config)
    
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def build_model(self, run_config):
        self.gradient_checkpointing = False
        self.config.num_experts = run_config["num-experts"]
        self.config.use_router = run_config["use-router"]
        self.config.num_experts_per_tok = run_config["top-k-experts"]
        self.config.jitter_noise = run_config["jitter-noise"]
        self.config.loss_method = run_config.get("loss", "all")

        self.run_config = run_config        
        # Olmo2 model
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([OlmoMiCRoMoEDecoderLayer(self.config, layer_idx) for layer_idx in range(self.config.backbone_num_layers)])
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.rotary_emb = Olmo2RotaryEmbedding(config=self.config)
        self.norm = Olmo2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Unfreeze Modules
        if "model" not in run_config["trainable"]:
            print(">> Freezing Model Except Experts + Routing Gates")
            for param in self.parameters():
                param.requires_grad = False

            for layer in self.layers:
                layer: OlmoMiCRoMoEDecoderLayer
                for param in layer.block_sparse_moe.parameters():
                    param.requires_grad = True

        if "experts" not in run_config["trainable"]:
            print(">> Freezing Experts")
            for layer in self.layers:
                layer: OlmoMiCRoMoEDecoderLayer
                for param in layer.block_sparse_moe.experts.parameters():
                    param.requires_grad = False

        if "experts-router" not in run_config["trainable"]:
            print(">> Freezing Routing Gates")
            for layer in self.layers:
                layer: OlmoMiCRoMoEDecoderLayer
                for param in layer.block_sparse_moe.gate.parameters():
                    param.requires_grad = False

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        routing_weights: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_routing_weights = ()

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs, router_logits = decoder_layer(
                hidden_states,
                routing_weights=routing_weights,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
                # **flash_attn_kwargs,
            )

            hidden_states = layer_outputs

            # if output_attentions:
            #     all_self_attns += (layer_outputs[1],)
                
            all_routing_weights += (router_logits,)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            routing_weights=all_routing_weights,
        )

    def load_pretrained(self, model_name):
        base_model: Olmo2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.lm_head.load_state_dict(base_model.lm_head.state_dict())
        self.embed_tokens.load_state_dict(base_model.get_input_embeddings().state_dict())
        self.rotary_emb.load_state_dict(base_model.model.rotary_emb.state_dict())
        self.norm.load_state_dict(base_model.model.norm.state_dict())

        for layer_idx, layer in enumerate(self.layers):
            
            layer: OlmoMiCRoMoEDecoderLayer
            attn_layer = base_model.model.layers[layer_idx].self_attn.state_dict()
            layer.self_attn.load_state_dict(attn_layer)
            
            post_attention_layernorm = base_model.model.layers[layer_idx].post_attention_layernorm.state_dict()
            layer.post_attention_layernorm.load_state_dict(post_attention_layernorm)

            post_feedforward_layernorm = base_model.model.layers[layer_idx].post_feedforward_layernorm.state_dict()
            layer.post_feedforward_layernorm.load_state_dict(post_feedforward_layernorm)
            
            mlp_model_layer = base_model.model.layers[layer_idx].mlp.state_dict()
            for expert in layer.block_sparse_moe.experts:
                expert.load_state_dict(mlp_model_layer)

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


__all__ = ["MiCRoOlmo"]