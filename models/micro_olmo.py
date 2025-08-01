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
# from transformers.modeling_layers import GradientCheckpointingLayer
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
    Olmo2DecoderLayer,
    Olmo2RotaryEmbedding,
    Olmo2PreTrainedModel,
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

class MiCRoOLMo2DecoderLayer(nn.Module):
    def __init__(self, config: Olmo2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.num_experts  = config.num_experts
        self.top_k        = config.num_experts_per_tok
        self.use_router   = config.use_router
        self.ablate       = config.ablate or []
        self.num_layers   = config.backbone_num_layers
        self.layer_idx    = layer_idx
        self.jitter_noise = config.jitter_noise
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads

        if isinstance(self.ablate, str):
            self.ablate = [self.ablate]

        # gating head
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Linear(self.hidden_size, self.num_experts, bias=False),
        )

        self.experts = nn.ModuleList([
            Olmo2DecoderLayer(config, layer_idx * self.num_experts + expert_idx)
            for expert_idx in range(self.num_experts)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        routing_weights: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        
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
            if len(routing_weights.shape) == 2:
                routing_weights = routing_weights.unsqueeze(1).tile((1,sequence_length,1)).float()
            else:
                routing_weights = routing_weights.float()
            router_logits = routing_weights

        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= (routing_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # We'll accumulate outputs here
        final_hidden_states = torch.zeros_like(hidden_states)

        # Flatten final_hidden_states to [batch_size * seq_len, hidden_dim]
        # so we can do a 2D "index_add_" at the end of each loop.
        final_hidden_states_2d = final_hidden_states.view(-1, hidden_dim)
    
        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        #^ [batch_size, seq_len, top_k, num_experts]

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer: Olmo2DecoderLayer = self.experts[expert_idx]
            batch_indices, seq_indices, top_k_indices = torch.where(expert_mask[..., expert_idx])
        
            if not self.training and sequence_length == 1 and batch_indices.numel() == 0:
                if past_key_value is not None:
                    
                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)

                    key_states = expert_layer.self_attn.k_proj(hidden_states)
                    key_states = expert_layer.self_attn.k_norm(key_states).view(hidden_shape).transpose(1, 2)
                    value_states = expert_layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)


                    cos, sin = position_embeddings
                    _, key_states = apply_rotary_pos_emb(key_states, key_states, cos, sin)
                    # sin and cos are specific to RoPE models; cache_position needed for the static cache
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    past_key_value.update(key_states, value_states, self.layer_idx * self.num_experts + expert_idx, cache_kwargs)

                continue
        
            current_hidden_states = expert_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )[0]
            
            flat_idx = batch_indices * sequence_length + seq_indices
            expert_weights = routing_weights[batch_indices, seq_indices, top_k_indices].unsqueeze(-1)
            current_hidden_states = current_hidden_states[batch_indices, seq_indices] * expert_weights

            final_hidden_states_2d.index_add_(0, flat_idx, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states_2d.view(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
class MiCRoOLMo(Olmo2PreTrainedModel, GenerationMixin):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Olmo2DecoderLayer`]

    Args:
        config: Olmo2Config
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Olmo2Config):
        with open(config.config_path, 'r', encoding="utf-8") as file:
            run_config = yaml.load(file.read(), Loader=yaml.FullLoader)

        self.config: Olmo2Config = config
        self.config.torch_dtype = torch.bfloat16
        self.config.use_bfloat16 = True
        self.config._attn_implementation = "flash_attention_2" # {sdpa, flash_attention_2, eager}
        self.config.use_cache = True
        self.config.backbone_num_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers = self.config.num_hidden_layers * run_config["num-experts"]
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
        # Qwen2 model
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MiCRoOLMo2DecoderLayer(self.config, layer_idx) for layer_idx in range(self.config.backbone_num_layers)])
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.rotary_emb = Olmo2RotaryEmbedding(config=self.config)
        self.norm = Olmo2RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Freeze Model
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze Modules
        if "reasoners" in run_config["trainable"]:
            print(">> Unfreezing Reasoning Modules")
            for layer in self.layers:
                layer: MiCRoOLMo2DecoderLayer
                for param in layer.experts.parameters():
                    param.requires_grad = True

        if "model" in run_config["trainable"]:
            print(">> Unfreezing Model")
            for param in self.layers.parameters():
                param.requires_grad = True

            for param in self.lm_head.parameters():
                param.requires_grad = True

            for param in self.rotary_emb.parameters():
                param.requires_grad = True

            for param in self.norm.parameters():
                param.requires_grad = True

            for param in self.embed_tokens.parameters():
                param.requires_grad = True

            for layer in self.layers:
                for param in layer.gate.parameters():
                    param.requires_grad = False


        if "experts-router" in run_config["trainable"]:
            print(">> Unfreezing Experts Router")
            for layer in self.layers:
                for param in layer.gate.parameters():
                    param.requires_grad = True

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
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.lm_head.load_state_dict(base_model.lm_head.state_dict())
        self.embed_tokens.load_state_dict(base_model.get_input_embeddings().state_dict())
        self.rotary_emb.load_state_dict(base_model.model.rotary_emb.state_dict())
        self.norm.load_state_dict(base_model.model.norm.state_dict())
        for layer_idx, layer in enumerate(self.layers):
            base_model_layer = base_model.model.layers[layer_idx].state_dict()
            for expert in layer.experts:
                expert.load_state_dict(base_model_layer)

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


__all__ = ["MiCRoOLMo"]