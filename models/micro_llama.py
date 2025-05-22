from typing import Optional, Tuple, Union, List, Callable
import logging 
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding, 
    LlamaRMSNorm, 
    LlamaMLP,
    LlamaDecoderLayer,
    KwargsForCausalLM,
    LlamaPreTrainedModel, 
    GenerationMixin,
    apply_rotary_pos_emb,
    eager_attention_forward,

)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache, StaticCache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import is_torchdynamo_compiling
from models.modules import CausalLMOutputWithPast

logger = logging.getLogger(__name__)

def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

class MiCRoLlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.use_router = config.use_router
        self.ablate = config.ablate
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_dim // config.num_attention_heads
        if isinstance(self.ablate, str):
            self.ablate = [self.ablate]

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False),
            nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        )

        self.num_layers = config.backbone_num_layers
        self.layer_idx = layer_idx

        self.experts = nn.ModuleList([LlamaDecoderLayer(config, layer_idx * self.num_experts + expert_idx) for expert_idx in range(self.num_experts)])

        self.jitter_noise = config.jitter_noise

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
            expert_layer: LlamaDecoderLayer = self.experts[expert_idx]
            batch_indices, seq_indices, top_k_indices = torch.where(expert_mask[..., expert_idx])

            if not self.training and sequence_length == 1 and batch_indices.numel() == 0:
                if past_key_value is not None:
                    
                    hidden_state_ln_norm = expert_layer.input_layernorm(hidden_states)

                    input_shape = hidden_state_ln_norm.shape[:-1]
                    hidden_shape = (*input_shape, -1, self.head_dim)

                    # query_states = expert_layer.self_attn.q_proj(hidden_state_ln_norm).view(hidden_shape).transpose(1, 2)
                    key_states = expert_layer.self_attn.k_proj(hidden_state_ln_norm).view(hidden_shape).transpose(1, 2)
                    value_states = expert_layer.self_attn.v_proj(hidden_state_ln_norm).view(hidden_shape).transpose(1, 2)

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
    
class MiCRoLlama(LlamaPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        with open(config.config_path, 'r', encoding="utf-8") as file:
            run_config = yaml.load(file.read(), Loader=yaml.FullLoader)

        self.config: LlamaConfig = config
        self.config.torch_dtype = torch.bfloat16
        self.config.use_bfloat16 = True
        self.config._attn_implementation = "flash_attention_2" # {sdpa, flash_attention_2, eager}
        self.config.use_cache = True
        self.config.backbone_num_layers = self.config.num_hidden_layers
        self.config.num_hidden_layers = self.config.num_hidden_layers * run_config["num-experts"]
        self.config.loss_type = "ForCausalLMLoss"
        
        self.tokenizer = AutoTokenizer.from_pretrained(run_config["tokenizer"])
        self.assistant_header_ids = torch.tensor(self.tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>")[1:])
        self.user_header_ids = torch.tensor(self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>")[1:])

        super(MiCRoLlama, self).__init__(self.config)
        self.build_model(run_config)

    def build_model(self, run_config):
    
        self.gradient_checkpointing = False
        self.config.num_experts = run_config["num-experts"]
        self.config.use_router = run_config["use-router"]
        self.config.num_experts_per_tok = run_config["top-k-experts"]
        self.config.jitter_noise = run_config["jitter-noise"]
        self.config.loss_method = run_config.get("loss", "all")

        self.run_config = run_config
        self.padding_idx = 128004
        
        # MiCRoLlama model
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MiCRoLlamaDecoderLayer(self.config, layer_idx) for layer_idx in range(self.config.backbone_num_layers)])
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)
        self.final_norm = LlamaRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Freeze Model
        for param in self.parameters():
            param.requires_grad = False

        # Unfreeze Modules
        if "reasoners" in run_config["trainable"]:
            print(">> Unfreezing Reasoning Modules")
            for layer in self.layers:
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

            for param in self.final_norm.parameters():
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

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        routing_weights: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

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

            if self.gradient_checkpointing and self.training:
                layer_outputs, router_logits = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    routing_weights,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
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
                )

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            all_routing_weights += (router_logits,)

        hidden_states = self.final_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + (past_key_values, all_hidden_states, all_self_attns, all_routing_weights) if use_cache else (logits, all_hidden_states, all_self_attns, all_routing_weights) 
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            routing_weights=all_routing_weights,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    def load_pretrained(self, model_name):
        base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.lm_head.load_state_dict(base_model.lm_head.state_dict())
        self.embed_tokens.load_state_dict(base_model.get_input_embeddings().state_dict())
        self.rotary_emb.load_state_dict(base_model.model.rotary_emb.state_dict())
        self.final_norm.load_state_dict(base_model.model.norm.state_dict())
        for layer_idx, layer in enumerate(self.layers):
            base_model_layer = base_model.model.layers[layer_idx].state_dict()
            for expert in layer.experts:
                expert.load_state_dict(base_model_layer)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs