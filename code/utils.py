from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import LlamaAttention


class RectifiedAttention(nn.Module):
    def __init__(
        self, llama_attention: LlamaAttention, fixed_head_idx: int, coefficient: float
    ):
        super().__init__()
        self.llama_attention = llama_attention

        self.fixed_head_idx = fixed_head_idx
        if not (
            -llama_attention.num_heads <= fixed_head_idx < llama_attention.num_heads
        ):
            raise ValueError(
                f"fixed_head_idx should be in the range [{-llama_attention.num_heads}, {llama_attention.num_heads})"
            )

        self.coefficient = coefficient
        if not (0 <= self.coefficient <= 1):
            raise ValueError("coefficient should be in the range [0, 1]")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        attn_output, attn_weights, past_key_value = self.llama_attention.forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            cache_position,
            **kwargs,
        )  # atten_output shape: (bsz, q_len, num_heads*head_dim)

        bsz, q_len, hidden_size = attn_output.shape
        num_heads = self.llama_attention.num_heads
        head_dim = self.llama_attention.head_dim

        reshaped_attn_output = attn_output.reshape(bsz, q_len, num_heads, head_dim)
        reshaped_attn_output[:, :, : self.fixed_head_idx, :] *= self.coefficient
        reshaped_attn_output[:, :, self.fixed_head_idx+1 :, :] *= self.coefficient
        rectified_attn_output = reshaped_attn_output.reshape(bsz, q_len, hidden_size)

        if not output_attentions:
            attn_weights_zeroed = None
        else:
            raise NotImplementedError("output_attentions=True is not supported yet")

        return rectified_attn_output, attn_weights_zeroed, past_key_value


def replace_attention(
    model: AutoModelForCausalLM, layer_idx: int, fixed_head_idx: int, coefficient: float
) -> AutoModelForCausalLM:
    """Replace the attention mechanism of a specific layer with a rectified attention mechanism.

    Args:
        model (AutoModelForCausalLM): The model to modify.
        layer_idx (int): The index of the layer to modify.
        fixed_head_idx (int): The index of the head to keep unchanged.
        coefficient (float): The coefficient to apply to the heads other than the fixed head.
    """

    if not (-len(model.model.layers) <= layer_idx < len(model.model.layers)):
        raise ValueError(
            f"layer_idx should be in the range [{-len(model.model.layers)}, {len(model.model.layers)})"
        )
    model.model.layers[layer_idx].self_attn = RectifiedAttention(
        llama_attention=model.model.layers[layer_idx].self_attn,
        fixed_head_idx=fixed_head_idx,
        coefficient=coefficient,
    )
    return model
