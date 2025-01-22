import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch.utils.checkpoint

import torch.nn.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
)
import types



def apply_rotary_pos_emb_single(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def dynamic_llama_attention_forward(layer_window_ratio : float, head_group : int = 4, head_diff : bool = False):

    def forward_noflashattn_window(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        window_size = int(q_len * layer_window_ratio)

        if q_len % window_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, window_size))
        num_group = q_len // window_size


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # bsz x n_head x q_len x d_head

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        assert kv_seq_len == q_len and not use_cache and not output_attentions
            
        query_states = query_states.reshape(bsz, self.num_heads, num_group, window_size, self.head_dim)
        key_states = key_states.reshape(bsz, self.num_heads, num_group, window_size, self.head_dim)
        value_states = value_states.reshape(bsz, self.num_heads, num_group, window_size, self.head_dim)
        # bsz x n_head x n_group x window_size x d_head

        attn_output_list = []
        for i in range(q_len // window_size):
            sub_query = query_states[:, :, i, :, :]
            sub_key = key_states[:, :, i, :, :]
            sub_value = value_states[:, :, i, :, :]
            sub_mask = attention_mask[:, :, :window_size, :window_size]

            if i > 0:
                sub_key = torch.cat((key_states[:, :, i - 1, :, :], sub_key), dim = 2)
                sub_value = torch.cat((value_states[:, :, i - 1, :, :], sub_value), dim = 2)
                mask_value = torch.finfo(sub_value.dtype).min    
                sub_mask = torch.cat((torch.tril(torch.full_like(sub_mask, mask_value), diagonal=-1), sub_mask), dim = 3)

            attn_weights = torch.einsum("bnid,bnjd->bnij", (sub_query, sub_key))  / math.sqrt(self.head_dim) + sub_mask
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(sub_query.dtype)
            # attn_weights: bsz x n_head x i x j
            # sub_value: bsz x n_head x j x d
            attn_output = torch.matmul(attn_weights, sub_value)

            attn_output_list.append(attn_output)

        attn_output = torch.cat(attn_output_list, dim = 2)  # bsz x n_head x q_len x d_head

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)  # bsz x n_head x d_embed

        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


    def forward_diff_window(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        window_size = int(q_len * layer_window_ratio)

        if q_len % window_size > 0:
            raise ValueError("q_len %d should be divisible by group size %d."%(q_len, window_size))
        num_group = q_len // window_size


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # bsz x n_head x q_len x d_head

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        assert kv_seq_len == q_len and not use_cache and not output_attentions

        def window_attention(query_states, key_states, value_states, window_size):

            n_head = query_states.size(1)
            n_group = q_len // window_size
            query_states = query_states.reshape(bsz, n_head, n_group, window_size, self.head_dim)
            key_states = key_states.reshape(bsz, n_head, n_group, window_size, self.head_dim)
            value_states = value_states.reshape(bsz, n_head, n_group, window_size, self.head_dim)
            # bsz x n_head x n_group x window_size x d_head

            attn_output_list = []
            for i in range(n_group):
                sub_query = query_states[:, :, i, :, :]
                sub_key = key_states[:, :, i, :, :]
                sub_value = value_states[:, :, i, :, :]
                sub_mask = attention_mask[0, 0, :window_size, :window_size]

                if i > 0:
                    sub_key = torch.cat((key_states[:, :, i - 1, :, :], sub_key), dim = 2)
                    sub_value = torch.cat((value_states[:, :, i - 1, :, :], sub_value), dim = 2)
                    mask_value = torch.finfo(sub_value.dtype).min    
                    sub_mask = torch.cat((torch.tril(torch.full_like(sub_mask, mask_value), diagonal=-1), sub_mask), dim = 1)

                attn_weights = torch.einsum("bnid,bnjd->bnij", (sub_query, sub_key))  / math.sqrt(self.head_dim) + sub_mask
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(sub_query.dtype)
                # attn_weights: bsz x n_head x i x j
                # sub_value: bsz x n_head x j x d
                attn_output = torch.matmul(attn_weights, sub_value)

                attn_output_list.append(attn_output)

            attn_output = torch.cat(attn_output_list, dim = 2)  # bsz x n_head x q_len x d_head
            return attn_output

        if head_diff is False:
            attn_output = window_attention(query_states, key_states, value_states, window_size)
        else:
            query_states = query_states.reshape(bsz, head_group, self.num_heads // head_group, q_len, self.head_dim)
            key_states = key_states.reshape(bsz, head_group, self.num_heads // head_group, q_len, self.head_dim)
            value_states = value_states.reshape(bsz, head_group, self.num_heads // head_group, q_len, self.head_dim)
            head_output_list = []
            for idx in range(head_group):
                sub_window_size = window_size * (head_group // 2) // min(head_group, (2 ** idx))
                # print(sub_window_size)
                sub_attn_output = window_attention(query_states[:,idx,:,:,:], key_states[:,idx,:,:,:], value_states[:,idx,:,:,:], sub_window_size)
                head_output_list.append(sub_attn_output)
            attn_output = torch.cat(head_output_list, dim = 1)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)  # bsz x n_head x d_embed

        attn_output = self.o_proj(attn_output)

        return attn_output, None, None
    
    
    return forward_diff_window



reverse_layer_idx = 0
def enable_llama_dynamic_attention(model, window_size_ratio = 1 / 4, group = 4, layer_diff = False, head_diff = False):
    global reverse_layer_idx
    '''
    e.g. group = 4, num_layer = 4, window_size_ratio = 1 / 4
    layer 4: layer_window_ratio = 1 / 2
    layer 3: layer_window_ratio = 1 / 4
    layer 2: layer_window_ratio = 1 / 8
    layer 1: layer_window_ratio = 1 / 8

    depends on model.config.num_hidden_layers
    calculated by reverse_layer_idx // (num_hidden_layers // group)
    '''
    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            enable_llama_dynamic_attention(
                module, window_size_ratio, group, layer_diff, head_diff
            )

        if isinstance(module, LlamaAttention):
            if layer_diff:
                layer_window_ratio = window_size_ratio * (group // 2) / min(group, (2 ** (reverse_layer_idx // (module.config.num_hidden_layers // group))))
            else:
                layer_window_ratio = window_size_ratio
            
            model._modules[name].forward = types.MethodType(
                # dynamic_llama_attention_forward(layer_start = start_size, layer_recent = 511), model._modules[name]
                dynamic_llama_attention_forward(layer_window_ratio, group, head_diff), model._modules[name]
            )
            reverse_layer_idx += 1