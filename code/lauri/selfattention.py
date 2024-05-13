import os
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from PE import RoPEEmbeddings

class InfiniAttention(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
            1, 1, max_positions, max_positions
        )

        # local causal self attention is a sliding window where each token can only attend to the previous
        # window_size tokens. This is implemented by updating the causal mask such that for each token
        # all other tokens are masked except the previous window_size tokens.
        if attention_type == "local":
            bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

        self.register_buffer("bias", bias, persistent=False)
        self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

        self.attn_dropout = nn.Dropout(float(config.attention_dropout))
        self.resid_dropout = nn.Dropout(float(config.resid_dropout))
        self.is_causal = True

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        
        self.position_embedder = RoPEEmbeddings(dim=self.head_dim, seq_len=max_positions)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.ELU = nn.ELU()
        self.betas = nn.Parameter(torch.randn(1, self.num_heads, 1, 1))
        # needs to become num_heads, head_dim, head_dim
        self.register_buffer("mem", torch.zeros((self.num_heads,self.head_dim, self.head_dim)))
        # self.register_buffer("mem", torch.zeros((self.embed_dim, self.embed_dim)))
        # needs to become num_heads, head_dim, 1
        self.register_buffer("z", torch.zeros((self.num_heads, self.head_dim, 1)))
        # self.register_buffer("z", torch.zeros((self.embed_dim, 1)))


    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden_size
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, mem, z, attention_mask=None, head_mask=None):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # We introduce our non-linearity
        sigma_q = self.ELU(query) + 1.0
        sigma_k = self.ELU(key) + 1.0
        print("memory before the split", mem.size())
        print("z before the split", z.size())
        # We retrieve from memory before we add to it
        A_mem = ((torch.matmul(sigma_q, mem)) / ((torch.matmul(sigma_q, z)) + 1e-6))
        print("A_mem size", A_mem.size())

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        # Vanilla A_dot local attention
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        # print("A_dot size", attn_weights.size())
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        print("attention output size before thing", attn_output.size())

        attn_output = F.sigmoid(self.betas) * A_mem + (torch.ones(1, 4, 1, 1) - F.sigmoid(self.betas)) * attn_output

        print("attention output size", attn_output.size())
        delta = torch.matmul(sigma_k, mem) / (torch.matmul(sigma_k, z) + 1e-6)
        print("mem update", (torch.matmul(sigma_k.transpose(-2, -1), value - delta)).size())
        mem = mem + torch.matmul(sigma_k.transpose(-2, -1), value - delta)
        z = z + sigma_k.sum(dim=-2, keepdim=True)
        print("z update", (sigma_k.sum(dim=-2, keepdim=True)).size())

        print("z after the split", z.size())
        print("memory after the split", mem.size())


        return attn_output, attn_weights, mem, z

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_past=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):  

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        mem = self.mem
        z = self.z
        print("memory before the split", mem.size())
        print("z before the split", z.size())

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        attn_output, attn_weights, mem, z = self._attn(query, key, value, mem, z, attention_mask, head_mask)
        self.mem = mem
        self.z = z

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)