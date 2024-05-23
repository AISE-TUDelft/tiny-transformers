import torch
from torch import nn
from transformers import GPTNeoForCausalLM


class GPTNeoGQASelfAttention(nn.Module):
        def __init__(self, config, attention_type, num_query_groups=None, kqv_size=None):
            super().__init__()
            # print("KQV_size: ", kqv_size)
            # print("Got to constructor")
            self.config = config

            if num_query_groups is None:
                self.num_query_groups = config.num_heads
            else:
                self.num_query_groups = num_query_groups

            if kqv_size is None:
                self.kqv_size = config.hidden_size
            else:
                self.kqv_size = kqv_size

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
            self.head_dim = self.kqv_size // self.num_heads
            # print("Left: ", self.head_dim * self.num_heads)
            # print("Right: ", self.kqv_size)
            if self.head_dim * self.num_heads != self.kqv_size:
                raise ValueError(
                    f"embed_dim and kqv size must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                    f" {self.num_heads})."
                )

            self.k_proj = nn.Linear(self.embed_dim, self.kqv_size, bias=False)
            self.v_proj = nn.Linear(self.embed_dim, self.kqv_size, bias=False)
            self.q_proj = nn.Linear(self.embed_dim, self.kqv_size, bias=False)
            self.out_proj = nn.Linear(self.kqv_size, self.embed_dim, bias=True)

        def _split_heads(self, tensor, num_heads, attn_head_size):
            """
            Splits hidden_size dim into attn_head_size and num_heads
            """
            # attn_head_size is self.head_dim = self.embed_dim // self.num_heads
            # print("Size in split heads passed as the input: ", tensor.shape)
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            # print("New shape in split heads before permuting: ", new_shape)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

        def _merge_heads(self, tensor, num_heads, attn_head_size):
            """
            Merges attn_head_size dim and num_attn_heads dim into hidden_size
            """
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
            return tensor.view(new_shape)

        def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            # Keep the attention weights computation in fp32 to avoid overflow issues
            # print("Got here _attn")
            query = query.to(torch.float32)
            key = key.to(torch.float32)

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

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(value.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)
            # print("Attn weight: ", attn_weights.size())

            return attn_output, attn_weights

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            # use_cache is set to true
            # output_attentions is set to false
            # print("Got to forward")
            # print("Hidden state size: ", hidden_states.size())
            query = self.q_proj(hidden_states)
            # print("Query size: ", query.size())
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
            # print("key shape: ", key.shape)
            # print("value shape: ", value.shape)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

            if layer_past is not None:
                # since cache is used, previous keys and values are extracted here
                # each time, the dimensionality is increased by 1 along sequence length dimension
                past_key = layer_past[0]
                past_value = layer_past[1]
                # print("past key shape: ", past_key.shape)
                # print("past value shape: ", past_value.shape)
                # print("key shape: ", key.shape)
                # print("value shape: ", value.shape)
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
                # print("key after: ", key.shape)
                # print("value after: ", value.shape)

            if use_cache is True:
                present = (key, value)
            else:
                present = None
            # print("Key shape before calculating: ", key.shape)
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            # print("Grid shape: ", attn_output.shape)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            # print("Shape after merging heads: ", attn_output.shape)
            attn_output = self.out_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            # print("Final result: ", attn_output.shape)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)
            # print("Final result 2: ", outputs[0].shape)
            return outputs  # a, present, (attentions)


class CustomGPTNeoForCausalLM(GPTNeoForCausalLM):
    def __init__(self, config_gqa):
        self.config = config_gqa
        super().__init__(config_gqa)
    def set_kqv_size(self, kqv_size):
        self.kqv_size = kqv_size

    def set_attention(self, num_query_groups=None):
        num_groups = None
        if num_query_groups is None:
            num_groups = self.config.num_heads
        else:
            num_groups = num_query_groups
        for block in self.transformer.h:
            block.attn.attention = GPTNeoGQASelfAttention(self.config, block.attn.attention_type, num_groups, self.kqv_size)