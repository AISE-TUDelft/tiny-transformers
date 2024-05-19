import torch
from torch import nn
from transformers import GPTNeoForCausalLM


class GPTNeoGQASelfAttention(nn.Module):
        def __init__(self, config, attention_type, num_kv_heads=None, kqv_size=None):
            super().__init__()
            # print("KQV_size: ", kqv_size)
            # print("Got to constructor")
            self.config = config

            self.num_kv_heads = num_kv_heads
            self.gqa_factor = num_kv_heads / self.config.num_heads

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
            if self.kqv_size * self.gqa_factor != int(self.kqv_size * self.gqa_factor):
                raise ValueError (
                    f"kqv size must be divisible by number of kv heads (got `kqv_size`: {self.kqv_size} and `num_kv_heads`:"
                    f" {self.num_kv_heads})."
                )

            self.k_proj = nn.Linear(self.embed_dim, int(self.kqv_size * self.gqa_factor), bias=False)
            self.v_proj = nn.Linear(self.embed_dim, int(self.kqv_size * self.gqa_factor), bias=False)

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
            # calculates attention outputs and weights for one query
            # Keep the attention weights computation in fp32 to avoid overflow issues
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            # print("_attn key shape: ", key.shape)
            # print("_attn query shape: ", query.shape)
            # transpose part is not changed, but dot product has to be edited
            attn_weights = torch.matmul(query, key.transpose(-1, -2))
            # print("_attn weights torch.matmul(query, key.transpose(-1, -2)): ", attn_weights.shape)

            # Transpose the key tensor along the last two dimensions
            key_transposed = key.transpose(-1, -2)
            # Initialize a list to collect the attention weights for each head
            attn_weights_list = []

            # Loop over each head (second dimension)
            i, j = 0, 0
            query_leftover = self.num_heads % self.num_kv_heads
            queries_per_group = (self.num_heads // self.num_kv_heads) + 1 if query_leftover > 0 \
                else self.num_heads // self.num_kv_heads
            while i < key.shape[1] :
                key_head = key_transposed[:, i, :, :]
                for k in range (j, j + queries_per_group):
                    query_head = query[:, k, :, :]
                    attn_weight_qk = torch.matmul(query_head, key_head)
                    attn_weights_list.append(attn_weight_qk)
                j += queries_per_group
                i += 1
                query_leftover -= 1

            attn_weights_decomposed = torch.stack(attn_weights_list, dim=1)

            # attn_weights_list = []
            # for head_idx in range(query.shape[1]):
            #     # Extract the query and key for the current head
            #     query_head = query[:, head_idx, :, :]  # Shape: [1, 12, 16]
            #     key_transposed_head = key_transposed[:, head_idx, :, :]  # Shape: [1, 16, 12]
            #
            #     # Perform the dot product
            #     attn_weight_head = torch.matmul(query_head, key_transposed_head)  # Shape: [1, 12, 12]
            #     # Append the result to the list
            #     attn_weights_list.append(attn_weight_head)
            # # Concatenate the results along the head dimension (second dimension)
            # attn_weights_decomposed = torch.stack(attn_weights_list, dim=1)  # Shape: [1, 4, 12, 12]
            # assert torch.allclose(attn_weights_decomposed, attn_weights)

            query_length, key_length = query.size(-2), key.size(-2)     # no need to change
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            # mask_value = torch.finfo(attn_weights.dtype).min
            # # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            # mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            # attn_weights = torch.where(causal_mask, attn_weights, mask_value)
            #
            # if attention_mask is not None:
            #     # Apply the attention mask
            #     attn_weights = attn_weights + attention_mask
            #
            # attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            # attn_weights = attn_weights.to(value.dtype)
            # attn_weights = self.attn_dropout(attn_weights)

            mask_value = torch.finfo(attn_weights_decomposed.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights_decomposed.dtype).to(attn_weights_decomposed.device)
            attn_weights_decomposed = torch.where(causal_mask, attn_weights_decomposed, mask_value)

            if attention_mask is not None:
                # Apply the attention mask
                attn_weights_decomposed = attn_weights_decomposed + attention_mask

            attn_weights_decomposed = nn.functional.softmax(attn_weights_decomposed, dim=-1)
            attn_weights_decomposed = attn_weights_decomposed.to(value.dtype)
            attn_weights_decomposed = self.attn_dropout(attn_weights_decomposed)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights_decomposed = attn_weights_decomposed * head_mask

            # assert torch.allclose(attn_weights_decomposed, attn_weights)
            # Initialize a list to collect the attention outputs for each head
            attn_output_list = []
            # Loop over each query head and apply the corresponding attention weights to the value heads
            i, j = 0, 0
            query_leftover = self.num_heads % self.num_kv_heads
            while i < value.shape[1]:
                value_head = value[:, i, :, :]
                for k in range(j, j + queries_per_group):
                    attn_weight_head = attn_weights_decomposed[:, k, :, :]
                    attn_output_head = torch.matmul(attn_weight_head, value_head)
                    attn_output_list.append(attn_output_head)
                j += queries_per_group
                i += 1
                query_leftover -= 1
            attn_output_decomposed = torch.stack(attn_output_list, dim=1)  # Shape: [1, 4, 12, 16]
            # attn_output_list = []
            # for head_idx in range(value.shape[1]):
            #     # Determine which value head to use
            #
            #     # Extract the value for the current head
            #     value_head = value[:, head_idx, :, :]  # Shape: [1, 12, 16]
            #
            #     # Extract the attention weights for the current head
            #     attn_weight_head = attn_weights_decomposed[:, head_idx, :, :]  # Shape: [1, 12, 12]
            #
            #     # Perform the weighted sum (dot product with attention weights)
            #     attn_output_head = torch.matmul(attn_weight_head, value_head)  # Shape: [1, 12, 16]
            #
            #     # Append the result to the list
            #     attn_output_list.append(attn_output_head)
            #
            # # Concatenate the results along the head dimension (second dimension)
            # attn_output_decomposed = torch.stack(attn_output_list, dim=1)  # Shape: [1, 4, 12, 16]
            #
            # attn_output = torch.matmul(attn_weights_decomposed, value)
            # assert torch.allclose(attn_output_decomposed, attn_output)

            return attn_output_decomposed, attn_weights_decomposed

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
            key = self._split_heads(key, self.num_kv_heads, self.head_dim)
            value = self._split_heads(value, self.num_kv_heads, self.head_dim)

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

    def set_attention(self, num_key_value_heads=None):
        num_groups = None
        if num_key_value_heads is None:
            num_kv_heads = self.config.num_heads
        else:
            num_kv_heads = num_key_value_heads
        for block in self.transformer.h:
            block.attn.attention = GPTNeoGQASelfAttention(self.config, block.attn.attention_type, num_kv_heads, self.kqv_size)