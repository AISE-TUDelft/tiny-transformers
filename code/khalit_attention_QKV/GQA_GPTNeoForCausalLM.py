import torch
from torch import nn
from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification
from CustomGQA_GPTNeoConfig import CustomGPTNeoConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class GPTNeoGQASelfAttention(nn.Module):
        def __init__(self, config, attention_type):
            super().__init__()
            # print("KQV_size: ", kqv_size)
            # print("Got to constructor")
            self.config = config

            self.num_kv_heads = config.num_kv_heads
            self.gqa_factor = config.num_kv_heads / self.config.num_heads

            self.kqv_size = config.kqv_size

            max_positions = config.max_position_embeddings
            bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
                1, 1, max_positions, max_positions
            )

            self.group_sizes = [(config.num_heads // config.num_kv_heads) + (1 if i < config.num_heads % config.num_kv_heads else 0)
                            for i in range(config.num_kv_heads)]
            
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
            # Ensure query and key are in float32 to avoid overflow
            # attn_2, attn_w_2 = self.attn_second(query, key, value, attention_mask, head_mask)
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            value = value.to(torch.float32)
            
            device = query.device  # Get the device of the query tensor
            
            # Prepare query groups
            query_indices = []
            for i, size in enumerate(self.group_sizes):
                query_indices.extend([i] * size)
            query_indices = torch.tensor(query_indices, device=device)
            
            # Reorganize queries to match the groups
            queries_reorganized = torch.index_select(query, 1, query_indices)

            # Prepare key groups
            key_indices = torch.arange(key.size(1), device=device)
            key_indices_expanded = key_indices.repeat_interleave(torch.tensor(self.group_sizes, device=device))
            
            # Reorganize keys to match the groups and expand them
            keys_reorganized = torch.index_select(key, 1, key_indices_expanded).transpose(-1, -2)

            # Perform batched matrix multiplication
            attn_weights = torch.matmul(queries_reorganized, keys_reorganized)

            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype, device=device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)

            if attention_mask is not None:
                # Apply the attention mask
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)
            attn_weights = attn_weights.to(value.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            # Reorganize value tensor to match the groups
            values_reorganized = torch.index_select(value, 1, key_indices_expanded)

            # Compute the attention output
            attn_output = torch.matmul(attn_weights, values_reorganized)

            # Reshape the output to merge the groups back
            attn_output = attn_output.view(attn_output.size(0), -1, attn_output.size(-2), attn_output.size(-1))
            # print(torch.allclose(attn_weights, attn_w_2, atol=1e-6))
            # print(torch.allclose(attn_output, attn_2, atol=1e-6))
            return attn_output, attn_weights

        # def attn_second(self, query, key, value, attention_mask=None, head_mask=None):
        #     # calculates attention outputs and weights for one query
        #     # Keep the attention weights computation in fp32 to avoid overflow issues
        #     query = query.to(torch.float32)
        #     key = key.to(torch.float32)

        #     key_transposed = key.transpose(-1, -2)

        #     # Initialize variables for grouping
        #     query_leftover = self.num_heads % self.num_kv_heads
        #     queries_per_group = self.num_heads // self.num_kv_heads
            
        #     # Initialize the list to collect attention weights
        #     attn_weights_list = []
        #     j = 0

        #     # Loop over each key head (i)
        #     for i in range(key.shape[1]):
        #         key_head = key_transposed[:, i, :, :]

        #         # Determine how many query heads to process in this group
        #         if query_leftover <= 0:
        #             toGroup = queries_per_group
        #         else:
        #             toGroup = queries_per_group + 1
        #             query_leftover -= 1
                
        #         # Collect the current group of query heads
        #         query_group = query[:, j:j + toGroup, :, :]

        #         # Perform batched matrix multiplication for the group
        #         attn_weight_qk = torch.matmul(query_group, key_head.unsqueeze(1))

        #         # Append the result to the list
        #         attn_weights_list.append(attn_weight_qk)
        #         j += toGroup

        #     # Stack the attention weights list into a single tensor
        #     attn_weights_decomposed = torch.cat(attn_weights_list, dim=1)

        #     # Determine query and key lengths
        #     query_length, key_length = query.size(-2), key.size(-2)

        #     # Apply causal mask
        #     causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

        #     mask_value = torch.finfo(attn_weights_decomposed.dtype).min
        #     mask_value = torch.tensor(mask_value, dtype=attn_weights_decomposed.dtype).to(attn_weights_decomposed.device)
        #     attn_weights_decomposed = torch.where(causal_mask, attn_weights_decomposed, mask_value)

        #     if attention_mask is not None:
        #         attn_weights_decomposed = attn_weights_decomposed + attention_mask

        #     attn_weights_decomposed = nn.functional.softmax(attn_weights_decomposed, dim=-1)
        #     attn_weights_decomposed = attn_weights_decomposed.to(value.dtype)
        #     attn_weights_decomposed = self.attn_dropout(attn_weights_decomposed)

        #     if head_mask is not None:
        #         attn_weights_decomposed = attn_weights_decomposed * head_mask

        #     # Optimize value multiplication using torch.matmul
        #     attn_output_list = []
        #     j = 0
        #     query_leftover = self.num_heads % self.num_kv_heads
        #     for i in range(value.shape[1]):
        #         value_head = value[:, i, :, :]

        #         if query_leftover <= 0:
        #             toGroup = queries_per_group
        #         else:
        #             toGroup = queries_per_group + 1
        #             query_leftover -= 1

        #         attn_weight_group = attn_weights_decomposed[:, j:j + toGroup, :, :]

        #         # Perform batched matrix multiplication for the group
        #         attn_output_group = torch.matmul(attn_weight_group, value_head.unsqueeze(1))
        #         attn_output_list.append(attn_output_group)
        #         j += toGroup

        #     attn_output_decomposed = torch.cat(attn_output_list, dim=1)

        #     return attn_output_decomposed, attn_weights_decomposed
        
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            # use_cache is set to true by default
            # output_attentions is set to false by default

            query = self.q_proj(hidden_states)

            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_kv_heads, self.head_dim)
            value = self._split_heads(value, self.num_kv_heads, self.head_dim)

            if layer_past is not None:
                # since cache is used, previous keys and values are extracted here
                # each time, the dimensionality is increased by 1 along sequence length dimension
                past_key = layer_past[0]
                past_value = layer_past[1]

                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            attn_output = self.out_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)


            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)
            return outputs  # a, present, (attentions)

class CustomGPTNeoForCausalLM(GPTNeoForCausalLM):
    config_class = CustomGPTNeoConfig
    def __init__(self, config: CustomGPTNeoConfig):
        self.config = config
        super().__init__(config)
        self.kqv_size = config.kqv_size
        self.num_kv_heads = config.num_kv_heads
        for block in self.transformer.h:
            block.attn.attention = GPTNeoGQASelfAttention(self.config, block.attn.attention_type)

class CustomGPTNeoForSequenceClassification(GPTNeoForSequenceClassification):
    def __init__(self, config: CustomGPTNeoConfig):
        super().__init__(config)
        self.config = config
        self.kqv_size = config.kqv_size
        self.num_kv_heads = config.num_kv_heads
        
        # Replace the attention mechanism in each transformer block
        for block in self.transformer.h:
            block.attn.attention = GPTNeoGQASelfAttention(self.config, block.attn.attention_type)