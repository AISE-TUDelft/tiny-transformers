import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QueryNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_heads, batch_norm=True):
        super(QueryNetwork, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.batch_norm = batch_norm
        self.Wq = nn.Linear(dim_in, dim_hidden * num_heads, bias=False)
        self.bn = nn.BatchNorm1d(dim_hidden * num_heads)

    def forward(self, x):
        # for memory of size less than 384*384 the paper says that batch norm gives no significant improvement
        # when using batch norm, padding tokens in the sequence can skew mean and variance estimate
        query = self.Wq(x)

        if self.batch_norm:
            original_shape = query.shape
            query = query.reshape(-1, self.dim_hidden * self.num_heads)
            normalized = self.bn(query)
            normalized = normalized.reshape(original_shape)
            return normalized.reshape(*normalized.shape[:-1], self.num_heads, self.dim_hidden)
        else:
            return query.reshape(*query.shape[:-1], self.num_heads,
                                 self.dim_hidden)  # batch size x context length x num heads x query dim

class ProductKey(nn.Module):
    def __init__(self, dim, num_subkeys, top_k, num_heads):
        super(ProductKey, self).__init__()
        assert dim % 2 == 0, "key must be able to be split into 2"
        self.dim = dim
        self.subkey_size = dim // 2
        self.top_k = top_k
        self.num_subkeys = num_subkeys

        keyl = torch.empty(num_heads, num_subkeys, self.subkey_size)
        keyr = torch.empty(num_heads, num_subkeys, self.subkey_size)

        std = 1 / math.sqrt(dim)
        keyl.uniform_(-std, std)
        keyr.uniform_(-std, std)

        self.keyl = nn.Parameter(keyl)
        self.keyr = nn.Parameter(keyr)

    def forward(self, query):
        # multihead query
        batch, context_length, num_heads, query_size = query.shape

        queryl = query[..., :self.subkey_size]
        queryr = query[..., self.subkey_size:]

        scorel = torch.einsum('bcnq,nkq->bcnk', queryl,
                              self.keyl)  # batch size x context length x num head x subquery length , num heads x num keys x subquery length
        scorer = torch.einsum('bcnq,nkq->bcnk', queryr, self.keyr)

        top_keys_l, top_idx_l = scorel.topk(self.top_k)  # batch, context, heads, top k
        top_keys_r, top_idx_r = scorer.topk(self.top_k)

        # duplicate along the rows
        product_scores_l = top_keys_l.reshape(*top_keys_l.shape[:-1], top_keys_l.shape[-1], 1).expand(
            *top_keys_l.shape[:-1], top_keys_l.shape[-1], top_keys_l.shape[-1])
        # duplicate along the columns
        product_scores_r = top_keys_r.reshape(*top_keys_r.shape[:-1], 1, top_keys_r.shape[-1]).expand(
            *top_keys_r.shape[:-1], top_keys_r.shape[-1], top_keys_r.shape[-1])

        product_scores = (product_scores_l + product_scores_r)  # batch, context, heads, top k, top k
        product_scores = product_scores.reshape(batch, context_length, num_heads, self.top_k * self.top_k)

        product_indices_l = top_idx_l.reshape(*top_idx_l.shape[:-1], top_idx_l.shape[-1], 1).expand(
            *top_idx_l.shape[:-1], top_idx_l.shape[-1], top_idx_l.shape[-1])
        product_indices_r = top_idx_r.reshape(*top_idx_r.shape[:-1], top_idx_r.shape[-1], 1).expand(
            *top_idx_r.shape[:-1], top_idx_r.shape[-1], top_idx_r.shape[-1])

        product_indices = product_indices_l * self.num_subkeys + product_indices_r
        product_indices = product_indices.reshape(batch, context_length, num_heads, self.top_k * self.top_k)

        top_product_scores, top_product_indices = product_scores.topk(self.top_k)
        selected_value_weights = F.softmax(top_product_scores, dim=-1)
        selected_value_indices = torch.gather(product_indices, -1, top_product_indices)
        # print(top_product_scores)
        # print(torch.gather(product_scores, -1, top_product_indices)) # they should be equal
        return selected_value_weights, selected_value_indices

class PKM(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_subkeys, top_k, num_heads, batch_norm=True):
        super(PKM, self).__init__()
        self.dim_in, self.dim_hidden, self.num_subkeys, self.top_k, self.num_heads = dim_in, dim_hidden, num_subkeys, top_k, num_heads
        self.query_network = QueryNetwork(dim_in, dim_hidden, num_heads, batch_norm)
        self.key_table = ProductKey(dim_hidden, num_subkeys, top_k, num_heads)
        self.value_table = nn.Embedding(num_subkeys * num_subkeys, dim_in)

    def forward(self, x):
        queries = self.query_network(x)
        weights, indices = self.key_table(queries)  # shape is batch, context length, num heads, top k
        original_shape = weights.shape

        weights, indices = weights.reshape(-1, self.top_k), indices.reshape(-1, self.top_k)
        values = self.value_table(indices)
        weights = weights.reshape(original_shape)
        values = values.reshape(*original_shape, self.dim_in)

        weighted_values = torch.einsum('bcnk,bcnkd->bcd', weights,
                                       values)  # take linear combination of weights & values and sum over all heads

        return weighted_values
