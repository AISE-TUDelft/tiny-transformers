import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Experts(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_experts):
        super(Experts, self).__init__()
        self.dim_in = dim_in
        self.num_experts = num_experts
        W1 = torch.empty(num_experts, dim_in, dim_hidden)
        b1 = torch.empty(num_experts, dim_hidden)
        W2 = torch.empty(num_experts, dim_hidden, dim_in)
        b2 = torch.empty(num_experts, dim_in)

        std = 1 / math.sqrt(self.dim_in)
        W1.uniform_(-std, std)
        b1.uniform_(-std, std)
        W2.uniform_(-std, std)
        b2.uniform_(-std, std)

        self.W1 = nn.Parameter(W1)
        self.b1 = nn.Parameter(b1)
        self.W2 = nn.Parameter(W2)
        self.b2 = nn.Parameter(b2)

    def forward(self, x):
        # x, weights, experts_indices = input_and_weights
        # batch, context_length, _ = x.shape
        # experts_mask = torch.zeros( (batch, context_length, self.num_experts), device = x.device, dtype = int) # x.shape[:-1] = batch, context_length

        # experts_mask.scatter_(-1, experts_indices, torch.ones_like(experts_indices, device = x.device))
        a = torch.einsum('bcd,ndh->bcnh', x, self.W1) + self.b1  # pass x to every expert
        z = F.relu(a)
        y = torch.einsum('bcnh,nhd->bcnd', z, self.W2) + self.b2
        return y


class GatingNetwork(nn.Module):
    def __init__(self, dim_in, num_experts, top_k, utilization_factor=1e-2):
        super(GatingNetwork, self).__init__()
        self.dim_in = dim_in
        self.num_experts = num_experts
        self.top_k = top_k
        self.Wg = nn.Linear(dim_in, num_experts, bias=False)
        self.Wnoise = nn.Linear(dim_in, num_experts, bias=False)
        self.utilization_factor = utilization_factor

    def forward(self, x):
        noise = F.softplus(self.Wnoise(x))
        noise *= torch.randn_like(noise).to(x.device)
        logits = self.Wg(x)
        logits += noise
        mask = torch.full_like(logits, -float('inf')).to(x.device)
        selected_logits, selected_indices = torch.topk(logits, self.top_k, dim=-1)
        mask.scatter_(-1, selected_indices, selected_logits)
        weights = F.softmax(mask, dim=-1)
        return weights, self.utilization_loss(weights)

    def utilization_loss(self, weights):
        importance = weights.reshape(-1, self.num_experts).sum(dim=0)
        square_cv = importance.var(correction=0) / importance.mean().pow(2)
        return self.utilization_factor * square_cv

class MoE(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_experts, top_k):
        super(MoE, self).__init__()
        # no need for dropout because it's already sparse?
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.gating = GatingNetwork(dim_in, num_experts, top_k)
        self.experts = Experts(dim_in, dim_hidden, num_experts)
        self.utilization_loss = 0 # REMEMBER TO CLEAR THIS AFTER EACH WEIGHT UPDATE!!!
    def forward(self, x):
        weights, loss = self.gating(x)
        self.utilization_loss += loss
        expert_results = self.experts(x)
        return torch.einsum('bcn,bcnd->bcd', weights, expert_results)
        # this implementation probably activates all the parameters, so no computational speed up. But that's not important for this RQ