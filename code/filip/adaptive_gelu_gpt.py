import torch
from torch import nn, Tensor
import math

from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification, GPTNeoConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification


class LearnableGELU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 1.0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.beta = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.beta, self.init)

    def forward(self, input: Tensor) -> Tensor:
        # Apply the learnable GELU function
        return 0.5 * self.beta * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# Fix MLP for GPT-Neo with Learnable GELU
class NeoLearnableGELUMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = LearnableGELU(num_parameters=intermediate_size, init=1.0)
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class LearnableGeluGPTNeoForCausalLM(GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Override MLP with KAN in each transformer block
        for block in self.transformer.h:
            block.mlp = NeoLearnableGELUMLP(config.intermediate_size, config)

class LearnableGeluGPTNeoForSequenceClassification(GPTNeoForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        # Override MLP with KAN in each transformer block
        for block in self.transformer.h:
            block.mlp = NeoLearnableGELUMLP(config.intermediate_size, config)

AutoModelForCausalLM.register(GPTNeoConfig, LearnableGeluGPTNeoForCausalLM)
print('test')
AutoModelForSequenceClassification.register(GPTNeoConfig, LearnableGeluGPTNeoForCausalLM)