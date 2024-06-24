import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification, GPTNeoConfig
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification
from activations_config_neo import ActivationsGPTNeoConfig

class LearnableGELU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 1.0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.alpha = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, self.init)

    def forward(self, input: Tensor) -> Tensor:
        # Apply the learnable GELU function
        return 0.5 * self.alpha * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    
# PyTorch PReLU implementation broken, so we need to reinplement it ourselves. Using the formula from the documentation:
# PReLU(x)=max(0,x)+a∗min(0,x)
class CustomPReLU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.alpha = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.alpha, self.init)

    def forward(self, input: Tensor) -> Tensor:
        # Apply PReLU function
        return torch.max(torch.zeros_like(input),input) + self.alpha * torch.min(torch.zeros_like(input),input)


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

# Fix MLP for GPT-Neo with PReLU
class NeoPReLUMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = CustomPReLU(num_parameters=intermediate_size, init = 0.0)
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

#REF: http://arxiv.org/abs/1710.05941
class LearnableSwish(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.bias = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.bias, self.init)

    def forward(self, x):
        # Apply the learnable Swish function
        return x * F.silu(self.bias * x)


# Fix MLP for GPT-Neo with Swish
class NeoSwishMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * hidden_size
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size) # to match size for GeGLU
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = LearnableSwish(num_parameters=intermediate_size, init=1.0) # according to http://arxiv.org/abs/1710.05941
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

from efficient_kan.model import KAN
class KANtoMLP(nn.Module):

    def __init__(self, hidden_size, inner_dim):
        super().__init__()
        self.c_fc = KAN([hidden_size, inner_dim])
        self.c_proj = KAN([inner_dim, hidden_size])
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
    
class NeoNoActivationMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size) 
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
class ActivationsGPTNeoForCausalLM(GPTNeoForCausalLM):
    config_class = ActivationsGPTNeoConfig
    def __init__(self, config: ActivationsGPTNeoConfig):
        super().__init__(config)
        if config.custom_activation == 'learnable_gelu':
            print('here')
            for block in self.transformer.h:
                block.mlp = NeoLearnableGELUMLP(config.intermediate_size, config)
        elif config.custom_activation == 'prelu':
            for block in self.transformer.h:
                block.mlp = NeoPReLUMLP(config.intermediate_size, config)
        elif config.custom_activation == 'swish':
            for block in self.transformer.h:
                block.mlp = NeoSwishMLP(config.intermediate_size, config)
        elif config.custom_activation == 'kan':
            for block in self.transformer.h:
                block.mlp = KANtoMLP(config.hidden_size, config.intermediate_size)
        elif config.custom_activation == 'no_act':
            for block in self.transformer.h:
                block.mlp = NeoNoActivationMLP(config.intermediate_size, config)
     

class ActivationsGPTNeoForSequenceClassification(GPTNeoForSequenceClassification):
    config_class = ActivationsGPTNeoConfig
    def __init__(self, config:  ActivationsGPTNeoConfig):
        super().__init__(config)
        if config.custom_activation == 'learnable_gelu':
            for block in self.transformer.h:
                block.mlp = NeoLearnableGELUMLP(config.intermediate_size, config)
        elif config.custom_activation == 'prelu':
            for block in self.transformer.h:
                block.mlp = NeoPReLUMLP(config.intermediate_size, config)
        elif config.custom_activation == 'swish':
            for block in self.transformer.h:
                block.mlp = NeoSwishMLP(config.intermediate_size, config)
        elif config.custom_activation == 'kan':
            for block in self.transformer.h:
                block.mlp = KANtoMLP(config.hidden_size, config.intermediate_size)
        elif config.custom_activation == 'no_act':
            for block in self.transformer.h:
                block.mlp = NeoNoActivationMLP(config.intermediate_size, config)

AutoConfig.register('activations_gpt_neo', ActivationsGPTNeoConfig)
AutoModelForCausalLM.register(ActivationsGPTNeoConfig, ActivationsGPTNeoForCausalLM)
AutoModelForSequenceClassification.register(ActivationsGPTNeoConfig, ActivationsGPTNeoForSequenceClassification)