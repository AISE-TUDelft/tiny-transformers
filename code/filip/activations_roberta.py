import torch
from torch import nn, Tensor
import torch.nn.functional as F
import math

from transformers import RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification
from activations_config_roberta import ActivationsRobertaConfig

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


# Fix MLP for RoBERTa with Learnable GELU
class RobertaLearnableGELUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = LearnableGELU(num_parameters=config.intermediate_size, init=1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

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
    
# Fix MLP for RoBERTa with PReLU
class RobertaPReLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = CustomPReLU(num_parameters=config.intermediate_size, init=0.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

#REF: http://arxiv.org/abs/1710.05941
class LearnableSwish(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.beta = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.beta, self.init)

    def forward(self, x):
        # Apply the learnable Swish function
        return x * F.silu(self.beta * x)


# Fix MLP for RoBERTa with Swish
class RobertaSwishMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = LearnableSwish(num_parameters=config.intermediate_size, init=1.0) # according to http://arxiv.org/abs/1710.05941

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

from efficient_kan.model import KAN
class KANtoMLP(nn.Module):

    def __init__(self, hidden_size, inner_dim):
        super().__init__()
        self.c_fc = KAN([hidden_size, inner_dim])
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        return x
    
class RobertaNoActivationMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return hidden_states
    
class ActivationsRobertaForMaskedLM(RobertaForMaskedLM):
    config_class = ActivationsRobertaConfig
    def __init__(self, config: ActivationsRobertaConfig):
        super().__init__(config)
        if config.custom_activation == 'learnable_gelu':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaLearnableGELUMLP(config)
        elif config.custom_activation == 'prelu':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaPReLUMLP(config)
        elif config.custom_activation == 'swish':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaSwishMLP(config)
        elif config.custom_activation == 'kan':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = KANtoMLP(config.hidden_size, config.intermediate_size)
        elif config.custom_activation == 'no_act':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaNoActivationMLP(config)
     

class ActivationsRobertaForSequenceClassification(RobertaForSequenceClassification):
    config_class = ActivationsRobertaConfig
    def __init__(self, config:  ActivationsRobertaConfig):
        super().__init__(config)
        if config.custom_activation == 'learnable_gelu':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaLearnableGELUMLP(config)
        elif config.custom_activation == 'prelu':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaPReLUMLP(config)
        elif config.custom_activation == 'swish':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaSwishMLP(config)
        elif config.custom_activation == 'kan':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = KANtoMLP(config.hidden_size, config.intermediate_size)
        elif config.custom_activation == 'no_act':
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaNoActivationMLP(config)

AutoConfig.register('activations_roberta', ActivationsRobertaConfig)
AutoModelForMaskedLM.register(ActivationsRobertaConfig, ActivationsRobertaForMaskedLM)
AutoModelForSequenceClassification.register(ActivationsRobertaConfig, ActivationsRobertaForSequenceClassification)