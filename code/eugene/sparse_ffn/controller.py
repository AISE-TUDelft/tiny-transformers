import torch
import torch.nn as nn
import torch.nn.functional as F

class Controller(nn.Module):
    def __init__(self, dim_in, dim_lowrank, dim_hidden, num_blocks):
        super(Controller, self).__init__()
        self.dim_in = dim_in
        self.dim_lowrank = dim_lowrank
        self.dim_hidden = dim_hidden
        self.num_blocks = num_blocks
        assert self.dim_hidden % self.num_blocks == 0, "hidden vector must be divisible into N blocks"
        self.U = nn.Linear(dim_in, dim_lowrank, bias=False)
        self.V = nn.Linear(dim_lowrank, dim_hidden, bias=False)

    def forward(self, x):
        logits = self.V(self.U(x))
        original_shape = logits.shape
        logits = logits.reshape(*logits.shape[:-1], self.num_blocks, self.dim_hidden // self.num_blocks)
        if self.training:
            mask = F.gumbel_softmax(logits, tau=0.1, hard=True)
            return mask.reshape(original_shape)
        else:
            selected = torch.argmax(logits, dim=-1)
            mask = F.one_hot(selected, num_classes=self.dim_hidden // self.num_blocks)
            return mask.reshape(original_shape)

class ControllerFFN(nn.Module):
    def __init__(self, dim_in, dim_lowrank, dim_hidden, num_blocks):
        super(ControllerFFN, self).__init__()
        self.dim_in = dim_in
        self.dim_lowrank = dim_lowrank
        self.dim_hidden = dim_hidden
        self.num_blocks = num_blocks
        assert self.dim_hidden % self.num_blocks == 0, "hidden vector must be divisible into N blocks"
        self.controller = Controller(dim_in, dim_lowrank, dim_hidden, num_blocks)
        self.layer1 = nn.Linear(dim_in, dim_hidden)
        self.layer2 = nn.Linear(dim_hidden, dim_in)
    def forward(self, x):
        return self.layer2(self.controller(x) * F.relu(self.layer1(x)))