import torch
from torch import nn

class Memory(nn.Module):
    def __init__(self, memory_size, memory_feature):
        super(Memory, self).__init__()
        self.memory_size = memory_size
        self.memory_feature = memory_feature
        self.memory = torch.Tensor(memory_size, memory_feature)

    def read(self):
        # to do make your own read
        return self.memory

    def write(self):
        # to do make your own write
        self.memory = torch.rand((self.memory_size, self.memory_feature))