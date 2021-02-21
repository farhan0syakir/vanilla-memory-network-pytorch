import torch
from torch import nn
from memory import Memory


class MyModel(nn.Module):
    def __init__(self, input_size, out_size, memory_size, memory_feature):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.memory = Memory(memory_size, memory_feature)
        self.fc = nn.Linear(in_features = input_size + memory_feature, out_features = out_size)

    def forward(self, x):
        # to do, add your own parameter inside write 
        self.memory.write()
        batch_size, _  = x.size() 

        # this is how you read the memory state
        mem_state = self.memory.read()
 
        # for simplicity reason, lets make it able to fit the next layer naively, by prune it by the batch size
        x = torch.cat((x, mem_state[:batch_size]), 1)

        # in this example, after we read from memory, we forward it to linear
        x = self.fc(x)
        return x, mem_state