import torch
from my_model import MyModel
from torch import nn, optim

#init data
batch_size = 100
memory_size = 256
memory_feature = 5
input_size = 10
output_size = 12

data = torch.rand((batch_size, input_size)) # example input, input feature = 10
labels = torch.rand((batch_size, output_size)) # example ground truth, output feature = 12

#init model
model = MyModel(input_size, output_size, memory_size, memory_feature)

# init train 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
model.train()
torch.set_grad_enabled(True)

# forward pass
out, mem_state = model(data)
print(out.size(),mem_state.size())
loss = criterion(out, labels)
print(loss)

# Backward pass
optimizer.zero_grad()
loss.backward()
optimizer.step()