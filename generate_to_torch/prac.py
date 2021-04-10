import torch
from utils import NeuralNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nn = NeuralNetwork()
model = nn.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=1e-3)
print(optim.param_groups)
for param_group in optim.param_groups:
    print(type(param_group))
    print(param_group.keys())
    param_group["lr"] = 1e-5
    print(param_group["lr"])

optim.param_groups[0]["lr"] = 1e-6
for param_group in optim.param_groups:
    print(type(param_group))
    print(param_group.keys())
    print(param_group["lr"])
