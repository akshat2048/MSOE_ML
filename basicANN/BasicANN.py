import torch.nn as nn

class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(224 * 224 * 1, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 2))
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        return x