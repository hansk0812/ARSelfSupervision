import torch
from torch import nn

class Net(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.out = out
        self.layer = nn.Linear(inp, out)

    def forward(self, x):
        return self.layer(x)

class NetFF(Net):
    def set_output_size(self, out):
        self.out_size = out

    def forward(self, x):
        self.layer_ff = nn.Linear(self.out, self.out_size)
        f = super().forward(x)
        return self.layer_ff(f)

if __name__ == "__main__":

    net = Net(5, 10)

    print (net.forward(torch.rand(8,5)).shape)

    net.__class__ = NetFF
    net.set_output_size(20)

    print (net.forward(torch.rand(8,5)).shape)

    print (net.state_dict())
