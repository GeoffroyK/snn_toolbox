import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, encoding, learning

class VPR(nn.Module):
    def __init__(self, tau, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(28 * 28, 100, bias=False),
            neuron.LIFNode(tau=tau)
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        return self.layer(x)


if __name__=="__main__":
    net = VPR(tau=2.)
    net.layer[1]