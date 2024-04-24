import torch
import torch.nn as nn

from torch.utils.data import Dataset
from spikingjelly.activation_based import neuron, learning, functional

class EventVPR(nn.Module):
    def __init__(self, n_in, n_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Input shape should be [B, C, W, H] with C = 2, ON & OFF Channel
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, n_out, bias=False),
            neuron.LIFNode(tau=2.)
        )

    def forward(self, x):
        return self.net(x)
    
    def spikeBinning():
        pass

    def timeBinning():
        pass

class eventDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.tbins = []
        self.sbins = []

    def __len__(self) -> int:
        return len(self.tins) + len(self.sbins)

    def __getitem__(self, index) -> any:
        return super().__getitem__(index)

if __name__=="__main__":
    net = EventVPR(n_in=1,n_out=2)