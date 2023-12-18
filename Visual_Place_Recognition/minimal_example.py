import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, surrogate, encoding, learning, layer
from utils.image_processing import processSingleImage, get_train_test_datapath
from utils.spike_visualisation import plot_stdp_learning
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from VPRDataset import VPRDataset
from utils.plasticity import STDPLearner

torch.manual_seed(42)

# SNN Class
class VPR(nn.Module):
    def __init__(self, tau, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            #nn.Flatten(start_dim=0, end_dim=-1),
            layer.Linear(1,1, bias=False),
            neuron.LIFNode(tau=2., v_threshold=0.5)
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        return self.layer(x)
    
def f_weight(x):
    return torch.clamp(x, -1, 1)

start_epoch = 0
epoch = 200
tau_pre = 5.
tau_post = 5.
lr = 0.01
T = 1000
w_min, w_max = -1., 1.

net = VPR(tau=1.1)

nn.init.constant_(net.layer[0].weight.data,0.75)

encoder = encoding.PoissonEncoder()
img = processSingleImage("/media/geoffroy/T7/VPRSNN/data/nordland/fall/images-00001.png", 28, 28, 7)
img = torch.tensor(img)

learner = STDPLearner(synapse=net.layer[0], tau_pre=tau_pre, tau_post=tau_post,learning_rate=lr, f_post=f_weight, f_pre=f_weight)

s_pre = torch.zeros([T, 1, 1])
s_post  = torch.zeros([T, 1, 1])

trace_pre = []
trace_post = []
w = []

for t in range(T):
    s_pre[t] = encoder(img)[13][13]
    s_post[t] = net(s_pre[t]).detach()
    learner.single_step(s_pre[t], s_post[t])

    trace_pre.append(learner.trace_pre.item())
    trace_post.append(learner.trace_post.item())
    w.append(net.layer[0].weight.item())

fig = plt.figure(figsize=(10, 6))
plt.suptitle("STDP, lr=1.")

s_pre = s_pre[:, 0].numpy()
s_post = s_post[:, 0].numpy()
t = np.arange(0, T)
plt.subplot(5, 1, 1)
plt.eventplot((t * s_pre[:, 0])[s_pre[:, 0] == 1.], lineoffsets=0, colors='r')
plt.yticks([])
plt.ylabel('$S_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(5, 1, 2)
plt.plot(t, trace_pre, c='orange')
plt.ylabel('$tr_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5, 1, 3)
plt.eventplot((t * s_post[:, 0])[s_post[:, 0] == 1.], lineoffsets=0, colors='g')
plt.yticks([])
plt.ylabel('$S_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(5, 1, 4)
plt.plot(t, trace_post)
#plt.axhline(y = 0.5, color = 'r', linestyle = 'dashed')     
plt.ylabel('$tr_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)
plt.subplot(5, 1, 5)
plt.plot(t, w, c='purple')
plt.ylabel('$w$', rotation=0, labelpad=10)
plt.xlim(0, T)

plt.show()
