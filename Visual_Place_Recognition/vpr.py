import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, surrogate, encoding, learning
from utils.image_processing import processSingleImage, get_train_test_datapath
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from VPRDataset import VPRDataset
from utils.plasticity import STDPLearner

# SNN Class
class VPR(nn.Module):
    def __init__(self, tau, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(28 * 28, 5, bias=False),
            neuron.LIFNode(tau=2., v_threshold=0.8)
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        return self.layer(x)

def f_pre(x):
    return torch.clamp(x, -1, 1.)

def f_post(x):
    return torch.clamp(x, -1, 1.)

start_epoch = 0
epoch = 200
tau_pre = 5.
tau_post = 5.
lr = 0.001
T = 100
w_min, w_max = -1., 1.

net = VPR(tau=1.1)
encoder = encoding.PoissonEncoder()
learner = STDPLearner(synapse=net.layer[1], tau_pre=tau_pre, tau_post=tau_post,learning_rate=lr, f_post=f_post, f_pre=f_pre)

img = processSingleImage("/media/geoffroy/T7/VPRSNN/data/nordland/fall/images-00001.png", 28, 28, 7)
img = torch.tensor(img)
encoded_img = encoder(img)

in_spike = []
out_spike = []
trace_pre = []
trace_post = []
weights = []

# img = np.ones_like(img)
# img = torch.from_numpy(img)

for t in range(T):
    # Current spiking vector following a Poisson law
    encoded_img = encoder(img)
   
    # Pre and post spikes, after forward
    in_spike.append(encoded_img[0][0].numpy())
    out_spike.append(net(encoded_img.float()).detach().numpy())

    # Torch Input processing for STDP 
    s_pre = torch.from_numpy(np.expand_dims(net.layer[0](encoded_img.float()), axis=0))
    s_post = torch.from_numpy(np.expand_dims(out_spike[t], axis=0))

    # STDP step
    learner.single_step(s_pre, s_post)

    # Plotting 
    trace_pre.append(learner.trace_pre[0][0].numpy())
    trace_post.append(learner.trace_post[0][0].numpy())
    #weights.append(net.layer[1].weight[0][0].detach().numpy())
    weights.append(net.layer[1].weight[0].detach().numpy().mean())
    #print(net.layer[1].weight.shape)

### MATPLOTLIB ###

t = np.arange(0, T)
out_spike = np.array(out_spike)

neuron1 = out_spike[:,0]
neuron2 = out_spike[:,1]
neuron3 = out_spike[:,2]
neuron4 = out_spike[:,3]
neuron5 = out_spike[:,4]

plt.figure()
plt.tight_layout()
plt.subplot(4,2,1)
plt.eventplot(t * in_spike, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(4,2,2)
plt.plot(t, trace_pre, c="green")
plt.yticks([])
plt.ylabel('$T_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(4,2,3)
plt.eventplot(t * neuron1, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{out} N_1$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(4,2,4)
plt.plot(t, trace_post, c="green")
plt.yticks([])
plt.ylabel('$T_{post} N_1$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,5)
plt.title("Mean weights")
plt.plot(t, weights)
plt.ylabel('$Weight$', rotation=0, labelpad=10)
plt.xlim(0, T)

plt.figure()
plt.title("Output neurons activity")
plt.subplot(5,1,1)
plt.eventplot(t * neuron1, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{out} N_1$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,2)
plt.eventplot(t * neuron2, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{out} N_2$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,3)
plt.eventplot(t * neuron3, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{out} N_3$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,4)
plt.eventplot(t * neuron4, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{out} N_4$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,5)
plt.eventplot(t * neuron5, lineoffsets=0, linewidths=0.5, colors='r')
plt.yticks([])
plt.ylabel('$S_{out} N_5$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.show()