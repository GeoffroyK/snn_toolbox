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
import cv2

torch.manual_seed(42)

# SNN Class
class VPR(nn.Module):
    def __init__(self, tau, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            layer.Linear(10*10,2, bias=False),
            neuron.LIFNode(tau=2., v_threshold=1.)
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        return self.layer(x)
    
def f_weight(x):
    return torch.clamp(x, -1, 1)

def dummy_image_processing(img):
    img = img / 255.
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] != 1 or 0:
                if img[i][j] > 0.1:
                    img[i][j] = 1
                else:
                    img[i][j] = 0
    return img

start_epoch = 0
epoch = 200
tau_pre = 5.
tau_post = 5.
lr = 0.01
T = 1000
w_min, w_max = -1., 1.

net = VPR(tau=1.1)

nn.init.normal_(net.layer[1].weight.data,mean=0)


encoder = encoding.PoissonEncoder()

img = cv2.imread('/home/geoffroy/vertical_bar.jpg', cv2.IMREAD_GRAYSCALE)
img = dummy_image_processing(img)
# add some noise later because it's boring otherwise
img = torch.tensor(img)

learner = STDPLearner(synapse=net.layer[1], tau_pre=tau_pre, tau_post=tau_post,learning_rate=lr, f_post=f_weight, f_pre=f_weight)

s_pre = torch.zeros([T, 1, 100])
s_post  = torch.zeros([T, 1, 2])
in_spikes = []
trace_pre = []
trace_post = []
w = []

for t in range(T):
    encoded_img = encoder(img)
    in_spikes.append(encoded_img)

    s_pre[t] = net.layer[0](encoded_img.float())
    s_post[t] = net(s_pre[t]).detach()
    
    #stdp step
    learner.single_step(s_pre[t], s_post[t])

    # plottinh
    trace_pre.append(learner.trace_pre[0][0])
    trace_post.append(learner.trace_post)
    w.append(net.layer[1].weight[0][0].detach())

fig = plt.figure(figsize=(10, 6))
plt.suptitle("STDP, lr=1.")

s_pre = s_pre[:, 0].numpy()
s_post = s_post[:, 0].numpy()
t = np.arange(0, T)
plt.subplot(5, 1, 1)
plt.imshow(in_spikes[0], cmap="gray")
plt.yticks([])
plt.ylabel('$S_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.subplot(5, 1, 2)
plt.plot(torch.stack(trace_post).squeeze().detach().numpy().T, c='orange')
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
