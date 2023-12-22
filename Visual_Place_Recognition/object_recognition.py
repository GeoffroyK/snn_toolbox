import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, surrogate, encoding, learning
from utils.image_processing import processSingleImage, get_train_test_datapath
from utils.spike_visualisation import plot_stdp_learning
import matplotlib.pyplot as plt
from VPRDataset import VPRDataset
from utils.plasticity import STDPLearner
import cv2

torch.manual_seed(1)

# SNN Class
class VPR(nn.Module):
    def __init__(self, tau, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Flatten(start_dim=0, end_dim=-1),
            nn.Linear(10 * 10, 2, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=1.)
        )

    def forward(self, x: torch.Tensor) -> nn.Sequential:
        return self.layer(x)


def f_weight(x):
    return torch.clamp(x, -1, 1)

def dummy_image_processing(img):
    # Open CV is kinda shit ngl
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
lr = 0.001
T = 100
w_min, w_max = -1., 1.
delta = 0.2 #Inhibition hyperparameter for WTA mechanism

net = VPR(tau=100.)
nn.init.constant_(net.layer[1].weight.data,0.3)

encoder = encoding.PoissonEncoder()
learner = STDPLearner(synapse=net.layer[1], tau_pre=tau_pre, tau_post=tau_post,learning_rate=lr, f_post=f_weight, f_pre=f_weight)


stimulus = []

stimulus.append(dummy_image_processing(cv2.imread('/home/geoffroy/horizontal_bar.jpg', cv2.IMREAD_GRAYSCALE)))
stimulus.append(dummy_image_processing(cv2.imread('/home/geoffroy/vertical_bar.jpg', cv2.IMREAD_GRAYSCALE)))

# Convert numpy arrays in torch tensors
for i in range(len(stimulus)):
    stimulus[i] = torch.from_numpy(stimulus[i])

ei = encoder(stimulus[0])


in_spike = []
out_spike = []
trace_pre = []
trace_post = []
weights = []
potential = []

for t in range(T*2):
    if t < T: #Present image 1    
        img = stimulus[0]
    else: #Present image 2
        img = stimulus[1]
    encoded_img = encoder(img)
    out_spike.append(net(encoded_img.float()).detach().numpy())
    
    s_pre = torch.from_numpy(np.expand_dims(net.layer[0](encoded_img.float()), axis=0))
    s_post = torch.from_numpy(np.expand_dims(out_spike[t], axis=0))

    learner.single_step(s_pre, s_post)

    potential.append(net.layer[2].v.detach().numpy())
    weights.append(net.layer[1].weight.detach().numpy().mean())

    # Soft Winner take all mechanism that retrieve a voltage delta for each other non-spiking neurons 
    spike_index = 0
    for spike in out_spike[t]:
        if spike:
            print("Spike on neuron", spike_index, " at timestep", t)
            current_index = 0
            for i in net.layer[2].v.detach().numpy():
                if current_index != spike_index: #Inhib only other non-spiking neurons
                    net.layer[2].v[current_index]  = net.layer[2].v[current_index] - delta
                current_index += 1
            break
        spike_index += 1


out_spike = np.asarray(out_spike)
potential = np.asarray(potential)
weights = np.asarray(weights)

t = np.arange(2 * T)
threshold = net.layer[2].v_threshold

plt.figure()
plt.subplot(1,2,1)
plt.title("Stiumulus 1")
plt.imshow(stimulus[0].numpy(), cmap="gray")
plt.subplot(1,2,2)
plt.title("Stiumulus 2")
plt.imshow(stimulus[1].numpy(), cmap="gray")

plt.figure()
plt.subplot(3,1,1)
plt.eventplot(t*out_spike[:,0],lineoffsets=0, linewidths=0.8, colors='r', label="neuron1")
plt.yticks([])
plt.xlim(0, 2*T)
plt.title("Neuron 1 activity")

plt.subplot(3,1,2)
plt.eventplot(t*out_spike[:,1],lineoffsets=0, linewidths=0.8, colors='b', label="neuron2")
plt.yticks([])
plt.xlim(0, 2*T)
plt.title("Neuron 2 activity")

plt.subplot(3,1,3)
plt.plot(potential[:,0], color='r')
plt.plot(potential[:,1], color='b')
plt.plot([0., 2*T], [threshold, threshold], "k--", label="threshold")
plt.xlim(0, 2*T)
plt.title("Membrane potential of both neurons")
plt.legend()
plt.tight_layout()

plt.figure()
plt.plot(weights)

plt.show()