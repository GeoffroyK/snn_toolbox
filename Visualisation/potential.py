import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, layer

torch.manual_seed(42)

net = nn.Sequential(
    nn.Flatten(start_dim=0, end_dim=-1),
    nn.Linear(2, 1, bias=False),
    neuron.IFNode(v_threshold=1.,step_mode='s')
)

T = 100
#nn.init.constant_(net[1].weight.data,1)
in_spikes = torch.zeros((100,2,1))
in_spikes[0: T][:] = (torch.rand_like(in_spikes[0: T][:]) > 0.7).float()

potential = []
neuron1_pot = []
neuron2_pot = []
out_spikes = []

for i in range(T):
    out_spikes.append(net(in_spikes[i]).squeeze().detach().numpy())
    potential.append(net[2].v.squeeze().detach().numpy())
out_spikes = np.asarray(out_spikes)


# Plotting
threshold = net[2].v_threshold

t = np.arange(T)
in_spikes = in_spikes.numpy()
plt.figure()
plt.title("Neuronal Dynamics of the SNN")

plt.subplot(3,1,1)
plt.plot(potential, color='orange')
plt.plot([0., T], [threshold, threshold], "k--", label="threshold")
plt.ylabel("V", rotation=0)
plt.xlim(0, T)
plt.legend()
plt.title("Potential of the neuron")

plt.subplot(3,1,2)
plt.eventplot((t * in_spikes[:,0, 0])[in_spikes[:,0, 0] == 1.], lineoffsets=0, colors='green', label="input1")
plt.eventplot((t * in_spikes[:,1, 0])[in_spikes[:,1, 0] == 1.], lineoffsets=0, colors='red', label="input2")
plt.yticks([])
plt.xlim(0, T)
plt.legend()
plt.title("In spikes")

plt.subplot(3,1,3)
plt.eventplot((t * out_spikes[:])[out_spikes[:] == 1.], lineoffsets=0, colors='blue')
plt.xlabel("Timesteps")
plt.yticks([])
plt.xlim(0, T)
plt.title("Out spikes")

plt.tight_layout()
plt.show()