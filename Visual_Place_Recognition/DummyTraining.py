import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, surrogate, encoding, learning
from utils.image_processing import processSingleImage, get_train_test_datapath
from VPRModel import VPR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from VPRDataset import VPRDataset
from utils.plasticity import STDPLearner
import cv2
import pylab
from DummyDataset import DummyDataset

def f_pre(x):
    return torch.clamp(x, -1, 1.)

def f_post(x):
    return torch.clamp(x, -1, 1.)

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

if __name__=="__main__":
    net = VPR(tau=2.)

    encoder = encoding.PoissonEncoder()

    dataset = DummyDataset(dataset_path="../Dummy_Inputs/dummy_inputs/")

    train_data_loader = DataLoader(
        dataset = dataset,
        batch_size = 1, # Change ?
        shuffle = True,
        num_workers = 1, # +1 ?
        pin_memory = True
    )

    for x in train_data_loader:
        pass

    img = "../Dummy_Inputs/dummy_inputs/0.jpg"
    img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    img = dummy_image_processing(img)
    img_tensor = torch.from_numpy(img)
    spike_trains = encoder(img_tensor.float())

    learner = STDPLearner(net.layer[1], 2., 2. ,0.01, f_post=f_post, f_pre=f_pre)

    T = 100

    trace_pre = []
    trace_post = []
    w = []
    in_spike = []
    out_spike = []

    for x in train_data_loader:
        spike_trains = encoder(x.float())
        for t in range(T):
            s_pre = net.layer[0](spike_trains)
            s_pre = np.expand_dims(s_pre, axis=0)

            s_post = net(spike_trains).detach().numpy()
            out_spike.append(s_post)
            s_post = np.expand_dims(s_post, axis =0)

            s_pre = torch.from_numpy(s_pre) 
            s_post = torch.from_numpy(s_post)

            learner.single_step(s_pre, s_post)

            # Plot
            print(learner.trace_pre[0])
            trace_pre.append(learner.trace_pre[0][784//2].numpy())
            trace_post.append(learner.trace_post[0][784//2].numpy())
            w.append(net.layer[1].weight.detach().numpy()[t].mean())
            in_spike.append(net.layer[0](spike_trains).numpy())

fig = plt.figure(figsize=(10, 6))

spike1 = np.array(in_spike[0])

spike1.reshape((28,28))
plt.suptitle("STDP, lr=1.")

t = np.arange(0, T)
plt.subplot(5, 1, 1)
plt.plot(np.arange(len(w)), w, c='violet')
plt.ylabel('weights', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,2)
plt.eventplot(spike1, lineoffsets=0, colors='g')
plt.yticks([])
plt.ylabel('$S_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5, 1, 3)
plt.plot(np.arange(len(trace_pre)), trace_pre, c='orange')
plt.ylabel('$tr_{pre}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.subplot(5,1,4)
plt.plot(np.arange(len(trace_post)), trace_post)
plt.ylabel('$tr_{post}$', rotation=0, labelpad=10)
plt.xticks([])
plt.xlim(0, T)

plt.show()
