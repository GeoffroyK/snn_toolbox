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


def f_pre(x):
    return torch.clamp(x, -1, 1.)

def f_post(x):
    return torch.clamp(x, -1, 1.)


if __name__=="__main__":
    net = VPR(tau=2.)
    net = net.float()
    nn.init.constant(net.layer[1].weight.data, 0.1)
    # Use Poisson encoder to transform the image in a spike-train (probability based on the pixel intensity)
    encoder = encoding.PoissonEncoder()

    print(net.layer[2])
    # TODO Refractor as args later 
    start_epoch = 0
    epoch = 200
    tau_pre = 2.
    tau_post = 2.
    T = 100
    w_min, w_max = -1., 1.

    # TODO change weight function
    def f_weight(x):
        return torch.clamp(x, -1, 1.)

    # STDP learner    
    learner = STDPLearner(net.layer[1], 2., 2. ,0.01, f_post=f_post, f_pre=f_pre)

    # TODO change .format to arg selected dataset
    trainingDataset = ['/media/geoffroy/T7/VPRSNN/data/{}/'.format('nordland')]  
    train_data_path, test_data_path = get_train_test_datapath(trainingDataset)
    dt = VPRDataset(train_data_path, 'train')

    train_data_loader = DataLoader(
        dataset = dt,
        batch_size = 1, # Change ?
        shuffle = True,
        num_workers = 1, # +1 ?
        pin_memory = True
    )

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []

    for epoch in range(1):
        net.train()
        for x, y in train_data_loader:
            '''x = x.to('cpu')
            encoded_img = encoder(x)
            '''
            for t in range(T):
                encoded_img = encoder(x)
                entry_signal = net.layer[0](encoded_img.float())
                entry_signal = np.expand_dims(entry_signal, axis=0)


                out = net(encoded_img.float()).detach().numpy()
                out = np.expand_dims(out,axis=0)

                entry_signal = torch.from_numpy(entry_signal)
                out = torch.from_numpy(out)

                learner.single_step(entry_signal, out)

        weight.append(net.layer[1].weight.data.clone().numpy())
        trace_post.append(learner.trace_post)

    #plt.plot(np.arange(len(weight)), weight)
    plt.show()

    # img = processSingleImage("/media/geoffroy/T7/VPRSNN/data/nordland/fall/images-00001.png", 28, 28, 7)

    # img = torch.tensor(img)
    # encoded_img = encoder(img)
    
    # out_spike = []
    # trace_pre = []
    # trace_post = []
    # weight = []

    # with torch.no_grad():
    #     for t in range(T):
    #         encoded_img = encoder(img)
    #         learner.step(on_grad=False)
    #         out_spike.append(net(encoded_img.float()))
    #         weight.append(net.layer[1].weight.data.clone())
    #         '''trace_pre.append(learner.trace_pre)
    #         trace_post.append(learner.trace_post)'''
        
    # #out_spike = torch.stack(out_spike)   # [T, batch_size, N_out]
    # #trace_pre = torch.stack(trace_pre)   # [T, batch_size, N_in]
    # #trace_post = torch.stack(trace_post) # [T, batch_size, N_out]
    # weight = torch.stack(weight)         # [T, N_out, N_in]

    # t = torch.arange(0, T).float()
    
    # in_spike = encoded_img[:, 0]
    # #out_spike = out_spike[:, 0, 0]
    # #trace_pre = trace_pre[:, 0, 0]
    # #trace_post = trace_post[:, 0, 0]
    # weight = weight[:, 0, 0]

    # cmap = plt.get_cmap('tab10')
    # plt.subplot(5, 1, 1)
    # #plt.eventplot((in_spike * t)[in_spike == 1], lineoffsets=0, colors=cmap(0))
    # plt.xlim(-0.5, T + 0.5)
    # plt.ylabel('$s[i]$', rotation=0, labelpad=10)
    # plt.xticks([])
    # plt.yticks([])
    # '''
    # plt.subplot(5, 1, 2)
    # plt.plot(t, trace_pre, c=cmap(1))
    # plt.xlim(-0.5, T + 0.5)
    # plt.ylabel('$tr_{pre}$', rotation=0)
    # plt.yticks([trace_pre.min().item(), trace_pre.max().item()])
    # plt.xticks([])
    
    # plt.subplot(5, 1, 3)
    # plt.eventplot((out_spike * t)[out_spike == 1], lineoffsets=0, colors=cmap(2))
    # plt.xlim(-0.5, T + 0.5)
    # plt.ylabel('$s[j]$', rotation=0, labelpad=10)
    # plt.xticks([])
    # plt.yticks([])
    
    # plt.subplot(5, 1, 4)
    # plt.plot(t, trace_post, c=cmap(3))
    # plt.ylabel('$tr_{post}$', rotation=0)
    # plt.yticks([trace_post.min().item(), trace_post.max().item()])
    # plt.xlim(-0.5, T + 0.5)
    # plt.xticks([])
    # '''
    # plt.subplot(5, 1, 5)
    # plt.plot(t, weight, c=cmap(4))
    # plt.xlim(-0.5, T + 0.5)
    # plt.ylabel('$w[i][j]$', rotation=0)
    # plt.yticks([weight.min().item(), weight.max().item()])
    # plt.xlabel('time-step')
    
    # plt.gcf().subplots_adjust(left=0.18)
    
    # plt.show()