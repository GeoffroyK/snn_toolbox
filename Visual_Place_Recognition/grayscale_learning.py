import torch

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.activity_visualisation import raster_plot, raster_plot_latency
from utils.image_processing import processSingleImage
from spikingjelly.visualizing import plot_one_neuron_v_s
from spikingjelly.activation_based import neuron, layer, encoding, learning, functional

'''
Parameters
'''
lif_decay = 5.
T = 80
threshold = 0.3
tau_pre = 60.
tau_post = 20.
lr = 0.009
epochs = 5
w_mean = 0.5
delta_inh = 1.

w_before = None

class SNNLayer(nn.Module):
    def __init__(self, lif_decay, threshold, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lif_decay = lif_decay
        self.threshold = threshold

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2, bias=False),
            neuron.LIFNode(tau=lif_decay, v_threshold=threshold)
        )

    def forward(self, x):
        return self.layer(x)
    
def f_weight(x):
    return torch.clamp(x, 0, 1)

def stdp_training(net, stdp_learner, optimizer, encoder, training_data):
    '''
    STDP Training with grayscaled, patches images of the nordland dataset
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on : {device}')
 
    pbar = tqdm(total=epochs * 2,
                desc=f"Training of model with STDP",
                position=0)
            
    nn.init.normal_(net.layer[1].weight.data, mean=w_mean) 

    w_before = net.layer[1].weight.data.detach()
    weight = []
    weight.append(net.layer[1].weight.detach())

    selective_neuron = {}
    for _ in range(epochs):
        net.train()
        
        for label in training_data:
            optimizer.zero_grad()
            img = training_data[label]
            img = img.to(device)

            label = label.to(device)
            
            inhibited_neurons = []
            winner = 0

            for t in range(encoder.T):
                encoded_img = encoder(img)
                out_spike = net(encoded_img).detach()
            
                # Create inhibition list to make a Winner-Take-All inhibiton           
                if 1. in out_spike:
                  winner = torch.argmax(out_spike)
                if label not in selective_neuron and winner not in selective_neuron.values():
                    selective_neuron[label] = winner
                if selective_neuron.__len__() > 0 and label in selective_neuron:
                    for idx in range(len(net.layer[-1].v[0].detach())):
                        if idx != selective_neuron[label]:
                            inhibited_neurons.append(idx)

                # Prevent the neuron from spiking with a negative fixed potential
                for neuron in inhibited_neurons:
                    net.layer[-1].v[0][neuron] = -10
                
                net.layer[1].weight.data = torch.clamp(net.layer[1].weight.data, 0, 1)
                stdp_learner.step(on_grad=True)
                optimizer.step()

            pbar.update(1)
        
            encoder.reset()
            # Reset membrane potential between images
            functional.reset_net(net)
            stdp_learner.reset()

    # Close progress bar
    pbar.close()
    print(selective_neuron)
    return net, w_before

def init_network_and_train(training_data) -> nn.Module:
    '''
    Init the SNN Layer and train on the specified dataset.
    '''
    # Init network and learner 
    net = SNNLayer(lif_decay, threshold)

    stdp_learner = learning.STDPLearner(step_mode='s',
                                        synapse=net.layer[1],
                                        sn=net.layer[-1],
                                        tau_pre=tau_pre,
                                        tau_post=tau_post,
                                        f_pre=f_weight,
                                        f_post=f_weight)

    stdp_optimizer = torch.optim.SGD(net.layer.parameters(), lr=lr, momentum=0.)

    # Spikes train are created with a temporal encoder to create a spike depending on the pixel intensity with 1 / value
    encoder = encoding.LatencyEncoder(T) # Quantized over 255 time bins (the input is in grayscale) TODO Maybe reducing the time bins can upgrade the recognition
    
    
    net, w_before = stdp_training(net, stdp_learner, stdp_optimizer, encoder, training_data)
   
    return net, w_before

def plot_spiking_timing(net, img, title):

    functional.reset_net(net)

    plt.figure(title)

    encoder = encoding.LatencyEncoder(T)
    
    v_list = []
    s_list = []
    
    net.eval()
    with torch.no_grad():
        for t in range(encoder.T):
            encoded_img = encoder(img)
            s_list.append(net(encoded_img).numpy().squeeze(0))
            v_list.append(net.layer[-1].v.numpy().squeeze(0))

    
    s_list = np.array(s_list)
    v_list = np.array(v_list)
    
    if np.mean(s_list[:,0]) > np.mean(s_list[:,1]):
        color = ["green", "black"]
    elif np.mean(s_list[:,0]) < np.mean(s_list[:,1]):
        color = ["black", "green"]
    else:
        color = ["black", "black"]

    plt.subplot(3,1,1)
    spike_times = [i for i, x in enumerate(s_list[:,0]) if x == 1]
    plt.eventplot(spike_times,lineoffsets=0, linewidths=1., colors='r', label="neuron1")
    plt.yticks([])
    plt.xlim(-0.5, T-1)
    plt.text(0.5,0.5,f"Fire rate : {np.mean(s_list[:,0]):.4f}", color=color[0])
    plt.title("Neuron 1 activity")

    plt.subplot(3,1,2)
    spike_times = [i for i, x in enumerate(s_list[:,1]) if x == 1]
    plt.eventplot(spike_times,lineoffsets=0, linewidths=1., colors='b', label="neuron2")
    plt.yticks([])
    plt.xlim(-0.5, T-1)
    plt.text(0.5,0.5,f"Fire rate : {np.mean(s_list[:,1]):.4f}", color=color[1])
    plt.title("Neuron 2 activity")

    plt.subplot(3,1,3)
    plt.plot(v_list[:,0], color='r', label="Neuron 1")
    plt.plot(v_list[:,1], color='b', label="Neuron 2")
    plt.plot([0., T-1], [threshold, threshold], "k--", label="threshold")
    plt.xlim(-0.5, T-1)
    plt.title("Membrane potential of both neurons")
    plt.legend()
    plt.tight_layout()


if __name__=="__main__":
    stimulus = []
    test = []

    img = processSingleImage("/media/geoffroy/T7/VPRSNN/data/nordland/fall/images-00001.png", 28, 28, 7)
    img = torch.rand((28,28))

    test.append(img)
    #img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    stimulus.append(img)

    img = processSingleImage("/media/geoffroy/T7/VPRSNN/data/nordland/fall/images-03513.png", 28, 28, 7)
    test.append(img)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    stimulus.append(img)
    
    ### Inverted image:
    # img = 1 - stimulus[0] 
    # stimulus.append(img)
    ###

    training_data = {torch.tensor([0]): stimulus[0], torch.tensor([1]): stimulus[1]}

    net = SNNLayer(lif_decay, threshold)
    nn.init.normal_(net.layer[1].weight.data, mean=w_mean) 
    net.layer[1].weight.data = torch.clamp(net.layer[1].weight.data, 0, 1)

    imgs = list(training_data.values())


    index = 0
    for label in training_data:
        plot_spiking_timing(net, training_data[label], title="Before Training on image " + str(index+1))
        index += 1

    net, w_before = init_network_and_train(training_data)
    index = 0
    for label in training_data:
        plot_spiking_timing(net, training_data[label], title="After Training on image " + str(index+1))
        index += 1

    plt.figure("Raster plot image 1")    
    raster_plot_latency(imgs[0], bins=T)


    
    plt.figure("Raster plot image 2")
    raster_plot_latency(imgs[1], bins=T)

    plt.figure("Weight matrix")

    if w_before is not None:
        plt.subplot(2,2,1)
        plt.title("W Neuron 1 before training")
        plt.imshow(w_before[0].reshape(28,28))

        plt.subplot(2,2,2)
        plt.title("W Neuron 2 before training")
        plt.imshow(w_before[1].reshape(28,28))


        plt.subplot(2,2,3)
        plt.title("W Neuron 1 after training")
        plt.imshow(net.layer[1].weight[0].data.detach().reshape(28,28))

        plt.subplot(2,2,4)
        plt.title("W Neuron 2 after training")
        plt.imshow(net.layer[1].weight[1].data.detach().reshape(28,28))
        plt.tight_layout()

    
    plt.figure("Interpolation of weights")
    plt.imshow((net.layer[1].weight[0].data.detach() * net.layer[1].weight[1].data.detach()).reshape(28,28))

    plt.show()

