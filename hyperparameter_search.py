import optuna
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, learning, surrogate, encoding, functional
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate

class SNNLayer(nn.Module):
    def __init__(self, decay_lif, threshold, *args, **kwargs):
        super(SNNLayer, self).__init__(*args, **kwargs)

        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 2, bias=False),
            neuron.LIFNode(tau=decay_lif, v_threshold=threshold, surrogate_function=surrogate.ATan())
        )

    def forward(self, x):
        return self.layer(x)

def f_weight(x):
    return torch.clamp(x, 0, 1)

def generate_dummy_input():
    '''
    Generate two fake places for learning purpose.
    Input 1 = Horizontal line
    Input 2 = Vertical Line 

    TODO if working fine with fake input change to real input from the nordland dataset.
    '''
    place1 = torch.zeros((28,28))
    place1[13] = 1
    place1 = place1.unsqueeze(0)

    place2 = torch.zeros((28,28))
    place2 [:,13] = 1
    place2 = place2.unsqueeze(0)
    
    training_data = {torch.tensor([0]): place1, torch.tensor([1]): place2}

    return training_data

def train_layer(net, T, tau_pre, tau_post, w_mean, epochs):
    '''
    STDP training of a SNN layer with the specified hyperparameters.
    '''

    # Fixed learning rate for now TODO change it to a generated one
    lr = 10e-3

    # Define STDP learner.
    stdp_learner = learning.STDPLearner(step_mode='s',
                                        synapse=net.layer[1],
                                        sn=net.layer[-1],
                                        tau_pre=tau_pre,
                                        tau_post=tau_post,
                                        f_pre=f_weight,
                                        f_post=f_weight)
    stdp_optimizer = torch.optim.SGD(net.layer.parameters(), lr=lr, momentum=0.)

    # Select learning device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Init weights of the network
    nn.init.normal_(net.layer[1].weight.data, mean=w_mean)

    # Generate training data
    training_data = generate_dummy_input()

    # Init encoder (pixel -> spike following Poisson with pixel intensity)
    encoder = encoding.PoissonEncoder()

    # Train the network
    for _ in range(epochs):
        net.train()
        
        for label in training_data:
            stdp_optimizer.zero_grad()
            img = training_data[label]
            img = img.to(device)

            label = label.to(device)
            label_onehot = F.one_hot(label, 2).float()

            for t in range(T):
                encoded_img = encoder(img)
                out_spike = net(encoded_img).detach()
                # TODO add inhibition
                stdp_learner.step(on_grad=True)
                stdp_optimizer.step()

            functional.reset_net(net)
        # Reset membrane potential between images
        functional.reset_net(net)
        functional.detach_net(net)
        stdp_learner.reset()      
    return net

def firing_rate_2_score(firing_rate):
    '''
    Return a score (int) depending on the recognition of places

    We should have a neuron with a firing rate of 1. and the other one of 0.
    If the firing rate array is [0,0] or [1,1] the learning has failed and thus the score is 0.
    Else the score should be one if [1,0] or [0,1] and 1
    '''
    if firing_rate.max() == 0:
        return 0
    if np.array_equal(firing_rate, np.ones((firing_rate.shape[0], firing_rate.shape[1]))):
        return 0
    else:
        return abs(firing_rate[0] - firing_rate[1])

def score_learning(net):
    '''
    Create a score of the network for predicting learned places.
    '''

    T = 20 
    encoder = encoding.PoissonEncoder()

    data = list(generate_dummy_input().values())
    place1 = data[0]
    place2 = data[1]
    score = 0

    for _ in range(10):
        functional.reset_net(net)
        with torch.no_grad():
            # Test with image 1
            s_list = []

            for t in range(T):
                encoded_img = encoder(place1)
                s_list.append(net(encoded_img))

            s_list = torch.cat(s_list)
            spikes = np.array(s_list)
            spikes = spikes.T
            firing_rate = np.mean(spikes, axis=1, keepdims=True)
            score += firing_rate_2_score(firing_rate)
    
            # Test with image 2
            functional.reset_net(net)
            s_list = []

            for t in range(T):
                encoded_img = encoder(place2)
                s_list.append(net(encoded_img))

            s_list = torch.cat(s_list)
            spikes = np.array(s_list)
            spikes = spikes.T
            firing_rate2 = np.mean(spikes, axis=1, keepdims=True)
            score += firing_rate_2_score(firing_rate2)

            # If each neurons respond to a specific place, score++
            if score > 0 and not np.array_equal(firing_rate, firing_rate2):
                score += 1
            else:
                score -= 1

    return score
    
def objective(trial):
    '''
    Define an objective function to be maximised.
    '''
    # Neuron parameters
    threshold = trial.suggest_float('threshold', 1., 100.)
    decay_lif = trial.suggest_float('decay_lif', 1., 100.)

    # Learning hyperparameters
    T = trial.suggest_int('T', 1, 1000)
    tau_pre = trial.suggest_float('tau_pre', 1., 10.)
    tau_post = trial.suggest_float('tau_post', 1. ,10.)
    w_mean = trial.suggest_float('w_mean', 0., 1.)
    epochs = trial.suggest_int('epochs', 1, 10)

    # Learning process
    net = SNNLayer(decay_lif, threshold)
    net = train_layer(net, T, tau_pre, tau_post, w_mean, epochs)
    score = score_learning(net)
    # Evaluation to maximise

    return score

if __name__=="__main__":
    study = optuna.create_study(
        direction='maximize',
        study_name='stdp_parameters',
        storage="sqlite:///db.sqlite3"
    )
    study.optimize(objective, n_trials=200)
    print(f"Best value: {study.best_value} (params: {study.best_params})")
