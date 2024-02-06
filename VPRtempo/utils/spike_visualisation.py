'''
Tool for STDP visualisation using matplotlib.

@GeoffroyK
'''

import numpy as np
import matplotlib.pyplot as plt

'''
What we need for visualisation:
pre_spike: 1D - Vector
post_spike: 1D - Vector
trace_post: 1D - Vector
trace_pre: 1D - Vector
weights : 1D - Vector, (list of weights)
Output: plt figure
'''
def plot_stdp_learning(pre_spike, post_spike, trace_post, trace_pre, weights, display=False):
    plt.figure()#figsize=(15,4))
    #plt.style.use('seaborn')
    # Event plot has key = 0, continous = 1
    figList = {
        "pre spikes":  pre_spike,
        "pre trace": trace_pre,
        "post_spikes": post_spike,
        "post_traces": trace_post,
        "weights": weights
    }

    figIndex = 1

    T = len(pre_spike)
    t = np.arange(0, T)
    plt.title("STDP Visualisation")

    # Plot events
    for fig in figList.items():
        if "spikes" in fig[0]: # Event plot
            plt.subplot(5,1,figIndex)    
            plt.eventplot(t * fig[1], lineoffsets=0, linewidths=0.5)
            plt.yticks([])
            plt.ylabel(fig[0], rotation=0, labelpad=28)
            plt.xticks([])
            plt.xlim(0, T)
        else: # Continuous plot
            plt.subplot(5,1,figIndex)    
            plt.plot(t, fig[1])
            if not "weights" in fig[0]:
                plt.yticks([])
                plt.xticks([])
            plt.ylabel(fig[0], rotation=0, labelpad=28)
            plt.xlim(0, T)
        figIndex += 1

    if display:
        plt.show()

    return fig

if __name__=="__main__":
    pass

