##############################################
# An update of spikingjelly's STDP
#
# spikingjelly version <= 0.0.0.12 (old)
# computes weight update at each timestep
# and directly apply the change without the
# need of a PyTorch optimizer
#
# includes codes for RRF2d layer
##############################################


from typing import Callable, Union

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, monitor, base


def stdp_linear_single_step(
        fc: nn.Linear, in_spike: torch.Tensor, out_spike: torch.Tensor,
        trace_pre: Union[float, torch.Tensor, None],
        trace_post: Union[float, torch.Tensor, None],
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if trace_pre is None:
        trace_pre = 0.

    if trace_post is None:
        trace_post = 0.

    weight = fc.weight.data
    trace_pre = trace_pre - trace_pre / tau_pre + in_spike  # shape = [batch_size, N_in]
    trace_post = trace_post - trace_post / tau_post + out_spike  # shape = [batch_size, N_out]

    # [batch_size, N_out, N_in] -> [N_out, N_in]

    delta_w_pre = -f_pre(weight) * (trace_post.unsqueeze(2) * in_spike.unsqueeze(1)).sum(0)  # sum(0) = sum over batches
    delta_w_post = f_post(weight) * (trace_pre.unsqueeze(1) * out_spike.unsqueeze(2)).sum(0)
    return trace_pre, trace_post, delta_w_pre + delta_w_post


def stdp_conv2d_single_step(
        conv: nn.Conv2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
        trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if conv.dilation != (1, 1):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv2d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv2d has not been implemented!'
        )

    stride_h = conv.stride[0]
    stride_w = conv.stride[1]

    if conv.padding == (0, 0):
        pass
    else:
        pH = conv.padding[0]
        pW = conv.padding[1]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for h in range(conv.weight.shape[2]):
        for w in range(conv.weight.shape[3]):
            h_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + h
            w_end = in_spike.shape[3] - conv.weight.shape[3] + 1 + w

            pre_spike = in_spike[:, :, h:h_end:stride_h, w:w_end:stride_w]  # shape = [batch_size, C_in, h_out, w_out]
            post_spike = out_spike  # shape = [batch_size, C_out, h_out, h_out]
            weight = conv.weight.data[:, :, h, w]  # shape = [batch_size_out, C_in]

            tr_pre = trace_pre[:, :, h:h_end:stride_h, w:w_end:stride_w]  # shape = [batch_size, C_in, h_out, w_out]
            tr_post = trace_post  # shape = [batch_size, C_out, h_out, w_out]

            delta_w_pre = - (f_pre(weight) *
                             (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                             .permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4]))
            delta_w_post = f_post(weight) * \
                           (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)) \
                               .permute([1, 2, 0, 3, 4]).sum(dim=[2, 3, 4])
            delta_w[:, :, h, w] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w


def stdp_conv1d_single_step(
        conv: nn.Conv1d, in_spike: torch.Tensor, out_spike: torch.Tensor,
        trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
        tau_pre: float, tau_post: float,
        f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
):
    if conv.dilation != (1,):
        raise NotImplementedError(
            'STDP with dilation != 1 for Conv1d has not been implemented!'
        )
    if conv.groups != 1:
        raise NotImplementedError(
            'STDP with groups != 1 for Conv1d has not been implemented!'
        )

    stride_l = conv.stride[0]

    if conv.padding == (0,):
        pass
    else:
        pL = conv.padding[0]
        if conv.padding_mode != 'zeros':
            in_spike = F.pad(
                in_spike, conv._reversed_padding_repeated_twice,
                mode=conv.padding_mode
            )
        else:
            in_spike = F.pad(in_spike, pad=(pL, pL))

    if trace_pre is None:
        trace_pre = torch.zeros_like(
            in_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    if trace_post is None:
        trace_post = torch.zeros_like(
            out_spike, device=in_spike.device, dtype=in_spike.dtype
        )

    trace_pre = trace_pre - trace_pre / tau_pre + in_spike
    trace_post = trace_post - trace_post / tau_post + out_spike

    delta_w = torch.zeros_like(conv.weight.data)
    for l in range(conv.weight.shape[2]):
        l_end = in_spike.shape[2] - conv.weight.shape[2] + 1 + l
        pre_spike = in_spike[:, :, l:l_end:stride_l]  # shape = [batch_size, C_in, l_out]
        post_spike = out_spike  # shape = [batch_size, C_out, l_out]
        weight = conv.weight.data[:, :, l]  # shape = [batch_size_out, C_in]

        tr_pre = trace_pre[:, :, l:l_end:stride_l]  # shape = [batch_size, C_in, l_out]
        tr_post = trace_post  # shape = [batch_size, C_out, l_out]

        delta_w_pre = - (f_pre(weight) *
                         (tr_post.unsqueeze(2) * pre_spike.unsqueeze(1))
                         .permute([1, 2, 0, 3]).sum(dim=[2, 3]))
        delta_w_post = f_post(weight) * \
                       (tr_pre.unsqueeze(1) * post_spike.unsqueeze(2)) \
                           .permute([1, 2, 0, 3]).sum(dim=[2, 3])
        delta_w[:, :, l] += delta_w_pre + delta_w_post

    return trace_pre, trace_post, delta_w


# def stdp_rrf2d_single_step(
#         rrf: RRF2d, in_spike: torch.Tensor, out_spike: torch.Tensor,
#         trace_pre: Union[torch.Tensor, None], trace_post: Union[torch.Tensor, None],
#         tau_pre: float, tau_post: float,
#         f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x
# ):
#     if rrf.dilation != (1, 1):
#         raise NotImplementedError(
#             'STDP with dilation != 1 for Conv2d has not been implemented!'
#         )

#     if rrf.padding == (0, 0):
#         pass
#     else:
#         pH = rrf.padding[0]
#         pW = rrf.padding[1]
#         if rrf.padding_mode != 'zeros':
#             in_spike = F.pad(
#                 in_spike, rrf._reversed_padding_repeated_twice,
#                 mode=rrf.padding_mode
#             )
#         else:
#             in_spike = F.pad(in_spike, pad=(pW, pW, pH, pH))

#     weight = rrf.weights.data  # [C_out, C_in * Kh * Kw, N_patches]

#     if trace_pre is None:
#         trace_pre = torch.zeros_like(
#             in_spike, device=in_spike.device, dtype=in_spike.dtype
#         )

#     if trace_post is None:
#         trace_post = torch.zeros_like(
#             out_spike, device=in_spike.device, dtype=in_spike.dtype
#         )

#     # compute next timestep for traces
#     trace_pre = trace_pre - trace_pre / tau_pre + in_spike
#     trace_post = trace_post - trace_post / tau_post + out_spike

#     # convert traces and spikes from image basis to patch basis (i.e. same as weights, eases computations)
#     trace_pre_patches = rrf.unfold(trace_pre)
#     in_spike_patches = rrf.unfold(in_spike)
#     trace_post_patches = trace_post.flatten(start_dim=-2, end_dim=-1)
#     out_spike_patches = out_spike.flatten(start_dim=-2, end_dim=-1)

#     # apply stdp
#     delta_w_pre = -f_pre(weight) * (trace_post_patches.unsqueeze(2) * in_spike_patches.unsqueeze(1)).sum(0)  # [C_out, C_in * Kh * Kw, N_patches]
#     delta_w_post = f_post(weight) * (trace_pre_patches.unsqueeze(1) * out_spike_patches.unsqueeze(2)).sum(0)  # [C_out, C_in * Kh * Kw, N_patches]

#     return trace_pre, trace_post, delta_w_pre + delta_w_post


class STDPLearner(base.MemoryModule):

    def __init__(self,
                 synapse: Union[nn.Conv2d, nn.Linear],#, RRF2d],
                 tau_pre: float, tau_post: float, learning_rate: float = 1.,
                 f_pre: Callable = lambda x: x, f_post: Callable = lambda x: x) -> None:
        super().__init__()

        if isinstance(synapse, nn.Linear):
            self.stdp_f_single_step = stdp_linear_single_step
        elif isinstance(synapse, nn.Conv2d):
            self.stdp_f_single_step = stdp_conv2d_single_step
        elif isinstance(synapse, nn.Conv1d):
            self.stdp_f_single_step = stdp_conv1d_single_step
        # elif isinstance(synapse, RRF2d):
        #     self.stdp_f_single_step = stdp_rrf2d_single_step
        else:
            raise NotImplementedError("STDPLearner compatible with torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2"
                                      f"and RRF2d; received {synapse}")

        self.synapse = synapse

        self.tau_pre = tau_pre
        self.tau_post = tau_post

        self.register_memory('trace_pre', None)
        self.register_memory('trace_post', None)

        self.learning_rate = learning_rate

        self.f_pre = f_pre
        self.f_post = f_post

    @torch.no_grad()
    def single_step(self, in_spike: torch.Tensor, out_spike: torch.Tensor):

        self.trace_pre, self.trace_post, delta_w = self.stdp_f_single_step(
            self.synapse, in_spike, out_spike,
            self.trace_pre, self.trace_post,
            self.tau_pre, self.tau_post,
            self.f_pre, self.f_post
        )

        self.synapse.weight += delta_w * self.learning_rate

    def stdp_multi_step(self, in_spike: torch.Tensor, out_spike: torch.Tensor):
        raise NotImplementedError


if __name__ == "__main__":

    import torch
    import torch.nn as nn
    from matplotlib import pyplot as plt
    import numpy as np

    torch.manual_seed(42)

    def f_pre(x):
        return torch.clamp(x, -1, 1.)

    def f_post(x):
        return torch.clamp(x, -1, 1.)


    fc = nn.Linear(1, 1, bias=False)
    tau_pre = 5.
    tau_post = 5.
    lr = 0.01

    stdp_learner = STDPLearner(fc, tau_pre, tau_post, lr, f_pre, f_post)

    trace_pre = []
    trace_post = []
    w = []
    T = 1024
    s_pre = torch.zeros([T, 1, 1])
    s_post = torch.zeros([T, 1, 1])
    s_pre[0: T // 2] = (torch.rand_like(s_pre[0: T // 2]) > 0.95).float()
    s_post[0: T // 2] = (torch.rand_like(s_post[0: T // 2]) > 0.9).float()

    s_pre[T // 2:] = (torch.rand_like(s_pre[T // 2:]) > 0.8).float()
    s_post[T // 2:] = (torch.rand_like(s_post[T // 2:]) > 0.95).float()

    print(s_pre[0])

    for t in range(T):
        print(s_pre[t])
    	# STDP weight update
        stdp_learner.single_step(s_pre[t], s_post[t])
        
        # plotting
        trace_pre.append(stdp_learner.trace_pre.item())
        trace_post.append(stdp_learner.trace_post.item())
        w.append(fc.weight.item())

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
    plt.ylabel('$tr_{post}$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.xlim(0, T)
    plt.subplot(5, 1, 5)
    plt.plot(t, w, c='purple')
    plt.ylabel('$w$', rotation=0, labelpad=10)
    plt.xlim(0, T)

    plt.show()

    print("ok")
