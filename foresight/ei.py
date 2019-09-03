# -*- coding: UTF-8 -*-

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


def H(x, dim=0):
    r"""Compute the Shannon information entropy of x.
    
    Given a tensor x, compute the shannon entropy along one of its axes. If
    x.shape == (N,) then returns a scalar (0-d tensor). If x.shape == (N, N)
    then information can be computed along vertical or horizontal axes by
    passing arguments dim=0 and dim=1, respectively.
    
    Note that the function does not check that the axis along which
    information will be computed represents a valid probability distribution.
    
    Args:
        x (torch.tensor) containing probability distribution
        dim (int) dimension along which to compute entropy
    
    Returns:
        (torch.tensor) of a lower order than input x
    """
    r = x * torch.log2(x)
    r[r != r] = 0
    return -torch.sum(r, dim=dim)


def soft_norm(W):
    r"""Turns 2x2 matrix W into a transition probability matrix.
    
    The weight/adjacency matrix of an ANN does not on its own allow for EI
    to be computed. This is because the out weights of a given neuron are not
    a probability distribution (they do not necessarily sum to 1). We therefore
    must normalize them. 
    
    Applies a softmax function to each row of matrix
    W to ensure that the out-weights are normalized.
    
    Args:
        W (torch.tensor) of shape (2, 2)
        
    Returns:
        (torch.tensor) of shape (2, 2)
    """
    return F.softmax(W, dim=1)


def lin_norm(W):
    r"""Turns 2x2 matrix W into a transition probability matrix.
    
    Applies a relu across the rows (to get rid of negative values), and normalize
    the rows based on their arithmetic mean.
    
    Args:
        W (torch.tensor) of shape (2, 2)
        
    Returns:
        (torch.tensor) of shape (2, 2)
    """
    W = F.relu(W)
    row_sums = torch.sum(W, dim=1)
    row_sums[row_sums == 0] = 1
    row_sums = row_sums.reshape((-1, 1))
    return W / row_sums


def sig_norm(W):
    r"""Turns 2x2 matrix W into a transition probability matrix.
    
    Applies logistic function on each element and normalize
    across rows.
    
    Args:
        W (torch.tensor) of shape (2, 2)
        
    Returns:
        (torch.tensor) of shape (2, 2)
    """
    W = torch.sigmoid(W)
    row_sums = torch.sum(W, dim=1).reshape((-1, 1))
    return W / row_sums


def linear_create_matrix(module, in_shape, out_shape):
    r"""Returns 2d connectivity matrix of an nn.Linear layer.

    This matrix has shape: (input_activations, output_activations).
    Therefore each row contains the output weights of a neuron. To compute
    the effective information, normalize across the rows of the returned matrix.

    Args:
        module (nn.Module): layer in feedforward network
        in_shape (tuple): shape of module input
        out_shape (tuple): shape of module output

    Returns:
        2d torch.tensor
    """
    with torch.no_grad():
        W = module.weight.t()
        assert W.shape[0] == in_shape[-1]
        assert W.shape[1] == out_shape[-1]
        return W


def conv2d_create_matrix(module, in_shape, out_shape):
    r"""Returns 2d connectivity matrix of an nn.Conv2d layer.

    This matrix has shape: (input_activations, output_activations).
    Therefore each row contains the output weights of a neuron. To compute
    the effective information, normalize across the rows of the returned matrix.

    Args:
        module (nn.Module): layer in feedforward network
        in_shape (tuple): shape of module input
        out_shape (tuple): shape of module output

    Returns:
        2d torch.tensor
    """
    with torch.no_grad():
        assert len(in_shape) == 4 and len(out_shape) == 4
        p_h, p_w = module.padding
        samples, channels, in_height, in_width = in_shape
        W_in_shape = (samples, channels, in_height + 2*p_h, in_width + 2*p_w)
        W = torch.zeros(out_shape[1:] + W_in_shape[1:]) # [1:] to ignore batch size
        weight = module.weight
        s_h, s_w = module.stride
        k_h, k_w = module.kernel_size
        for c_out in range(out_shape[1]):
            for h in range(0, out_shape[2]):
                for w in range(0, out_shape[3]):
                    in_h, in_w = h*s_h, w*s_w
                    W[c_out][h][w][:, in_h:in_h+k_h, in_w:in_w+k_w] = weight[c_out]
        ins = reduce(lambda x, y: x*y, in_shape[1:])
        outs = reduce(lambda x, y: x*y, out_shape[1:])
        if p_h != 0:
            W = W[:, :, :, :, p_h:-p_h, :] # get rid of vertical padding
        if p_w != 0:
            W = W[:, :, :, :, :, p_w:-p_w] # get rid of horizontal padding
        return W.reshape((outs, ins)).t()


def avgpool2d_create_matrix(module, in_shape, out_shape):
    r"""Returns 2d connectivity matrix of an nn.AvgPool2d layer.

    This matrix has shape: (input_activations, output_activations).
    Therefore each row contains the output weights of a neuron. To compute
    the effective information, normalize across the rows of the returned matrix.

    Args:
        module (nn.Module): layer in feedforward network
        in_shape (tuple): shape of module input
        out_shape (tuple): shape of module output

    Returns:
        2d torch.tensor
    """
    with torch.no_grad():
        assert module.padding == 0
        assert len(in_shape) == 4 and len(out_shape) == 4
        W = torch.zeros(out_shape[1:] + in_shape[1:]) # [1:] to ignore batch size
        if type(module.stride) is tuple:
            assert len(module.stride) == 2, "stride tuple must have 2 elements for 2d Pool"
            s_h, s_w = module.stride
        else:
            s_h = s_w = module.stride
        if type(module.kernel_size) is tuple:
            assert len(module.kernel_size) == 2, "kernel_size tuple must have 2 elements for 2d Pool"
            k_h, k_w = module.kernel_size
        else:
            k_h = k_w = module.kernel_size
        weight = 1 / (k_h * k_w)
        for c_out in range(out_shape[1]):
            for h in range(0, out_shape[2]):
                for w in range(0, out_shape[3]):
                    in_h, in_w = h*s_h, w*s_w
                    W[c_out][h][w][:, in_h:in_h+k_h, in_w:in_w+k_w] = weight
        ins = reduce(lambda x, y: x*y, in_shape[1:])
        outs = reduce(lambda x, y: x*y, out_shape[1:])
        return W.reshape((outs, ins)).t()


r"""
    The modules for which a create_matrix() function has been defined. 
    The create_matrix() function generates a 2d connectivity matrix for
    each layer. If the network is feedforward, with no skip-connections,
    then determinism and degeneracy can be computed using each layer's
    connectivity matrix, without computing the whole network connectivity
    matrix.
"""
VALID_MODULES = {
    nn.Linear: linear_create_matrix,
    nn.Conv2d: conv2d_create_matrix,
    nn.AvgPool2d: avgpool2d_create_matrix
}


def get_shapes(model, input):
    r"""Get a dictionary {module: (in_shape, out_shape), ...} for modules in `model`.

    Because PyTorch uses a dynamic computation graph, the number of activations
    that a given module will return is not intrinsic to the definition of the module,
    but can depend on the shape of its input. We therefore need to pass data through
    the network to determine its connectivity. 

    This function passes `input` into `model` and gets the shapes of the tensor 
    inputs and outputs of each child module in model, provided that they are
    instances of VALID_MODULES.

    Args:
        model (nn.Module): feedforward neural network
        input (torch.tensor): a valid input to the network

    Returns:
        Dictionary {`nn.Module`: tuple(in_shape, out_shape)}
    """

    shapes = {}
    hooks = []
    
    def register_hook(module):
        def hook(module, input, output):
            shapes[module] = (tuple(input[0].shape), tuple(output.shape))
        if type(module) in VALID_MODULES:
            hooks.append(module.register_forward_hook(hook))

    model.apply(register_hook)
    model(input)
    for hook in hooks:
        hook.remove()
    return shapes


def determinism(model, input=None, shapes=None, norm=lin_norm, device='cpu'):
    r"""Compute the determinism of neural network `model`.

    Determinism is the average entropy of the outweights of each node (neuron)
    in the graph: 

    .. math::
        \text{determinism} = \langle H(W^\text{out}) \rangle

    If a `shapes` argument is provided, then `input` will not be used and
    need not be provided. If no `shapes` argument is provided, then an 
    `input` argument must be provided to build its computation graph.

    Args:
        model (nn.Module): neural network defined with PyTorch
        input (torch.tensor): an input for the model (needed to build computation graph)
        shapes (dict): dictionary containing mappings from child modules
            to their input and output shape (created by get_shapes() function)
        norm (func): function to normalize the out weights of each neuron.
        device: (str): must be 'cpu' or 'cuda'

    Returns:
        determinism (float)
    """
    if shapes is None:
        if input is None:
            raise Exception("Missing argument `input` needed to compute shapes.")
        shapes = get_shapes(model, input)
    H_sum = 0
    N = 0
    for module, (in_shape, out_shape) in shapes.items():
        create_matrix = VALID_MODULES[type(module)]
        W = create_matrix(module, in_shape, out_shape)
        if norm:
            W = norm(W)
        H_sum += torch.sum(H(W, dim=1)).item()
        N += W.shape[0]
    return (H_sum / N)


def degeneracy(model, input=None, shapes=None, norm=lin_norm, device='cpu'):
    r"""Compute the degeneracy of neural network `model`.

    Degeneracy is the entropy of the cumulative, normalized in-weights for each
    neuron in the graph: 

    .. math::
        \text{degeneracy} = H( \langle W^\text{out} \rangle )

    If a `shapes` argument is provided, then `input` will not be used and
    need not be provided. If no `shapes` argument is provided, then an 
    `input` argument must be provided to build its computation graph.

    Args:
        model (nn.Module): neural network defined with PyTorch
        input (torch.tensor): an input for the model (needed to build computation graph)
        shapes (dict): dictionary containing mappings from child modules
            to their input and output shape (created by get_shapes() function)
        norm (func): function to normalize the out weights of each neuron.
        device: (str): must be 'cpu' or 'cuda'

    Returns:
        degeneracy (float)
    """
    if shapes is None:
        if input is None:
            raise Exception("Missing argument `input` needed to compute shapes.")
        shapes = get_shapes(model, input)
    in_weights = torch.zeros((0,)).to(device)
    total_weight = 0
    for module, (in_shape, out_shape) in shapes.items():
        create_matrix = VALID_MODULES[type(module)]
        W = create_matrix(module, in_shape, out_shape)
        if norm:
            W = norm(W)
        in_weights = torch.cat((in_weights, torch.sum(W, dim=0)))
        total_weight += torch.sum(W).item()
    return H(in_weights / total_weight).item()


def ei(model, input=None, shapes=None, norm=lin_norm, device='cpu'):
    r"""Compute the effective information of neural network `model`.

    Effective information is a useful measure of the information
    contained in the weighted connectivity structure of a network. It 
    is used in theoretical neuroscience to study emergent structure
    in networks. It is defined by:

    .. math::
        \text{EI} = \text{determinism} - \text{degeneracy}

    explicitly:

    .. math::
        \text{EI} = \langle H(W^\text{out}) \rangle - H( \langle W^\text{out} \rangle )

    Which is equal to the average KL-divergence between the normalized
    out-weights of the neurons and the distribution of in-weights across
    the neurons in the network.

    If a `shapes` argument is provided, then `input` will not be used and
    need not be provided. If no `shapes` argument is provided, then an 
    `input` argument must be provided to build its computation graph.

    Args:
        model (nn.Module): neural network defined with PyTorch
        input (torch.tensor): an input for the model (needed to build computation graph)
        shapes (dict): dictionary containing mappings from child modules
            to their input and output shape (created by get_shapes() function)
        norm (func): function to normalize the out weights of each neuron.
        device: (str): must be 'cpu' or 'cuda'

    Returns:
        ei (float)
    """
    if shapes is None:
        if input is None:
            raise Exception("Missing argument `input` needed to compute shapes.")
        shapes = get_shapes(model, input)
    return degeneracy(model, shapes=shapes, norm=norm, device=device) \
            - determinism(model, shapes=shapes, norm=norm, device=device)



