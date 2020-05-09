import os
import torch as T
from torch import nn


def format_weights_name(episode:int, reward:float, env_name:str):
    """Format weights name as [episode]_[reward]_[envname].pt

    Parameters
    ----------
    episode : int
        Episode id
    reward : float
        Episode reward
    env_name : str
        Environment name

    Returns
    -------
    weights_name : str
        Formated weights name
    """
    weights_name = 'episode{}_reward{:.2f}_{}.pt'.format(episode, 
                                                         reward, 
                                                         env_name)
    return weights_name


def save_weights(module: nn.Module, weights_file:str='weights.pt', 
                 path:str='./'):
    """Save weights of module as .pt file

    Parameters
    ----------
    module : nn.Module
        Torch module or agent for which to save weights
    weights_file_path : str
        File path to save the weights (should end in .pt)
    path : str
        Path to leave the weights
    """
    # Getting weights
    weights = module.state_dict()
    
    # Building path
    path = os.path.join('./logs', path)
    weights_file = os.path.join(path, weights_file)
    T.save(weights, weights_file)
    return weights_file


def load_weights(model: nn.Module, weights_file:str='weights.pt'):
    """Loads model or agent weights

    Parameters
    ----------
    model : nn.Module
        Torch module or agent for which to load the weights
    weights_file_path : str
        File path to the weights to load (should end in .pt)
    """
    weights = T.load(weights_file)
    model.load_state_dict(weights)