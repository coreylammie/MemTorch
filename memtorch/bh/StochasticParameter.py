import torch
import memtorch


def StochasticParameter(mean, distribution=torch.distributions.normal.Normal, min=0, max=float('Inf'), function=True, **kwargs):
    """Method to model a stochatic parameter.

    Parameters
    ----------
    mean : float
        Mean value of the stochatic parameter.
    distribution : torch.distributions
        torch distribution.
    min : float
        Minimum value to sample.
    max: float
        Maximum value to sample.
    function : bool
        A sampled value is returned (True). A function to return a sampled value or mean is returned (False).

    Returns
    -------
    float or function
        A sampled value of the stochatic parameter, or a sample-value generator.
    """
    assert distribution == torch.distributions.normal.Normal, 'Currently, only torch.distributions.normal.Normal is supported.'
    assert kwargs['std'] is not None, 'std must be defined when distribution=torch.distributions.normal.Normal.'
    m = distribution(mean, kwargs['std'])
    def f(return_mean=False):
        """Method to return a sampled value or the mean of the stochatic parameters.

        Parameters
        ----------
        return_mean : bool
            Return the mean value of the stochatic parameter (True). Return a sampled value of the stochatic parameter (False).

        Returns
        -------
        float
            The mean value, or a sampled value of the stochatic parameter.
        """
        if return_mean:
            return mean
        else:
            return m.sample().clamp(min, max).item()

    if function:
        return f
    else:
        return f()

def unpack_parameters(local_args):
    """Method to sample from stochastic sample-value generators

    Parameters
    ----------
    local_args : locals()
        Local arguments with stochastic sample-value generators from which to sample from.

    Returns
    -------
    **
        locals() with sampled stochastic parameters.

    """
    if 'reference' in local_args:
        return_mean = True
    else:
        return_mean = False

    for arg in local_args:
        if callable(local_args[arg]) and '__' not in str(arg):
            local_args[arg] = local_args[arg](return_mean=return_mean)

    return Dict2Obj(local_args)


class Dict2Obj(object):
    """Class used to instantiate a object given a dictionary."""

    def __init__(self, dictionary):
        for key in dictionary:
            if key == '__class__':
                continue

            setattr(self, key, dictionary[key])
