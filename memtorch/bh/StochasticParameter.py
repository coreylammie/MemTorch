import torch
import memtorch
import inspect
import math


def StochasticParameter(distribution=torch.distributions.normal.Normal, min=0, max=float('Inf'), function=True, **kwargs):
    """Method to model a stochatic parameter.

    Parameters
    ----------
    distribution : torch.distributions
        torch distribution.
    min : float
        Minimum value to sample.
    max: float
        Maximum value to sample.
    function : bool
        A sampled value is returned (False). A function to return a sampled value or mean is returned (True).

    Returns
    -------
    float or function
        A sampled value of the stochatic parameter, or a sample-value generator.
    """
    assert issubclass(distribution, torch.distributions.distribution.Distribution), 'Distribution is not in torch.distributions.'
    for arg in inspect.signature(distribution).parameters.values():
        if arg.name not in kwargs and arg.name != 'validate_args':
            raise Exception('Argument %s is required for %s' % (arg.name, distribution))

    m = distribution(**kwargs)
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
            return m.mean
        else:
            return m.sample().clamp(min, max).item()

    if function:
        return f
    else:
        return f()

def unpack_parameters(local_args, failure_threshold=5):
    """Method to sample from stochastic sample-value generators

    Parameters
    ----------
    local_args : locals()
        Local arguments with stochastic sample-value generators from which to sample from.

    failure_threshold : int
        Failure threshold to raise an Exception if r_off and r_on are indistinguishable.

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

    args = Dict2Obj(local_args)
    if hasattr(args, 'r_off') and hasattr(args, 'r_on'):
        assert type(failure_threshold) == int and failure_threshold > 0, 'Invalid failure_threshold value.'
        failure_idx = 0
        while True:
            failure_idx += 1
            if failure_idx > failure_threshold:
                raise Exception('r_off and r_on values are indistinguishable.')

            if not math.isclose(args.r_off, args.r_on):
                break

    return args


class Dict2Obj(object):
    """Class used to instantiate a object given a dictionary."""

    def __init__(self, dictionary):
        for key in dictionary:
            if key == '__class__':
                continue

            setattr(self, key, dictionary[key])
