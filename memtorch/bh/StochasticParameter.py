import copy
import inspect
import math

import torch

import memtorch


def StochasticParameter(
    distribution=torch.distributions.normal.Normal,
    min=0,
    max=float("Inf"),
    function=True,
    **kwargs
):
    """Method to model a stochastic parameter.

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
    assert issubclass(
        distribution, torch.distributions.distribution.Distribution
    ), "Distribution is not in torch.distributions."
    for arg in inspect.signature(distribution).parameters.values():
        if arg.name not in kwargs and arg.name != "validate_args":
            raise Exception("Argument %s is required for %s" % (arg.name, distribution))

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


def unpack_parameters(local_args, r_rel_tol=None, r_abs_tol=None, resample_threshold=5):
    """Method to sample from stochastic sample-value generators

    Parameters
    ----------
    local_args : locals()
        Local arguments with stochastic sample-value generators from which to sample from.
    r_rel_tol : float
        Relative threshold tolerance.
    r_abs_tol : float
        Absolute threshold tolerance.
    resample_threshold : int
        Number of times to resample r_off and r_on when their proximity is within the threshold tolerance before raising an exception.

    Returns
    -------
    **
        locals() with sampled stochastic parameters.
    """
    assert (
        r_rel_tol is None or r_abs_tol is None
    ), "r_rel_tol or r_abs_tol must be None."
    assert (
        type(resample_threshold) == int and resample_threshold >= 0
    ), "resample_threshold must be of type int and >= 0."
    if "reference" in local_args:
        return_mean = True
    else:
        return_mean = False

    local_args_copy = copy.deepcopy(local_args)
    for arg in local_args:
        if callable(local_args[arg]) and "__" not in str(arg):
            local_args[arg] = local_args[arg](return_mean=return_mean)

    args = Dict2Obj(local_args)
    if hasattr(args, "r_off") and hasattr(args, "r_on"):
        resample_idx = 0
        r_off_generator = local_args_copy["r_off"]
        r_on_generator = local_args_copy["r_on"]
        while True:
            if r_abs_tol is None and r_rel_tol is not None:
                if not math.isclose(args.r_off, args.r_on, rel_tol=r_rel_tol):
                    break
            elif r_rel_tol is None and r_abs_tol is not None:
                if not math.isclose(args.r_off, args.r_on, abs_tol=r_abs_tol):
                    break
            else:
                if not math.isclose(args.r_off, args.r_on):
                    break

            if callable(r_off_generator) and callable(r_on_generator):
                args.r_off = copy.deepcopy(r_off_generator)(return_mean=return_mean)
                args.r_on = copy.deepcopy(r_on_generator)(return_mean=return_mean)
            else:
                raise Exception(
                    "Resample threshold exceeded (deterministic values used)."
                )

            resample_idx += 1
            if resample_idx > resample_threshold:
                raise Exception("Resample threshold exceeded.")

    return args


class Dict2Obj(object):
    """Class used to instantiate a object given a dictionary."""

    def __init__(self, dictionary):
        for key in dictionary:
            if key == "__class__":
                continue

            setattr(self, key, dictionary[key])
