#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats


def anderson(x, axis=None):
    """Anderson-Darling test for data coming from a Normal distribution.

    The Anderson-Darling test is a modification of the Kolmogorov-
    Smirnov test `kstest` for the null hypothesis that a sample is
    drawn from a population that follows a particular distribution.

    Parameters
    ----------
    x : array_like
        array of sample data
    axis : None or int or tuple of ints, optional
        Axis or axes along which a test is performed.  The default,
        axis=None, will test all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.

    Returns
    -------
    statistic : float
        The Anderson-Darling test statistic
    significance_level : list
        The significance levels for the corresponding critical values
        in percents.  The function returns critical values for a
        differing set of significance levels depending on the
        distribution that is being tested against.

    """
    if axis is None:
        results = stats.anderson(x, dist="norm")
        return results[0], results[2]
    rsize = np.shape(x)[axis]
    ntst, pvals = np.empty(rsize), np.empty(rsize)
    for idx, values in enumerate(np.rollaxis(x, axis)):
        results = stats.anderson(values, dist="norm")
        ntst[idx], pvals[idx] = results[0], results[2]
    return ntst, pvals


def shapiro(x, axis=None):
    if axis is None:
        results = stats.shapiro(x)
        return results[0], results[1]
    rsize = np.shape(x)[axis]
    ntst, pvals = np.empty(rsize), np.empty(rsize)
    for idx, values in enumerate(np.rollaxis(x, axis)):
        results = stats.shapiro(values)
        ntst[idx], pvals[idx] = results[0], results[1]
    return ntst, pvals


def kstest(x, axis=None, *args, **kwargs):
    if axis is None:
        results = stats.kstest(x, cdf="norm", *args, **kwargs)
        return results.statistic, results.pvalue
    rsize = np.shape(x)[axis]
    ntst, pvals = np.empty(rsize), np.empty(rsize)
    for idx, values in enumerate(np.rollaxis(x, axis)):
        results = stats.kstest(values, cdf="norm", *args, **kwargs)
        ntst[idx], pvals[idx] = results.statistic, results.pvalue
    return ntst, pvals
