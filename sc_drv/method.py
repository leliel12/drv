#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018, Cabral, Juan; Luczywo, Nadia; Zanazi Jose Luis
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# =============================================================================
# DOCS
# =============================================================================

"""DRV method implementation

"""

__all__ = ["DRVProcess"]


# =============================================================================
# IMPORTS
# =============================================================================

import itertools as it

import numpy as np

from scipy import stats

import attr

import joblib

from skcriteria import norm
from skcriteria.madm import simple

from . import normtests, plot


# =============================================================================
# CONSTANTS
# =============================================================================

NORMAL_TESTS = {"shapiro": normtests.shapiro,
                "ks": normtests.kstest}


# =============================================================================
# STRUCTURATION FUNCTIONS
# =============================================================================


def nproduct_indexes(nproducts, climit):
    """Calculate the indexs of the products"""
    sst = np.sum((nproducts - np.mean(nproducts)) ** 2)
    ssw = np.sum((nproducts - np.mean(nproducts, axis=0)) ** 2)
    ssb = sst - ssw
    ssu = (nproducts.shape[0] - 1) / float(nproducts.shape[1] * 3)

    ivr = ssw / ssu
    inc = ivr <= climit

    resume = np.mean(nproducts, axis=0)

    return sst, ssw, ssb, ssu, ivr, inc, resume


def solve_nproducts(mtx):
    """Create the product (normalized) matrix"""
    rmtx = np.flip(mtx, axis=1)
    rcumprod = np.cumprod(rmtx, axis=1)
    wproducts = np.flip(rcumprod, axis=1)
    return norm.sum(wproducts, axis=1)


def subproblem(mtx, climit, ntest, ntest_kwargs, alpha):
    """Create and evaluate the product (normalized) matrix"""
    nproducts = solve_nproducts(mtx)

    sst, ssw, ssb, ssu, ivr, inc, resume = nproduct_indexes(nproducts, climit)

    n_sts, pvals = ntest(nproducts, axis=1, **ntest_kwargs)
    n_reject_h0 = pvals <= alpha

    return {
        "nproducts": nproducts,
        "sst": sst,
        "ssw": ssw,
        "ssb": ssb,
        "ssu": ssu,
        "ivr": ivr,
        "in_consensus": inc,
        "ntest_sts": n_sts,
        "ntest_pvals": pvals,
        "ntest_reject_h0": n_reject_h0,
        "resume": resume}


# =============================================================================
# AGGREGATION STAGE
# =============================================================================


def run_aggregator(idx, mtxs, criteria, weights, aggregator):
    """Helper to run the aggregator with joblib"""
    mtx = np.vstack(m[idx] for m in mtxs).T
    weight = 1 if weights is None else weights[idx]
    return aggregator.decide(mtx, criteria=criteria, weights=weight)


def rank_ttest_rel(agg_p, aidx, bidx):
    """Helper to run the t-test with joblib"""
    a_vals = np.array([r.e_.points[aidx] for r in agg_p])
    b_vals = np.array([r.e_.points[bidx] for r in agg_p])
    return stats.ttest_rel(a_vals, b_vals)


# =============================================================================
# DRV as FUNCTION
# =============================================================================


def drv(
    weights, abc, climit, ntest, ntest_kwargs, alpha, njobs, agg_only_consensus
):
    # PREPROCESS

    # determine numbers of parallel jobs
    njobs = joblib.cpu_count() if njobs is None else njobs

    # determine the normal test
    ntest = NORMAL_TESTS.get(ntest, ntest)
    ntest_kwargs = {} if ntest_kwargs is None else ntest_kwargs

    # number of participants & alternatives
    N, I = np.shape(abc[0])

    # number of criteria
    J = len(abc)

    # placeholder to store the results
    results = {"N_": N, "I_": I, "J_": J}

    # WEIGHTS
    if np.ndim(weights) > 1:
        wresults = subproblem(
            mtx=weights,
            climit=climit,
            alpha=alpha,
            ntest=ntest,
            ntest_kwargs=ntest_kwargs)
    else:
        wresults = {}

    # copy weights results to the global results
    results.update({
        "wmtx_": wresults.get("nproducts"),
        "wsst_": wresults.get("sst"),
        "wssw_": wresults.get("ssw"),
        "wssb_": wresults.get("ssb"),
        "wssu_": wresults.get("ssu"),
        "wivr_": wresults.get("ivr"),
        "wntest_sts_": wresults.get("ntest_sts"),
        "wntest_pvals_": wresults.get("ntest_pvals"),
        "wntest_reject_h0_": wresults.get("ntest_reject_h0"),
        "win_consensus_": wresults.get("in_consensus"),
        "weights_mean_": wresults.get("resume")})

    # ALTERNATIVES
    with joblib.Parallel(n_jobs=njobs) as jobs:
        wresults = jobs(
            joblib.delayed(subproblem)(
                amtx,
                climit=climit,
                alpha=alpha,
                ntest=ntest,
                ntest_kwargs=ntest_kwargs)
            for amtx in abc)

    # copy alt results to the global results
    results.update({
        "amtx_criteria_": tuple(r["nproducts"] for r in wresults),
        "asst_": np.hstack(r["sst"] for r in wresults),
        "assw_": np.hstack(r["ssw"] for r in wresults),
        "assb_": np.hstack(r["ssb"] for r in wresults),
        "assu_": np.hstack(r["ssu"] for r in wresults),
        "aivr_": np.hstack(r["ivr"] for r in wresults),
        "ain_consensus_": np.hstack(r["in_consensus"] for r in wresults),
        "antest_sts_": np.vstack(r["ntest_sts"] for r in wresults),
        "antest_pvals_": np.vstack(r["ntest_pvals"] for r in wresults),
        "antest_reject_h0_": np.vstack(
            r["ntest_reject_h0"] for r in wresults),
        "amtx_mean_": np.vstack(r["resume"] for r in wresults)})

    # CONSENSUS
    consensus = np.all(results["ain_consensus_"])
    if consensus and results["weights_mean_"] is not None:
        consensus = consensus and results["win_consensus_"]
    results["consensus_"] = consensus  # to global results

    # GLOBAL REJECT H0
    reject_h0 = np.any(results["antest_reject_h0_"])
    if not reject_h0 and results["wntest_reject_h0_"] is not None:
        reject_h0 = reject_h0 or np.any(results["wntest_reject_h0_"])
    results["ntest_reject_h0_"] = reject_h0

    # AGGREGATION
    if consensus or not agg_only_consensus:
        aggregator = simple.WeightedSum(mnorm="none", wnorm="none")

        criteria = [max] * J

        weights_mean = (
            1 if results["weights_mean_"] is None else results["weights_mean_"]
        )
        agg_m = aggregator.decide(
            results["amtx_mean_"].T, criteria=criteria, weights=weights_mean
        )

        with joblib.Parallel(n_jobs=njobs) as jobs:
            agg_p = jobs(
                joblib.delayed(run_aggregator)(
                    idx=idx,
                    mtxs=results["amtx_criteria_"],
                    criteria=criteria,
                    weights=results["wmtx_"],
                    aggregator=aggregator,
                )
                for idx in range(N)
            )
            agg_p = tuple(agg_p)

            # rank verification
            ttest_results = jobs(
                joblib.delayed(rank_ttest_rel)(
                    agg_p=agg_p, aidx=aidx, bidx=bidx
                )
                for aidx, bidx in it.combinations(range(I), 2)
            )

            rank_t, rank_p = np.empty(I), np.empty(I)
            for idx, r in enumerate(ttest_results):
                rank_t[idx] = r.statistic
                rank_p[idx] = r.pvalue
    else:
        agg_p, agg_m, rank_t, rank_p = None, None, None, None

    # to global results
    results["aggregation_criteria_"] = agg_p
    results["aggregation_mean_"] = agg_m
    results["rank_check_t_"] = rank_t
    results["rank_check_pval_"] = rank_p

    return results


# =============================================================================
# RESULT CLASS
# =============================================================================


@attr.s(frozen=True)
class DRVResult(object):
    """Result set of the DRV method.

    Parameters
    ----------

    ntest : str
        Normality-test. Test to check if the priorities established by group
        members must have a random behavior, represented by Normal
        Distribution.

    ntest_kwargs : dict or None
        Parameters for the normal test function.

    alpha : float
        significance. If the any p-value of n-test is less than `alpha`, we
        reject the null hypothesis of the normality tests.

    climit : float
        Consensus limit. Maximum value of the IVR to asume that the solution
        is stable.

        The Stability is verified using the normality analysis of priorities
        for each element of a sub-problem, or by using the IVR
        (Índice de Variabilidad Remanente, Remaining Variability Index)
        IVR <= ``climit`` are indicative of stability.

    N_ : int
        Number of participants

    I_ : int
        Number of alternatives

    J_ : int
        Number of criteria.

    consensus_ : bool
        If all the sub-problems are in consensus. In other words if
        every problem has ther IVR <= climit.

    ntest_reject_h0_ : bool
        True if any subproble reject one of their normality test H0 then this.

    weights_mean_ : array or None
        If the weight preference if provided, this attribute contains
        a array where every j-nth element is mean of the weights assigned by
        the participants to the j-nth criteria.

    wmtx_ : array or None
        If the weight preference if provided, this attribute contains
        a 2D array where every row is a weight assigned by a single
        participant.

    wsst_ : float or None
        Weights sub-problem Square Sum Total.
        If the weight preference if provided, this attribute contains
        the total sum of squares of the weight sub-problem. This value
        is calculated as
        `sum((wmtx_ - mean(wmtx_))**2))`.

    wssw_ : float or None
        Weights sub-problem Square-Sum Within.
        If the weight preference if provided, this attribute contains
        the sum of squares within criteria of the weight sub-problem, and
        represents the residual variability after a stage of analysis.
        This value is calculated as
        `sum((wmtx_ - mean_by_row(wmtx_))**2))`

    wssb_ : float or None
        Weights sub-problem Square-Sum Between.
        If the weight preference if provided, this attribute contains
        the sum of squares between criteria of the weight sub-problem,
        This value is calculated as `wsst_ - wssw_`.

    wssu_ : float or None
        Weights sub-problem Square-Sum of Uniform distribution.
        Corresponds to the uniform distribution and reflects a situation of
        complete disagreement within the group.

    wivr_ : float or None
        Weights sub-problem Índice de Variabilidad Remanente
        (Translation: Remaining Variability Index).
        If the weight preference if provided, this attribute contains
        Is a ratio of agreement calculates as `wssw_ / wssu_`.

    win_consensus_ : bool or None
        Weights sub-problem In Consensus.
        If the weight preference if provided, this attribute contains
        the weights sub-problem is in consensus. In other words if all
        the weight sub-problem `wivr_ <= climit`.

    wntest_sts_ : ndarray or None
        Weights Normal Test Statistics.
        If the weight preference if provided, this attribute contains an array
        with the normality test statistic by criteria.

    wntest_pvals_ : array or None
        Weights Normal Test P-value.
        If the weight preference if provided, this attribute contains an array
        with the normality test Normality test p-value by criteria. This
        values are useful if you have an small number of criteria to reinforce
        the assumption normality.

    wntest_reject_h0_ : array or None
        If the weight preference if provided, this attribute contains an array
        where the j-nth element is True if the normality test fails for
        the values of the criteria j-nth.

    amtx_criteria_ : tuple of arrays
        Alternatives matrix by criteria.
        A tuple where the j-nth element is a 2D array of preference of
        the `I_` alternatives by the criteria j.

    asst_ : array
        Alternatives by criteria sub-problems Square-Sum Within.
        Array where the j-nth element is the total sum of squares of the
        evaluation of the alternatives by the criteria j.
        Every element on this array  is calculated as
        `sum((amtx_criteria_[j] - mean(amtx_criteria_[j]))**2))`.

    assw_ : array
        Alternatives by criteria  sub-problems Square-Sum Within.
        Array where the j-nth element is the total sum of squares within
        of the evaluation of the alternatives by the criteria j, and
        represents the residual variability after a stage of analysis.
        Every element on this array  is calculated as
        `sum((amtx_criteria_[j] - mean_by_row(amtx_criteria_[j]))**2))`

    assb_ : array
        Alternatives by criteria sub-problems Square-Sum Between.
        Array where the j-nth element is the total sum of squares between
        of the evaluation of the alternatives by the criteria j.
        Every element on this array  is calculated as `asst_ - assw_`.

    assu_ : array
        Alternatives by criteria  sub-problems Square-Sum of Uniform
        distribution. Corresponds to the uniform distribution and reflects a
        situation of complete disagreement within the group.

    aivr_ : array
        Alternatives by criteria sub-problems Índice de Variabilidad Remanente
        (Translation: Remaining Variability Index).
        Array where the j-nth element a ratio of agreement of the alternatives
        by the criteria j. Is calculated as follows: `assw_ / assu_`.

    ain_consensus_ : array
        Alternatives by criteria sub-problems In Consensus.
        Array where the j-nth element is True if the alternatives
        by the criteria j are in consensus. In other words if
        `aivr_[j] <= climit`.

    amtx_mean_ : 2D array
        Alternative matrix.
        Created as a mean of all alternatives matrix by criteria.

    antest_sts_ : 2D array
        Alternatives by criteria sub-problems Normal Test Statistics.
        Array where the A_ij element contains the statistic of the normality
        test for the alternative i under the criteria j.

    antest_pvals_ : 2D array
        Alternatives by criteria sub-problems Normal Test Statistics.
        Array where the A_ij element contains the p-value of the normality
        test for the alternative i under the criteria j.

    antest_reject_h0_ : 2D array
        Alternatives by criteria sub-problems status of the null hypothesis.
        Array where the A_ij element contains the null hypotesys of the
        normality test for the alternative i under the criteria j must be
        rejected.

    aggregation_criteria_ : tuple
    aggregation_mean_ : skcriteria.madm.Decision
    rank_check_t_ : array or None
    rank_check_pval_ : array or None


    """

    ntest = attr.ib()
    ntest_kwargs = attr.ib()
    alpha = attr.ib()
    climit = attr.ib()

    N_ = attr.ib()
    I_ = attr.ib()
    J_ = attr.ib()

    consensus_ = attr.ib()
    ntest_reject_h0_ = attr.ib()

    wmtx_ = attr.ib(repr=False)
    wsst_ = attr.ib(repr=False)
    wssw_ = attr.ib(repr=False)
    wssb_ = attr.ib(repr=False)
    wssu_ = attr.ib(repr=False)
    wivr_ = attr.ib(repr=False)
    win_consensus_ = attr.ib(repr=False)
    weights_mean_ = attr.ib(repr=False)
    wntest_sts_ = attr.ib(repr=False)
    wntest_pvals_ = attr.ib(repr=False)
    wntest_reject_h0_ = attr.ib(repr=False)

    amtx_criteria_ = attr.ib(repr=False)
    asst_ = attr.ib(repr=False)
    assw_ = attr.ib(repr=False)
    assb_ = attr.ib(repr=False)
    assu_ = attr.ib(repr=False)
    aivr_ = attr.ib(repr=False)
    ain_consensus_ = attr.ib(repr=False)
    amtx_mean_ = attr.ib(repr=False)
    antest_sts_ = attr.ib(repr=False)
    antest_pvals_ = attr.ib(repr=False)
    antest_reject_h0_ = attr.ib(repr=False)

    aggregation_criteria_ = attr.ib(repr=False)
    aggregation_mean_ = attr.ib(repr=False)
    rank_check_t_ = attr.ib(repr=False)
    rank_check_pval_ = attr.ib(repr=False)

    plot = attr.ib(repr=False, init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "plot", plot.PlotProxy(self))

    @property
    def has_weights_(self):
        """True if the weight preference if provided so all the
        weight sub-problem attributes are available; otherwise the attributes
        are setted to None

        """
        return self.weights_mean_ is not None

    @property
    def data_(self):
        """Data object used in the aggregation mean or None"""
        if self.aggregation_mean_ is not None:
            return self.aggregation_mean_.data


# =============================================================================
# API
# =============================================================================


@attr.s(frozen=True)
class DRVProcess(object):
    """DRV Processes (Decision with Reduction of Variability).

    DRV processes have been developed to support Group Decision
    Making. They are applicable to the cases in which
    all members of the group operate in the same organization and, therefore,
    they must share organizational values, knowledge and preferences.
    Assumes that it is necessary to generate agreement on the preferences of
    group members [1]_.

    Parameters
    ----------

    climit : float, optional (default=.25)
        Consensus limit. Maximum value of the IVR to asume that the solution
        is stable.

        The Stability is verified using the normality analysis of priorities
        for each element of a subproblem, or by using the IVR
        (Índice de Variabilidad Remanente, Remaining Variability Index)
        IVR <= ``climit`` are indicative of stability.

    ntest : 'shapiro' or 'ks' (default='shapiro')
        Normality-test. Test to check if the priorities established by group
        members must have a random behavior, represented by Normal
        Distribution. The values must be 'shapiro' for the Shapito-Wilk test
        [2]_ or 'ks' for the Kolmogorov-Smirnov test for goodness of fit. [3]_

    ntest_kwargs : dict or None, optional (default=None)
        Parameters to the normal test function.

    alpha : float, optional (default=0.01)
        significance. If the any p-value of n-test is less than `alpha`, we
        reject the null hypothesis.

    njobs : int, optional (default=-1)
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of cores.
        For more information check
        `joblib <https://pythonhosted.org/joblib/>`_ documentation.

    agg_only_consensus : bool, optional (default=True)
        Calculate the aggregation only when a consensus is achieved.


    References
    ----------

    .. [1] Zanazzi, J. L., Gomes, L. F. A. M., & Dimitroff, M. (2014).
           Group decision making applied to preventive maintenance systems.
           Pesquisa Operacional, 34(1), 91-105.
    .. [2] Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for
           normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
    .. [3] Daniel, Wayne W. (1990). "Kolmogorov–Smirnov one-sample test".
           Applied Nonparametric Statistics (2nd ed.). Boston: PWS-Kent.
           pp. 319–330. ISBN 0-534-91976-6.

    """

    climit: float = attr.ib(default=.25)
    ntest: str = attr.ib(default="shapiro")
    ntest_kwargs: dict = attr.ib(default=None)
    alpha: float = attr.ib(default=0.01)
    njobs: int = attr.ib(default=-1)
    agg_only_consensus: bool = attr.ib(default=True)

    @climit.validator
    def climit_check(self, attribute, value):
        if not isinstance(value, float):
            raise ValueError("'climit' value must be an instance of float")
        elif value < 0 or value > 1:
            raise ValueError("'climit' has to be >= 0 and <= 1")

    @alpha.validator
    def alpha_check(self, attribute, value):
        if not isinstance(value, float):
            raise ValueError("'alpha' value must be an instance of float")
        elif value < 0 or value > 1:
            raise ValueError("'alpha' has to be >= 0 and <= 1")

    @njobs.validator
    def njobs_check(self, attribute, value):
        if not isinstance(value, int):
            raise ValueError("'njobs' must be an integer")

    @ntest.validator
    def ntest_check(self, attribute, value):
        if value not in NORMAL_TESTS and not callable(value):
            ntests = tuple(NORMAL_TESTS)
            raise ValueError(f"'ntests' must be a callable or str in {ntests}")

    @ntest_kwargs.validator
    def ntest_kwargs_check(self, attribute, value):
        if value is not None and not isinstance(value, dict):
            raise ValueError("'ntest_kwargs' must be a dict or None")

    def decide(self, abc: list, weights: np.ndarray = None) -> DRVResult:
        """Execute the DRV Processes.

        Parameters
        ----------

        abc : list of 2D array-like
            Alternative by criteria list. Every element of the list
            is a 2D array where the element $A_{ij}$ of the matrix $k$
            represent the valoration of the participant $i$ of the
            alternative $j$ by the criteria $k$.

        weights : 2D array-like or None (default=None)
            Weight valoration matrix. Where the element $W_{ik} represent
            the valoration of the participant $i$ of the weight of the
            criterion $k$. If is None, all the criteria has the same
            weight.

        Returns
        -------

        result : DRVResult
            Resume of the entire DRV process. If the problem not achieve a
            consensus (`result.consensus == False`) the aggregation phase
            are not executed.

        """
        # run the rdv
        drv_result = drv(
            weights,
            abc,
            ntest=self.ntest,
            ntest_kwargs=self.ntest_kwargs,
            climit=self.climit,
            njobs=self.njobs,
            alpha=self.alpha,
            agg_only_consensus=self.agg_only_consensus)

        return DRVResult(
            climit=self.climit,
            ntest=self.ntest,
            alpha=self.alpha,
            ntest_kwargs=self.ntest_kwargs,
            **drv_result)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
