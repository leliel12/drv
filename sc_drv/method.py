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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOCS
# =============================================================================

__doc__ = """DRV method implementation

"""

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

from . import normtests


# =============================================================================
# CONSTANTS
# =============================================================================

NORMAL_TESTS = {
    "shapiro": normtests.shapiro,
    "ks": normtests.kstest,
}


# =============================================================================
# STRUCTURATION FUNCTIONS
# =============================================================================

def nproduct_indexes(nproducts, climit):
    """Calculate the indexs of the products"""
    sctotal = np.sum((nproducts - np.mean(nproducts)) ** 2)
    ssw = np.sum((nproducts - np.mean(nproducts, axis=0)) ** 2)
    ssb = sctotal - ssw
    scu = (
        (nproducts.shape[0] - 1) /
        float(nproducts.shape[1] * 3))

    ivr = ssw / scu
    inc = ivr <= climit

    resume = np.mean(nproducts, axis=0)

    return sctotal, ssw, ssb, scu, ivr, inc, resume


def solve_nproducts(mtx):
    """Create the product (normalized) matrix"""
    rmtx = np.flip(mtx, axis=1)
    rcumprod = np.cumprod(rmtx, axis=1)
    wproducts = np.flip(rcumprod, axis=1)
    return norm.sum(wproducts, axis=1)


def subproblem(mtx, climit, ntest, ntest_kwargs):
    """Create and evaluate the product (normalized) matrix"""
    nproducts = solve_nproducts(mtx)

    sctotal, ssw, ssb, scu, ivr, inc, resume = nproduct_indexes(
        nproducts, climit)

    n_sts, pvals = ntest(nproducts, axis=1, **ntest_kwargs)

    return {"nproducts": nproducts, "sctotal": sctotal,
            "ssw": ssw, "ssb": ssb, "ssu": scu, "ivr": ivr,
            "in_consensus": inc, "ntest_sts": n_sts, "ntest_pvals": pvals,
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

def drv(weights, abc, climit, ntest, ntest_kwargs, njobs):
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
    results = {"N": N, "I": I, "J": J}

    # WEIGHTS
    if np.ndim(weights) > 1:
        wresults = subproblem(
            mtx=weights, climit=climit,
            ntest=ntest, ntest_kwargs=ntest_kwargs)
    else:
        wresults = {}

    # copy weights results to the global results
    results.update({
        "weights_participants": wresults.get("nproducts"),
        "wsctotal": wresults.get("sctotal"),
        "wssw": wresults.get("ssw"),
        "wssb": wresults.get("ssb"),
        "wscu": wresults.get("ssu"),
        "wivr": wresults.get("ivr"),
        "wntest_sts": wresults.get("ntest_sts"),
        "wntest_pvals": wresults.get("ntest_pvals"),
        "win_consensus": wresults.get("in_consensus"),
        "weights_mean": wresults.get("resume")})

    # ALTERNATIVES
    with joblib.Parallel(n_jobs=njobs) as jobs:
        wresults = jobs(
            joblib.delayed(subproblem)(
                amtx, climit=climit,
                ntest=ntest, ntest_kwargs=ntest_kwargs)
            for amtx in abc)

    # copy alt results to the global results
    results.update({
        "mtx_participants": tuple(r["nproducts"] for r in wresults),
        "asctotal": np.hstack(r["sctotal"] for r in wresults),
        "assw": np.hstack(r["ssw"] for r in wresults),
        "assb": np.hstack(r["ssb"] for r in wresults),
        "ascu": np.hstack(r["ssu"] for r in wresults),
        "aivr": np.hstack(r["ivr"] for r in wresults),
        "ain_consensus": np.hstack(r["in_consensus"] for r in wresults),
        "antest_sts": np.vstack(r["ntest_sts"] for r in wresults),
        "antest_pvals": np.vstack(r["ntest_pvals"] for r in wresults),
        "mtx_mean": np.vstack(r["resume"] for r in wresults)})

    # CONSENSUS
    consensus = np.all(results["ain_consensus"])
    if consensus and results["weights_mean"] is not None:
        consensus = consensus and results["win_consensus"]
    results["consensus"] = consensus  # to global results

    # AGGREGATION
    if consensus:
        aggregator = simple.WeightedSum(mnorm="none", wnorm="none")

        criteria = [max] * J

        weights_mean = (
            1 if results["weights_mean"] is None else results["weights_mean"])
        agg_m = aggregator.decide(
            results["mtx_mean"].T,
            criteria=criteria, weights=weights_mean)

        with joblib.Parallel(n_jobs=njobs) as jobs:
            agg_p = jobs(
                joblib.delayed(run_aggregator)(
                    idx=idx,
                    mtxs=results["mtx_participants"],
                    criteria=criteria,
                    weights=results["weights_participants"],
                    aggregator=aggregator)
                for idx in range(N))
            agg_p = tuple(agg_p)

            # rank verification
            ttest_results = jobs(
                joblib.delayed(rank_ttest_rel)(
                    agg_p=agg_p, aidx=aidx, bidx=bidx)
                for aidx, bidx in it.combinations(range(I), 2))

            rank_t, rank_p = np.empty(I), np.empty(I)
            for idx, r in enumerate(ttest_results):
                rank_t[idx] = r.statistic
                rank_p[idx] = r.pvalue
    else:
        agg_p, agg_m, rank_t, rank_p = None, None, None, None

    # to global results
    results["aggregation_participants"] = agg_p
    results["aggregation_mean"] = agg_m
    results["rank_check_t"] = rank_t
    results["rank_check_pval"] = rank_p

    return results


# =============================================================================
# CLASSES
# =============================================================================

@attr.s(frozen=True)
class DRVResult(object):

    N = attr.ib()
    I = attr.ib()
    J = attr.ib()
    ntest = attr.ib()
    ntest_kwargs = attr.ib()
    climit = attr.ib()
    consensus = attr.ib()

    weights_participants = attr.ib(repr=False)
    wsctotal = attr.ib(repr=False)
    wssw = attr.ib(repr=False)
    wssb = attr.ib(repr=False)
    wscu = attr.ib(repr=False)
    wivr = attr.ib(repr=False)
    win_consensus = attr.ib(repr=False)
    weights_mean = attr.ib(repr=False)
    wntest_sts = attr.ib(repr=False)
    wntest_pvals = attr.ib(repr=False)

    mtx_participants = attr.ib(repr=False)
    asctotal = attr.ib(repr=False)
    assw = attr.ib(repr=False)
    assb = attr.ib(repr=False)
    ascu = attr.ib(repr=False)
    aivr = attr.ib(repr=False)
    ain_consensus = attr.ib(repr=False)
    mtx_mean = attr.ib(repr=False)
    antest_sts = attr.ib(repr=False)
    antest_pvals = attr.ib(repr=False)

    aggregation_participants = attr.ib(repr=False)
    aggregation_mean = attr.ib(repr=False)
    rank_check_t = attr.ib(repr=False)
    rank_check_pval = attr.ib(repr=False)


@attr.s(frozen=True)
class DRVProcess(object):
    """DRV Processes (Decision with Reduction of Variability).

    DRV processes have been developed to support Group Decision
    Making. They are applicable to the cases in which
    all members of the group operate in the same organization and, therefore,
    they must share organizational values, knowledge and preferences.
    Assumes that it is necessary to generate agreement on the preferences of
    group members.

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
        Normaloty-test. Test to check if the priorities established by group
        members must have a random behavior, represented by Normal
        Distribution. The values must be 'shapiro' for the Shapito-Wilk test
        [1]_ or 'ks' for the Kolmogorov-Smirnov test for goodness of fit. [2]_

    ntest_kwargs : dict or None, optional (default=None)
        Parameters to the normal test function.

    njobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        If None, then the number of jobs is set to the number of cores.

    References
    ----------

    .. [1] Shapiro, S. S. & Wilk, M.B (1965). An analysis of variance test for
           normality (complete samples), Biometrika, Vol. 52, pp. 591-611.
    .. [2] Daniel, Wayne W. (1990). "Kolmogorov–Smirnov one-sample test".
           Applied Nonparametric Statistics (2nd ed.). Boston: PWS-Kent.
           pp. 319–330. ISBN 0-534-91976-6.

    """
    climit: float = attr.ib(default=.25)
    ntest: str = attr.ib(default="shapiro")
    ntest_kwargs: dict = attr.ib(default=None)
    njobs: int = attr.ib(default=None)

    @climit.validator
    def climit_check(self, attribute, value):
        if not isinstance(value, float):
            raise ValueError("'climit' value must be an instance of float")
        elif value < 0 or value > 1:
            raise ValueError("'climit' has to be >= 0 and <= 1")

    @njobs.validator
    def njobs_check(self, attribute, value):
        if value is None:
            return
        elif not isinstance(value, int) and value <= 0:
            raise ValueError("'njobs' must be an integer > 0")

    @ntest.validator
    def ntest_check(self, attribute, value):
        if value not in NORMAL_TESTS and not callable(value):
            ntests = tuple(NORMAL_TESTS)
            raise ValueError(
                f"'ntests' must be a callable or str in {ntests}")

    @ntest_kwargs.validator
    def ntest_kwargs_check(self, attribute, value):
        if value is not None and not isinstance(value, dict):
            raise ValueError("'ntest_kwargs' must be a dict or None")

    def decide(self, abc: list, weights: np.ndarray=None):
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
            weights, abc, ntest=self.ntest, ntest_kwargs=self.ntest_kwargs,
            climit=self.climit, njobs=self.njobs)

        return DRVResult(
            climit=self.climit, ntest=self.ntest,
            ntest_kwargs=self.ntest_kwargs, **drv_result)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
