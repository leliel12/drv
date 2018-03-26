
# coding: utf-8

# In[1]:


import numpy as np

from scipy import stats

import attr

import joblib

from skcriteria import norm
from skcriteria.madm.simple import WSum

import normtests


# =============================================================================
# CONSTANTS
# =============================================================================

NORMAL_TESTS = {
    "shapiro": normtests.shapiro,
    "kstest": normtests.kstest,
    "anderson": normtests.anderson,
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def nproduct_indexes(nproducts, climit):
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
    rmtx = np.flip(mtx, axis=1)
    rcumprod = np.cumprod(rmtx, axis=1)
    wproducts = np.flip(rcumprod, axis=1)
    return norm.sum(wproducts, axis=1)


def subproblem(mtx, climit, ntest, ntest_kwargs):
    nproducts = solve_nproducts(mtx)

    sctotal, ssw, ssb, scu, ivr, inc, resume = nproduct_indexes(
        nproducts, climit)

    n_sts, pvals = ntest(nproducts, axis=1, **ntest_kwargs)
    import ipdb; ipdb.set_trace()

    return {"nproducts": nproducts, "sctotal": sctotal,
            "ssw": ssw, "ssb": ssb, "ssu": scu, "ivr": ivr,
            "in_consensus": inc, "ntest_sts": n_sts, "ntest_pvals": pvals,
            "resume": resume}


def drv(weights, abc, climit, ntest, ntest_kwargs, njobs):

    # determine numbers of parallel jobs
    njobs = joblib.cpu_count() if njobs is None else njobs

    # determine the normal test
    ntest = NORMAL_TESTS.get(ntest, ntest)
    ntest_kwargs = {} if ntest_kwargs is None else ntest_kwargs

    # number of participants & alternatives
    N, I = np.shape(abc[0])

    # number of criteria
    J = len(abc)

    # place to store the results
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
        "wnproducts": wresults.get("nproducts"),
        "wsctotal": wresults.get("sctotal"),
        "wssw": wresults.get("ssw"),
        "wssb": wresults.get("ssb"),
        "wscu": wresults.get("ssu"),
        "wivr": wresults.get("ivr"),
        "wntest_sts": wresults.get("ntest_sts"),
        "wntest_pvals": wresults.get("ntest_pvals"),
        "win_consensus": wresults.get("in_consensus"),
        "weights": wresults.get("resume")})

    # ALTERNATIVES
    with joblib.Parallel(n_jobs=njobs) as jobs:
        wresults = jobs(
            joblib.delayed(subproblem)(
                amtx, climit=climit,
                ntest=ntest, ntest_kwargs=ntest_kwargs)
            for amtx in abc)

    results.update({
        "anproducts": tuple(r["nproducts"] for r in wresults),
        "asctotal": np.hstack(r["sctotal"] for r in wresults),
        "assw": np.hstack(r["ssw"] for r in wresults),
        "assb": np.hstack(r["ssb"] for r in wresults),
        "ascu": np.hstack(r["ssu"] for r in wresults),
        "aivr": np.hstack(r["ivr"] for r in wresults),
        "ain_consensus": np.hstack(r["in_consensus"] for r in wresults),
        "antest_sts": np.vstack(r["ntest_sts"] for r in wresults),
        "antest_pvals": np.vstack(r["ntest_pvals"] for r in wresults),
        "aagg": np.vstack(r["resume"] for r in wresults)})

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

    wnproducts = attr.ib(repr=False)
    wsctotal = attr.ib(repr=False)
    wssw = attr.ib(repr=False)
    wssb = attr.ib(repr=False)
    wscu = attr.ib(repr=False)
    wivr = attr.ib(repr=False)
    win_consensus = attr.ib(repr=False)
    weights = attr.ib(repr=False)
    wntest_sts = attr.ib()
    wntest_pvals = attr.ib()

    anproducts = attr.ib(repr=False)
    asctotal = attr.ib(repr=False)
    assw = attr.ib(repr=False)
    assb = attr.ib(repr=False)
    ascu = attr.ib(repr=False)
    aivr = attr.ib(repr=False)
    ain_consensus = attr.ib(repr=False)
    aagg = attr.ib(repr=False)
    antest_sts = attr.ib()
    antest_pvals = attr.ib()

    data = attr.ib(repr=False)


@attr.s(frozen=True)
class DRVProcess(object):

    climit = attr.ib(default=.25)
    njobs = attr.ib(default=None)
    ntest = attr.ib(default="shapiro")
    ntest_kwargs = attr.ib(default=None)
    aggregator = attr.ib(default

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

    def decide(self, weights, abc):
        # run the rdv
        drv_result = drv(
            weights, abc, ntest=self.ntest, ntest_kwargs=self.ntest_kwargs,
            climit=self.climit, njobs=self.njobs)

        return DRVResult(
            climit=self.climit, ntest=self.ntest,
            ntest_kwargs=self.ntest_kwargs, data=None, **drv_result)




# =============================================================================
# TESTS
# =============================================================================

def test_drv():
    wmtx = [
        [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
        [1.5, 2.0, 1.2, 1.5, 1.0, 1.5, 1.0],
        [1.5, 1.5, 1.2, 1.5, 1.2, 1.0, 1.0],
        [2.0, 1.5, 1.0, 1.0, 1.1, 1.0, 1.0]]

    e_wp_matrix = norm.sum([
        [8.0, 8.0, 4.0, 2.0, 2.0, 2.0, 1.0],
        [4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        [3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
        [8.1, 5.4, 2.7, 2.25, 1.5, 1.5, 1.0],
        [4.86, 3.24, 2.16, 1.8, 1.2, 1.0, 1.0],
        [3.3, 1.65, 1.1, 1.1, 1.1, 1.0, 1.0]], axis=1)

    abc = [
        # MO
        np.array([
            [2.5, 2.0, 1.0],
            [0.5, 3.0, 1.0],
            [2.5, 2.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 4.0, 1.0],
            [6.0, 5.0, 1.0]]),

        # COSTO
        np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [3.0, 2.5, 1.0],
            [1.4, 1.3, 1.0],
            [2.5, 2.0, 1.0],
            [0.5, 0.5, 1.0]]),

        # EXP
        np.array([
            [3.0, 2.5, 1.0],
            [2.4, 1.2, 1.0],
            [1.0, 1.0, 1.0],
            [5.0, 4.0, 1.0],
            [1.5, 2.0, 1.0],
            [1.0, 1.0, 1.0]]),

        # FLOTA
        np.array([
            [0.67, 3.0, 1.0],
            [0.9, 2.1, 1.0],
            [1.2, 4.0, 1.0],
            [1.5, 2.0, 1.0],
            [0.9, 4.4, 1.0],
            [1.5, 2.0, 1.0]]),

        # MEJ SERV
        np.array([
            [1.5, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.5, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 3.0, 1.0]]),

        # HyS
        np.array([
            [1.5, 4.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.0, 3.0, 1.0],
            [1.2, 4.0, 1.0],
            [1.1, 3.0, 1.0]]),

        # trat
        np.array([
            [2.0, 1.5, 1.0],
            [1.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [2.0, 1.2, 1.0],
            [4.0, 1.0, 1.0],
            [1.5, 1.1, 1.0]])
    ]

    dec = DRVProcess()

    result = dec.decide(weights=wmtx, abc=abc)
    np.testing.assert_allclose(result.wnproducts, e_wp_matrix)
    np.testing.assert_allclose(result.wsctotal, 0.3178, rtol=1e-03)
    np.testing.assert_allclose(result.wssw, 0.0345, rtol=1e-03)
    np.testing.assert_allclose(result.wssb, 0.2833, rtol=1e-03)
    np.testing.assert_allclose(result.wscu, 0.2381, rtol=1e-03)
    np.testing.assert_allclose(result.wivr, 0.145, rtol=1e-03)
    return result


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    test_drv()
