
# coding: utf-8

# In[1]:


import numpy as np

import attr

import joblib

from skcriteria import norm


# In[33]:


@attr.s(frozen=True)
class DRVResult(object):

    I = attr.ib(repr=False)
    climit = attr.ib(repr=False)

    wnproducts = attr.ib(repr=False)
    wsctotal = attr.ib(repr=False)
    wssw = attr.ib(repr=False)
    wssb = attr.ib(repr=False)
    wscu = attr.ib(repr=False)
    wivr = attr.ib()
    win_consensus = attr.ib()

    anproducts = attr.ib(repr=False)
    asctotal = attr.ib(repr=False)
    assw = attr.ib(repr=False)
    assb = attr.ib(repr=False)
    ascu = attr.ib(repr=False)
    aivr = attr.ib()
    ain_consensus = attr.ib()

    pval = attr.ib()


def nproduct_indexes(nproducts):
    sctotal = np.sum((nproducts - np.mean(nproducts)) ** 2)
    ssw = np.sum((nproducts - np.mean(nproducts, axis=0)) ** 2)
    ssb = sctotal - ssw
    scu = (
        (nproducts.shape[0] - 1) /
        float(nproducts.shape[1] * 3))

    ivr = ssw / scu
    resume = np.mean(nproducts, axis=0)

    return sctotal, ssw, ssb, scu, ivr, resume


def solve_nproducts(mtx):
    rmtx = np.flip(mtx, axis=1)
    rcumprod = np.cumprod(rmtx, axis=1)
    wproducts = np.flip(rcumprod, axis=1)
    return norm.sum(wproducts, axis=1)


def subproblem(mtx):
    nproducts = solve_nproducts(mtx)
    sctotal, ssw, ssb, scu, ivr, resume = nproduct_indexes(nproducts)
    return nproducts, sctotal, ssw, ssb, scu, ivr, resume


def drv(weights, abc, climit=.25, njobs=None):

    # number of participants & alternatives
    N, I = np.shape(abc[0])

    # number of criteria
    J = len(abc)

    if np.ndim(weights) > 1:
        wnp, wsct, wssw, wssb, wscu, wivr, weights = subproblem(weights)
        winc = wivr <= climit
    else:
        wwnp, wsct, wssw, wscu, wivr, wssb = None, None, None, None, None, None
        winc = None

    anp = []
    asct = np.empty(J)
    assw = np.empty(J)
    assb = np.empty(J)
    ascu = np.empty(J)
    aivr = np.empty(J)
    aagg = np.empty((J, I))

    njobs = joblib.cpu_count() if njobs is None else njobs
    with joblib.Parallel(n_jobs=njobs) as jobs:
        results = jobs(
            joblib.delayed(subproblem)(amtx)
            for amtx in abc)
        for idx, r in enumerate(results):
            anp.append(r[0])
            asct[idx] = r[1]
            assw[idx] = r[2]
            assb[idx] = r[3]
            ascu[idx] = r[4]
            aivr[idx] = r[5]
            aagg[idx] = r[6]

    ainc = aivr <= climit

    return DRVResult(
        I=I, climit=climit, wnproducts=wnp, pval=.5,
        wsctotal=wsct, wssw=wssw, wssb=wssb,
        wscu=wscu, wivr=wivr, win_consensus=winc,
        anproducts=tuple(anp), asctotal=asct, assw=assw,
        assb=assb, ascu=ascu, aivr=aivr, ain_consensus=ainc), weights , aagg


# In[34]:

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

    result, weights, aagg = drv(weights=wmtx, abc=abc)
    np.testing.assert_allclose(result.wnproducts, e_wp_matrix)
    np.testing.assert_allclose(result.wsctotal, 0.3178, rtol=1e-03)
    np.testing.assert_allclose(result.wssw, 0.0345, rtol=1e-03)
    np.testing.assert_allclose(result.wssb, 0.2833, rtol=1e-03)
    np.testing.assert_allclose(result.wscu, 0.2381, rtol=1e-03)
    np.testing.assert_allclose(result.wivr, 0.145, rtol=1e-03)
    return result, weights, aagg


if __name__ == "__main__":
    test_drv()
