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

"""DRV Processes tests

"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import unittest

import pytest

import numpy as np

from matplotlib import axes, pyplot as plt

from skcriteria import norm

from .method import DRVProcess


# =============================================================================
# TESTS
# =============================================================================

class DRVTestCase(unittest.TestCase):

    def setUp(self):
        self.wmtx = [
            [1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0],
            [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
            [1.5, 2.0, 1.2, 1.5, 1.0, 1.5, 1.0],
            [1.5, 1.5, 1.2, 1.5, 1.2, 1.0, 1.0],
            [2.0, 1.5, 1.0, 1.0, 1.1, 1.0, 1.0]]

        self.e_wp_matrix = norm.sum([
            [8.0, 8.0, 4.0, 2.0, 2.0, 2.0, 1.0],
            [4.0, 4.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            [3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            [8.1, 5.4, 2.7, 2.25, 1.5, 1.5, 1.0],
            [4.86, 3.24, 2.16, 1.8, 1.2, 1.0, 1.0],
            [3.3, 1.65, 1.1, 1.1, 1.1, 1.0, 1.0]], axis=1)

        self.abc_nc = [
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

        self.abc_c = [
            # MO
            np.array([
                [1.5, 3.0, 1.0],
                [1.0, 3.0, 1.0],
                [0.833, 3.6, 1.0],
                [1.1, 2.5, 1.0],
                [1.0, 3.0, 1.0],
                [1.2, 3.0, 1.0]]),

            # COSTO
            np.array([
                [1.1, 1.1, 1.0],
                [1.2, 1.5, 1.0],
                [1.1, 2.0, 1.0],
                [1.1, 1.2, 1.0],
                [1.2, 1.1, 1.0],
                [1.1, 1.5, 1.0]]),

            # EXP
            np.array([
                [1.1, 2.0, 1.0],
                [1.0, 1.5, 1.0],
                [1.1, 2.0, 1.0],
                [2.1, 2.0, 1.0],
                [1.5, 1.5, 1.0],
                [1.2, 3.0, 1.0]])] + self.abc_nc[3:]

    def tearDown(self):
        plt.close()

    def test_drv_shapiro_no_consensus(self):
        dec = DRVProcess(njobs=1, ntest="shapiro", agg_only_consensus=False)
        result = dec.decide(weights=self.wmtx, abc=self.abc_nc)
        np.testing.assert_allclose(result.wmtx_, self.e_wp_matrix)
        np.testing.assert_allclose(result.wsst_, 0.3178, rtol=1e-03)
        np.testing.assert_allclose(result.wssw_, 0.0345, rtol=1e-03)
        np.testing.assert_allclose(result.wssb_, 0.2833, rtol=1e-03)
        np.testing.assert_allclose(result.wssu_, 0.2381, rtol=1e-03)
        np.testing.assert_allclose(result.wivr_, 0.145, rtol=1e-03)

    def test_drv_shapiro_consensus(self):
        dec = DRVProcess(njobs=1, ntest="shapiro")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        np.testing.assert_allclose(result.wmtx_, self.e_wp_matrix)
        np.testing.assert_allclose(result.wsst_, 0.3178, rtol=1e-03)
        np.testing.assert_allclose(result.wssw_, 0.0345, rtol=1e-03)
        np.testing.assert_allclose(result.wssb_, 0.2833, rtol=1e-03)
        np.testing.assert_allclose(result.wssu_, 0.2381, rtol=1e-03)
        np.testing.assert_allclose(result.wivr_, 0.145, rtol=1e-03)

    def test_drv_ks_no_consensus(self):
        dec = DRVProcess(njobs=1, ntest="ks")

        result = dec.decide(weights=self.wmtx, abc=self.abc_nc)
        np.testing.assert_allclose(result.wmtx_, self.e_wp_matrix)
        np.testing.assert_allclose(result.wsst_, 0.3178, rtol=1e-03)
        np.testing.assert_allclose(result.wssw_, 0.0345, rtol=1e-03)
        np.testing.assert_allclose(result.wssb_, 0.2833, rtol=1e-03)
        np.testing.assert_allclose(result.wssu_, 0.2381, rtol=1e-03)
        np.testing.assert_allclose(result.wivr_, 0.145, rtol=1e-03)

    def test_drv_ks_consensus(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        np.testing.assert_allclose(result.wmtx_, self.e_wp_matrix)
        np.testing.assert_allclose(result.wsst_, 0.3178, rtol=1e-03)
        np.testing.assert_allclose(result.wssw_, 0.0345, rtol=1e-03)
        np.testing.assert_allclose(result.wssb_, 0.2833, rtol=1e-03)
        np.testing.assert_allclose(result.wssu_, 0.2381, rtol=1e-03)
        np.testing.assert_allclose(result.wivr_, 0.145, rtol=1e-03)

    def test_violin_plots(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        ptype = "violin"
        self.assertIsInstance(
            result.plot.weights_by_participants(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.weights_by_subproblems(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_participants(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_alternatives(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_participants(1, ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_alternatives(1, ptype=ptype), axes.Axes)

    def test_box_plots(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        ptype = "box"
        self.assertIsInstance(
            result.plot.weights_by_participants(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.weights_by_subproblems(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_participants(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_alternatives(ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_participants(1, ptype=ptype), axes.Axes)
        self.assertIsInstance(
            result.plot.utilities_by_alternatives(1, ptype=ptype), axes.Axes)

    def test_preference(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        self.assertIsInstance(result.plot(), axes.Axes)
        self.assertIsInstance(result.plot.preference(), axes.Axes)

    def test_ivr_plot(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        self.assertIsInstance(result.plot.ivr(), axes.Axes)

    def test_heatmap_plots(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        self.assertIsInstance(result.plot.weight_heatmap(), axes.Axes)
        self.assertIsInstance(result.plot.utilities_heatmap(), axes.Axes)
        self.assertIsInstance(result.plot.utilities_heatmap(1), axes.Axes)

    def test_invalid_violin_box_plots(self):
        dec = DRVProcess(njobs=1, ntest="ks")
        result = dec.decide(weights=self.wmtx, abc=self.abc_c)
        ptype = "WRONG!"
        with self.assertRaises(ValueError):
            result.plot.weights_by_participants(ptype=ptype)
        with self.assertRaises(ValueError):
            result.plot.weights_by_subproblems(ptype=ptype)
        with self.assertRaises(ValueError):
            result.plot.utilities_by_participants(ptype=ptype)
        with self.assertRaises(ValueError):
            result.plot.utilities_by_alternatives(ptype=ptype)
        with self.assertRaises(ValueError):
            result.plot.utilities_by_participants(1, ptype=ptype)
        with self.assertRaises(ValueError):
            result.plot.utilities_by_alternatives(1, ptype=ptype)

    def test_comb_I_2(self):
        # test the number of t-test is constent witht the
        # comb I in 2.
        for i in [4, 5, 6, 7, 11]:
            abc = [np.random.rand(6, i) for _ in range(7)]
            dec = DRVProcess(njobs=1, agg_only_consensus=False)

            # in old times this fails
            dec.decide(weights=self.wmtx, abc=abc)


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Execute all the tests"""
    return pytest.main(sys.argv)


if __name__ == "__main__":
    run_tests()
