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

__doc__ = """DRV processes have been developed to support Group Decision
Making.

They are applicable to the cases in which
all members of the group operate in the same organization and, therefore,
they must share organizational values, knowledge and preferences.
Assumes that it is necessary to generate agreement on the preferences of
group members.

"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import unittest

import pytest

import numpy as np

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

        self.abc = [
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

    def test_drv(self):
        dec = DRVProcess(njobs=1)

        result = dec.decide(weights=self.wmtx, abc=self.abc)
        np.testing.assert_allclose(result.wnproducts, self.e_wp_matrix)
        np.testing.assert_allclose(result.wsctotal, 0.3178, rtol=1e-03)
        np.testing.assert_allclose(result.wssw, 0.0345, rtol=1e-03)
        np.testing.assert_allclose(result.wssb, 0.2833, rtol=1e-03)
        np.testing.assert_allclose(result.wscu, 0.2381, rtol=1e-03)
        np.testing.assert_allclose(result.wivr, 0.145, rtol=1e-03)
        #~ return result


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    return pytest.main(sys.argv)

if __name__ == "__main__":
    run_tests()
