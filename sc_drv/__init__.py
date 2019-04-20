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

__doc__ = """DRV processes: Group Decision Making.

They are applicable to the cases in which
all members of the group operate in the same organization and, therefore,
they must share organizational values, knowledge and preferences.
Assumes that it is necessary to generate agreement on the preferences of
group members.

"""

# =============================================================================
# CONSTANTS
# =============================================================================

__version__ = ("0", "2", "1")

NAME = "sc-drv"

LONG_DESCRIPTION = "\n".join([l.strip() for l in __doc__.splitlines()])

DESCRIPTION = LONG_DESCRIPTION.splitlines()[0]

VERSION = ".".join(__version__)

AUTHORS = "Cabral, Luczywo & Zanazzi"

EMAIL = "jbc.develop@gmail.com"

URL = "https://github.com/leliel12/sc_drv"

LICENSE = "3 Clause BSD"

KEYWORDS = "mcda mcdm group-decision-making muti-criteria".split()


# =============================================================================
# IMPORTS
# =============================================================================

import os

if os.getenv("SCDRV_IN_SETUP") != "True":
    from .method import *  # noqa
    from .tests import run_tests  # noqa

del os

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(__doc__)
