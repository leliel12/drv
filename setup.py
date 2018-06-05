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

"""This file is for distribute scikit-criteria DRV extensions

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

from ez_setup import use_setuptools
use_setuptools()

from setuptools import setup, find_packages

os.environ["SCDRV_IN_SETUP"] = "True"
import sc_drv


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = ["scikit-criteria", "pytest", "matplotlib", "jinja2"]


# =============================================================================
# FUNCTIONS
# =============================================================================

def do_setup():
    setup(
        name=sc_drv.NAME,
        version=sc_drv.VERSION,
        long_description=sc_drv.LONG_DESCRIPTION,
        description=sc_drv.DESCRIPTION,
        author=sc_drv.AUTHORS,
        author_email=sc_drv.EMAIL,
        url=sc_drv.URL,
        license=sc_drv.LICENSE,
        keywords=sc_drv.KEYWORDS,
        classifiers=[
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved',
            'Programming Language :: Python',
            'Topic :: Software Development',
            'Topic :: Scientific/Engineering',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6'],
        packages=[
            pkg for pkg in find_packages() if pkg.startswith("sc_drv")],
        py_modules=["ez_setup"],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":
    do_setup()
