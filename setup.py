#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2016 David Pinto <david.pinto@bioch.ox.ac.uk>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

import setuptools
setuptools.setup(
  name = "microAO",
  version = "0.1.0+dev",
  description = "An extensible microscope adaptive optics use.",
  license = "GPL-3.0+",

  ## Do not use author_email because that won't play nice once there
  ## are multiple authors.
  author = "Nick Hall <nicholas.hall@dtc.ox.ac.uk>",

  url = "https://github.com/NickHallONBI/microAO",

  packages = [
    "microAO",
    "microAO.testsuite",
  ],

  install_requires = [
    "numpy",
    "scipy",
    "scikit-image",
    "aotools",
    ## We use six instead of anything else because microscope is already
    ## dependent on size
    "six",
    "sympy",
  ],

  ## https://pypi.python.org/pypi?:action=list_classifiers
  classifiers = [
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  ],
  test_suite="microAO.testsuite",
)

