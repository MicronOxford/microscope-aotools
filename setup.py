#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2017 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

manifest_files = [
  "COPYING",
  "README.md",
]

setuptools.setup(
  name = "microscope-aotools",
  version = "1.1.4+dev",
  description = "An extensible microscope adaptive optics use.",
  license = "GPL-3.0+",

  ## Do not use author_email because that won't play nice once there
  ## are multiple authors.
  author = "Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>",

  url = "https://github.com/MicronOxford/microscope-aotools",

  packages = [
    "microAO",
    "microAO.testsuite",
  ],

  python_requires = ">=3.5",

  install_requires = [
    "numpy",
    "scipy",
    "scikit-image",
    "aotools",
    "Pyro4",
    "microscope>=0.6.0",
  ],

  extra_requires = {
    "cockpit": ["microscope-cockpit", "wxPython", "matplotlib"],
  },

  ## https://pypi.python.org/pypi?:action=list_classifiers
  classifiers = [
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
  ],
  test_suite="microAO.testsuite",

  long_description=long_description,
  long_description_content_type="text/markdown"
)
