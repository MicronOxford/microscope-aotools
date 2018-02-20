#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2017 David Pinto <david.pinto@bioch.ox.ac.uk>
##
## This file is part of Microscope.
##
## Microscope is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Microscope is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Microscope.  If not, see <http://www.gnu.org/licenses/>.

import unittest

import numpy
import six
import aotools

import microscope.testsuite.devices as dummies

class TestAOFunctions(unittest.TestCase):
  def setUp(self):
    self.planned_n_actuators = 10
    self.pattern = numpy.zeros((self.planned_n_actuators))
    self.dm = dummies.TestDeformableMirror(self.planned_n_actuators)
    self.cam = dummies.TestCamera

  def test_applying_pattern(self):
    ## This mainly checks the implementation of the dummy device.  It
    ## is not that important but it is the basis to trust the other
    ## tests wich will actually test the base class.
    self.pattern[:] = 0.2
    self.dm.apply_pattern(self.pattern)
    numpy.testing.assert_array_equal(self.dm._current_pattern, self.pattern)

  def test_make(self):
    pass

  def test_fourier_filter(self):
    pass

  def test_phase_unwrap(self):
    pass

  def test_aqcuire_zernike_modes(self):
    pass

  def test_calibrate(self):
    pass

  def test_flatten(self):
    pass

if __name__ == '__main__':
    unittest.main()
