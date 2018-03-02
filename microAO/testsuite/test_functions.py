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

import numpy as np
import six
import aotools
import microAO
from skimage.restoration import unwrap_phase

import microscope.testsuite.devices as dummies

class TestAOFunctions(unittest.TestCase):
  def __construct_interferogram(self, stripe_frequency_x, stripe_frequency_y,
                                interferogram_shape = (2048,2048)):
    mid_y = int(interferogram_shape[0]/2)
    mid_x = int(interferogram_shape[1]/2)
    stripes_ft = np.zeros(interferogram_shape)
    stripes_ft[mid_y-stripe_frequency_y, mid_x-stripe_frequency_x] = 25
    stripes_ft[mid_y, mid_x] = 50
    stripes_ft[mid_y+stripe_frequency_y, mid_x+stripe_frequency_x] = 25

    stripes_shift = np.fft.fftshift(stripes_ft)
    stripes = np.fft.fft2(stripes_shift).real

    radius = mid_x
    diameter = radius * 2
    mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius

    test_interferogram = stripes * mask
    return test_interferogram

  def setUp(self):
    self.planned_n_actuators = 10
    self.pattern = np.zeros((self.planned_n_actuators))
    self.dm = dummies.TestDeformableMirror(self.planned_n_actuators)
    self.cam = dummies.TestCamera
    self.AO = microAO.AdaptiveOpticsDevice()

  def test_applying_pattern(self):
    ## This mainly checks the implementation of the dummy device.  It
    ## is not that important but it is the basis to trust the other
    ## tests wich will actually test the base class.
    self.pattern[:] = 0.2
    self.dm.apply_pattern(self.pattern)
    np.testing.assert_array_equal(self.dm._current_pattern, self.pattern)

  def test_makemask(self):
    self.radius = 1024
    self.diameter = self.radius * 2
    test_mask = np.sqrt((np.arange(-self.radius,self.radius)**2).reshape((
        self.diameter,1)) + (np.arange(-self.radius,self.radius)**2)) < self.radius
    mask = self.AO.makemask(self.radius)
    np.testing.assert_array_equal(mask, test_mask)
    self.mask = mask

  def test_fourier_filter(self):
    pass

  def test_mgcentroid(self):
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
