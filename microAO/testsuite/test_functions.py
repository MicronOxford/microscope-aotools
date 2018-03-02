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
    self.true_x_freq = 350
    self.true_y_freq = 0
    self.test_inter = self.__construct_interferogram(stripe_frequency_x=
                    self.true_x_freq,stripe_frequency_y=self.true_y_freq)
    test_fft_filter = self.AO.getfourierfilter(self.test_inter)

    fft_filter = np.zeros(np.shape(self.test_inter))
    gauss_dim = int(self.test_inter.shape[0]*(5.0/16.0))
    FWHM = int((3.0/8.0) * gauss_dim)
    stdv = FWHM/np.sqrt(8 * np.log(2))
    x = np.gaussian(gauss_dim,stdv)
    gauss = np.outer(x,x.T)
    gauss = gauss*(gauss>(np.max(x)*np.min(x)))

    fft_filter[(self.radius-(gauss_dim/2)):(self.radius+(gauss_dim/2)),
    (self.radius+10-(gauss_dim/2)):(self.radius+10+(gauss_dim/2))] = gauss
    np.testing.assert_array_equal(test_fft_filter,fft_filter)
    self.fft_filter = test_fft_filter

  def test_mgcentroid(self):
    g0, g1 = self.AO.mgcentroid(self.fft_filter) - np.round(self.fft_filter.shape[0]//2)
    assert abs(g0) == range(self.true_x_freq-2, self.true_x_freq+3)
    assert abs(g1) == range(self.true_x_freq-2, self.true_x_freq+3)

  def test_phase_unwrap(self):
    zcoeffs_in = np.zeros(self.self.planned_n_actuators)
    zcoeffs_in[2] = 1
    aberration_angle = aotools.phaseFromZernikes(zcoeffs_in, self.test_inter.shape[1])
    aberration_phase = (1 + np.cos(aberration_angle) + (1j * np.sin(aberration_angle))) * self.mask
    test_phase = self.test_inter * aberration_phase
    aberration = unwrap_phase(np.arctan2(aberration_phase.imag,aberration_phase.real))

    test_aberration = self.AO.phaseunwrap(test_phase)
    assert ((np.sum(abs(test_aberration)) - np.sum(abs(aberration)))/
            (np.shape(aberration)[0]* np.shape(aberration)[1])) < 0.001

  def test_aqcuire_zernike_modes(self):
    pass

  def test_calibrate(self):
    pass

  def test_flatten(self):
    pass

if __name__ == '__main__':
    unittest.main()
