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
import aotools
import microAO
from scipy.signal import gaussian
from skimage.restoration import unwrap_phase

import microscope.testsuite.devices as dummies

class TestAOFunctions(unittest.TestCase):
  def _construct_interferogram(self,interferogram_shape = (2048,2048)):
    mid_y = int(interferogram_shape[0]/2)
    mid_x = int(interferogram_shape[1]/2)
    stripes_ft = np.zeros(interferogram_shape)
    stripes_ft[mid_y-self.true_y_freq, mid_x-self.true_x_freq] = 25
    stripes_ft[mid_y, mid_x] = 50
    stripes_ft[mid_y+self.true_y_freq, mid_x+self.true_x_freq] = 25

    stripes_shift = np.fft.fftshift(stripes_ft)
    stripes = np.fft.fft2(stripes_shift).real

    radius = mid_x
    diameter = radius * 2
    mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius

    test_interferogram = stripes * mask
    return test_interferogram

  def _construct_true_mask(self):
    diameter = self.radius * 2
    mask = np.sqrt((np.arange(-self.radius,self.radius)**2).reshape((
        diameter,1)) + (np.arange(-self.radius,self.radius)**2)) < self.radius
    return mask

  def _construct_true_fft_filter(self):
    diameter = self.radius * 2
    fft_filter = np.zeros((diameter,diameter))
    gauss_dim = int(diameter*(5.0/16.0))
    FWHM = int((3.0/8.0) * gauss_dim)
    stdv = FWHM/np.sqrt(8 * np.log(2))
    x = gaussian(gauss_dim,stdv)
    gauss = np.outer(x,x.T)
    gauss = gauss*(gauss>(np.max(x)*np.min(x)))

    fft_filter[(self.radius+self.true_y_freq-(gauss_dim/2)):
               (self.radius+self.true_y_freq+(gauss_dim/2)),
               (self.radius+self.true_x_freq-(gauss_dim/2)):
               (self.radius+self.true_x_freq+(gauss_dim/2))] = gauss
    return fft_filter

  def setUp(self):
    self.planned_n_actuators = 10
    self.pattern = np.zeros((self.planned_n_actuators))
    self.dm = dummies.TestDeformableMirror(self.planned_n_actuators)
    cam = dummies.TestCamera()
    self.AO = microAO.AdaptiveOpticsDevice(camera=cam, mirror=self.dm)

    #Initialize necessary variables
    self.AO.set_roi(x0=1023, y0=1023, radius=1024)
    self.nzernike = 10
    self.radius = 1024
    self.true_x_freq = 350
    self.true_y_freq = 0
    self.test_inter = self._construct_interferogram()
    self.true_mask = self._construct_true_mask()
    self.true_fft_filter = self._construct_true_fft_filter()

  def test_applying_pattern(self):
    ## This mainly checks the implementation of the dummy device.  It
    ## is not that important but it is the basis to trust the other
    ## tests wich will actually test the base class.
    self.pattern[:] = 0.2
    self.dm.apply_pattern(self.pattern)
    np.testing.assert_array_equal(self.dm._current_pattern, self.pattern)

  def test_makemask(self):
    test_mask = self.AO.makemask(self.radius)
    np.testing.assert_array_equal(self.true_mask, test_mask)

  def test_fourier_filter(self):
    test_fft_filter = self.AO.getfourierfilter(self.test_inter)

    true_pos = np.asarray([self.true_y_freq, self.true_x_freq])
    max_pos = abs(np.asarray(np.where(test_fft_filter == np.max(test_fft_filter))) - 1024)
    test_pos = np.mean(max_pos, axis=1)
    np.testing.assert_almost_equal(test_pos[0], true_pos[0], decimal=0)
    np.testing.assert_almost_equal(test_pos[1], true_pos[1], decimal=0)

  def test_mgcentroid(self):
    g0, g1 = np.asarray(self.AO.mgcentroid(self.true_fft_filter)) - self.radius
    np.testing.assert_almost_equal(abs(g0), self.true_x_freq, decimal=0)
    np.testing.assert_almost_equal(abs(g1), self.true_y_freq, decimal=0)

  def test_phase_unwrap(self):
    zcoeffs_in = np.zeros(self.planned_n_actuators)
    zcoeffs_in[2] = 1
    aberration_angle = aotools.phaseFromZernikes(zcoeffs_in, self.test_inter.shape[1])
    aberration_phase = (1 + np.cos(aberration_angle) + (1j * np.sin(aberration_angle))) * self.true_mask
    test_phase = self.test_inter * aberration_phase
    aberration = unwrap_phase(np.arctan2(aberration_phase.imag,aberration_phase.real))

    test_aberration = self.AO.phaseunwrap(image=test_phase)
    aberration_diff = ((np.sum(abs(test_aberration)) - np.sum(abs(aberration)))/
                      (np.shape(aberration)[0]* np.shape(aberration)[1]))
    np.testing.assert_almost_equal(aberration_diff, 0, decimal=3)

  def test_aqcuire_zernike_modes(self):
    diameter = 128
    zcoeffs_in = np.zeros(self.nzernike)
    zcoeffs_in[5] = 1
    img = np.zeros((diameter,diameter))
    img[:,:] = aotools.phaseFromZernikes(zcoeffs_in, diameter)

    zc_out = np.zeros((5,self.nzernike))
    for ii in range(5):
      zc_out[ii, :] = self.AO.getzernikemodes(img, self.nzernike)
      max_z_mode = np.where(zc_out[0,:] == np.max(zc_out[0,:]))[0][0]
      np.testing.assert_equal(max_z_mode, 5)

    z_diff = zcoeffs_in-zc_out
    z_mean_diff = np.mean(z_diff, axis=0)
    z_var_diff = np.var(z_diff, axis=0)

    np.testing.assert_almost_equal(np.mean(z_mean_diff), 0, decimal=3)
    np.testing.assert_almost_equal(np.mean(z_var_diff), 0, decimal=5)

  def test_createcontrolmatrix(self):

    #self.AO.createcontrolmatrix(imageStack=test_stack, noZernikeModes=self.nzernike)
    pass

  def test_calibrate(self):
    pass

  def test_flatten(self):
    pass

if __name__ == '__main__':
    unittest.main()
