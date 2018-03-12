#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2017 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>
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

  def _construct_interferogram(self):
    mid_y = self.radius
    mid_x = self.radius
    interferogram_shape = ((self.radius*2),(self.radius*2))
    stripes_ft = np.zeros(interferogram_shape)
    stripes_ft[mid_y-self.true_y_freq, mid_x-self.true_x_freq] = 25
    stripes_ft[mid_y, mid_x] = 50
    stripes_ft[mid_y+self.true_y_freq, mid_x+self.true_x_freq] = 25

    stripes_shift = np.fft.fftshift(stripes_ft)
    stripes = np.fft.fft2(stripes_shift).real

    test_interferogram = stripes * self.true_mask
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

    fft_filter[(self.radius-self.true_y_freq-(gauss_dim/2)):
               (self.radius-self.true_y_freq+(gauss_dim/2)),
               (self.radius-self.true_x_freq-(gauss_dim/2)):
               (self.radius-self.true_x_freq+(gauss_dim/2))] = gauss
    return fft_filter

  def setUp(self):
    self.planned_n_actuators = 10
    self.pattern = np.zeros((self.planned_n_actuators))
    self.dm = dummies.TestDeformableMirror(self.planned_n_actuators)
    cam = dummies.TestCamera()
    self.AO = microAO.AdaptiveOpticsDevice(camera=cam, mirror=self.dm)

    #Initialize necessary variables
    self.radius = 1024
    self.AO.set_roi(x0=self.radius, y0=self.radius, radius=self.radius)
    self.nzernike = 10
    self.true_x_freq = 350
    self.true_y_freq = 0
    self.true_mask = self._construct_true_mask()
    self.test_inter = self._construct_interferogram()
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
    #Test that the test aberrations isn't all 0s
    np.testing.assert_equal(np.not_equal(np.sum(test_aberration),0), True)
    ab_ratio_mean = np.mean(test_aberration[aberration != 0]/aberration[aberration != 0])
    ab_ratio_var = np.var(test_aberration[aberration != 0]/aberration[aberration != 0])

    #Test that there is a sensible ratio between the test and true aberration
    #and that the variance of ratio is small
    np.testing.assert_equal((abs(ab_ratio_mean) < 10), True)
    np.testing.assert_almost_equal(ab_ratio_var, 0, decimal=1)

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
    z_mean_diff = np.mean(z_diff)
    z_var_diff = np.var(z_diff)

    np.testing.assert_almost_equal(z_mean_diff, 0, decimal=3)
    np.testing.assert_almost_equal(z_var_diff, 0, decimal=5)

  def test_createcontrolmatrix(self):
    test_stack = np.zeros((self.nzernike*10,self.test_inter.shape[0],self.test_inter.shape[1]),
                          dtype=np.complex_)
    true_control_matrix = np.diag(np.ones(self.nzernike))

    count = 0
    pokeSteps = np.linspace(0.05,0.95,10)
    for ii in range(self.nzernike):
      for jj in pokeSteps:
        zcoeffs_in = np.zeros(self.nzernike)
        zcoeffs_in[ii] = 1*jj
        aberration_angle = aotools.phaseFromZernikes(zcoeffs_in, (self.radius*2))
        aberration_phase = ((1 + np.cos(aberration_angle) + (1j *
                            np.sin(aberration_angle))) * self.true_mask)
        test_stack[count,:,:] = self.test_inter * aberration_phase
        print "Test image %d\%d constructed" %(int(count+1),int(self.nzernike*10))
        count += 1

    test_control_matrix = self.AO.createcontrolmatrix(test_stack, self.nzernike, pokeSteps)
    max_ind = []
    for ii in range(self.nzernike):
      max_ind.append(np.where(test_control_matrix[:,ii] ==
                              np.max(test_control_matrix[:,ii]))[0][0])
    np.testing.assert_equal(max_ind, range(self.nzernike))

    CM_diff = test_control_matrix - true_control_matrix
    CM_var_diff = np.var(np.diag(CM_diff))

    np.testing.assert_almost_equal(CM_var_diff, 0, decimal=3)

  def test_calibrate(self):
    test_control_matrix = self.AO.calibrate(self.AO.acquire(),numPokeSteps=10)
    test_control_matrix_shape = np.shape(test_control_matrix)
    np.testing.assert_equal(test_control_matrix_shape,(10,self.nzernike))

  def test_flatten(self):
    pass

if __name__ == '__main__':
    unittest.main()
