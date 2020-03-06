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
import microAO.aoAlg as AO
from scipy.signal import gaussian
from skimage.restoration import unwrap_phase

class TestAOFunctions(unittest.TestCase):

  def _gaussian_funcion(self, x, offset, normalising, mean, std_dev):
    return (offset - normalising) + (normalising * np.exp((-(x - mean) ** 2) / (2 * std_dev ** 2)))

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

    fft_filter[(self.radius-self.true_y_freq-int(gauss_dim/2)):
               (self.radius-self.true_y_freq+int(gauss_dim/2)),
               (self.radius-self.true_x_freq-int(gauss_dim/2)):
               (self.radius-self.true_x_freq+int(gauss_dim/2))] = gauss
    return fft_filter

  def _construct_single_mode_measurements(self, shape, z_min, z_max, num_mes, true_max):
    stack = np.ones((num_mes, shape[0], shape[1]))
    z_measurements = np.linspace(z_min, z_max, num_mes)

    min_y = (stack.shape[1] // 2) - int(stack.shape[1] * 0.2)
    max_y = (stack.shape[1] // 2) + int(stack.shape[1] * 0.2)
    min_x = (stack.shape[2] // 2) - int(stack.shape[2] * 0.2)
    max_x = (stack.shape[2] // 2) + int(stack.shape[2] * 0.2)
    for ii in range(num_mes):
      stack[ii, min_y:max_y, min_x:max_x] = self._gaussian_funcion(z_measurements[ii],
                                                   100, 100, true_max, ((z_max-z_min)/4))

    return stack

  def _construct_multiple_mode_measurements(self, shape, z_min, z_max, num_mes, all_true_max, noll_zernike):
    stack = np.ones((num_mes*len(noll_zernike), shape[0], shape[1]))
    z_measurements = np.linspace(z_min, z_max, num_mes)

    min_y = (stack.shape[1] // 2) - int(stack.shape[1] * 0.2)
    max_y = (stack.shape[1] // 2) + int(stack.shape[1] * 0.2)
    min_x = (stack.shape[2] // 2) - int(stack.shape[2] * 0.2)
    max_x = (stack.shape[2] // 2) + int(stack.shape[2] * 0.2)

    for ii in range(len(noll_zernike)):
        for jj in range(num_mes):
            stack[jj+(num_mes*ii), min_y:max_y, min_x:max_x] = self._gaussian_funcion(z_measurements[jj],
                                                                                      100, 100,
                                                                                      all_true_max[noll_zernike[ii]-1],
                                                                                      ((z_max - z_min) / 4))

    return stack

  def setUp(self):
    #Initialize necessary variables
    self.planned_n_actuators = 10
    self.num_poke_steps = 5
    self.pattern = np.zeros((self.planned_n_actuators))
    self.radius = 1024
    self.nzernike = 10
    self.true_x_freq = 350
    self.true_y_freq = 0
    self.true_mask = self._construct_true_mask()
    self.test_inter = self._construct_interferogram()
    self.true_fft_filter = self._construct_true_fft_filter()
    self.true_control_matrix = np.diag(np.ones(self.nzernike))
    self.AO_func = AO.AdaptiveOpticsFunctions()
    self.AO_mask = self.AO_func.make_mask(self.radius)
    self.AO_fft_filter = self.AO_func.make_fft_filter(image = self.test_inter, region=None)
    self.true_ac_applied = np.linspace(0, 1, self.nzernike)
    self.true_metric_single_measure = np.outer(gaussian(100,10),gaussian(100,10).T)
    self.true_fourier_metric = 5700
    self.true_fourier_power_metric = 373000
    self.true_contrast_metric = 771000
    self.true_gradient_metric = 0.00537
    self.true_second_moment_metric = 83
    self.test_NA = 1.1
    self.test_wavelength = 500 * (10**-9)
    self.test_pixel_size = 0.1193 * (10 ** -6)

    self.true_num_mes = 15
    self.true_z_min = -1
    self.true_z_max = 1
    self.true_max_mode_z = 0.5
    self.true_single_mode_measurements = self._construct_single_mode_measurements((100, 100), self.true_z_min,
                                                                                  self.true_z_max, self.true_num_mes,
                                                                                  self.true_max_mode_z)

    self.true_noll_zernike = np.asarray([1, 3, 5])
    self.true_max_modes_z = np.zeros(self.planned_n_actuators)
    self.true_max_modes_z[self.true_noll_zernike[0] - 1] = -0.35
    self.true_max_modes_z[self.true_noll_zernike[1] - 1] = 0.3
    self.true_max_modes_z[self.true_noll_zernike[2] - 1] = -0.55

    self.true_multi_mode_measurements = self._construct_multiple_mode_measurements((100,100), self.true_z_min,
                                                                                   self.true_z_max, self.true_num_mes,
                                                                                   self.true_max_modes_z,
                                                                                   self.true_noll_zernike)

  def test_make_mask(self):
    test_mask = self.AO_func.make_mask(self.radius)
    np.testing.assert_array_equal(self.true_mask, test_mask)

  def test_fourier_filter(self):
    test_fft_filter = self.AO_func.make_fft_filter(image = self.test_inter, region=None)

    true_pos = np.asarray([self.true_y_freq, self.true_x_freq])
    max_pos = abs(np.asarray(np.where(test_fft_filter == np.max(test_fft_filter))) - 1024)
    test_pos = np.mean(max_pos, axis=1)
    np.testing.assert_almost_equal(test_pos[0], true_pos[0], decimal=0)
    np.testing.assert_almost_equal(test_pos[1], true_pos[1], decimal=0)

  def test_mgcentroid(self):
    g0, g1 = np.asarray(self.AO_func.mgcentroid(self.true_fft_filter)) - self.radius
    np.testing.assert_almost_equal(abs(g0), self.true_x_freq, decimal=0)
    np.testing.assert_almost_equal(abs(g1), self.true_y_freq, decimal=0)

  def test_unwrap_interferometry(self):
    zcoeffs_in = np.zeros(self.planned_n_actuators)
    zcoeffs_in[2] = 1
    aberration_angle = aotools.phaseFromZernikes(zcoeffs_in, self.test_inter.shape[1])
    aberration_phase = (1 + np.cos(aberration_angle) + (1j * np.sin(aberration_angle))) * self.true_mask
    test_phase = self.test_inter * aberration_phase
    aberration = unwrap_phase(np.arctan2(aberration_phase.imag,aberration_phase.real))

    test_aberration = self.AO_func.unwrap_interferometry(image=test_phase)
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
      zc_out[ii, :] = self.AO_func.get_zernike_modes(img, self.nzernike)
      max_z_mode = np.where(zc_out[0,:] == np.max(zc_out[0,:]))[0][0]
      np.testing.assert_equal(max_z_mode, 5)

    z_diff = zcoeffs_in-zc_out
    z_mean_diff = np.mean(z_diff)
    z_var_diff = np.var(z_diff)

    np.testing.assert_almost_equal(z_mean_diff, 0, decimal=3)
    np.testing.assert_almost_equal(z_var_diff, 0, decimal=5)

  def test_createcontrolmatrix(self):
    pokeSteps = np.linspace(0.05,0.95,self.num_poke_steps)

    allTestZernikeAmps = []
    allPokeSteps = []
    for ii in range(self.planned_n_actuators):
      for jj in pokeSteps:
        currPokeAmps = np.zeros(self.planned_n_actuators)
        currPokeAmps[ii] = jj
        zcoeffs_in = np.zeros(self.nzernike)
        zcoeffs_in[ii] = 1*jj

        allTestZernikeAmps.append(zcoeffs_in)
        allPokeSteps.append(currPokeAmps)

    allTestZernikeAmps = np.asarray(allTestZernikeAmps)
    allPokeSteps = np.asarray(allPokeSteps)

    test_control_matrix = self.AO_func.create_control_matrix(zernikeAmps=allTestZernikeAmps,
                                                             pokeSteps=allPokeSteps,
                                                             numActuators=self.planned_n_actuators,
                                                             pupil_ac = None,
                                                             threshold = 0.005)

    np.testing.assert_array_equal(test_control_matrix,self.true_control_matrix)

  def test_ac_pos_from_zernike(self):
    self.AO_func.set_controlMatrix(self.true_control_matrix)
    test_zernike_applied = np.linspace(0,1,self.nzernike)
    test_ac_pos = self.AO_func.ac_pos_from_zernike(test_zernike_applied,self.planned_n_actuators)
    np.testing.assert_array_equal(test_ac_pos,self.true_ac_applied)

  def test_measure_metric_fourier(self):
    self.AO_func.set_metric('fourier')
    test_fourier_metric = self.AO_func.measure_metric(self.true_metric_single_measure, wavelength=self.test_wavelength,
                                                      NA=self.test_NA, pixel_size=self.test_pixel_size)
    np.testing.assert_almost_equal(test_fourier_metric/self.true_fourier_metric, 1, decimal=2)

  def test_measure_metric_fourier_power(self):
    self.AO_func.set_metric('fourier_power')
    test_fourier_power_metric = self.AO_func.measure_metric(self.true_metric_single_measure, wavelength=self.test_wavelength,
                                                      NA=self.test_NA, pixel_size=self.test_pixel_size)
    np.testing.assert_almost_equal(test_fourier_power_metric / self.true_fourier_power_metric, 1, decimal=2)

  def test_measure_metric_contrast(self):
    self.AO_func.set_metric('contrast')
    test_contrast_metric = self.AO_func.measure_metric(self.true_metric_single_measure)
    np.testing.assert_almost_equal(test_contrast_metric / self.true_contrast_metric, 1, decimal=2)

  def test_measure_metric_gradient(self):
    self.AO_func.set_metric('gradient')
    test_gradient_metric = self.AO_func.measure_metric(self.true_metric_single_measure)
    np.testing.assert_almost_equal(test_gradient_metric / self.true_gradient_metric, 1, decimal=2)

  def test_measure_metric_second_moment(self):
    self.AO_func.set_metric('second_moment')
    test_second_moment_metric = self.AO_func.measure_metric(self.true_metric_single_measure, wavelength=self.test_wavelength,
                                                      NA=self.test_NA, pixel_size=self.test_pixel_size)
    np.testing.assert_almost_equal(test_second_moment_metric / self.true_second_moment_metric, 1, decimal=2)

  def test_find_zernike_amp_sensorless(self):
    self.AO_func.set_metric('contrast')
    zernike_amplitudes = np.linspace(self.true_z_min, self.true_z_max, self.true_num_mes,)
    amplitude_present = self.AO_func.find_zernike_amp_sensorless(self.true_single_mode_measurements,
                                                                 zernike_amplitudes, wavelength=self.test_wavelength,
                                                                 NA=self.test_NA, pixel_size=self.test_pixel_size)

    print(amplitude_present)
    print(self.true_max_mode_z)
    np.testing.assert_almost_equal(-1 * amplitude_present, self.true_max_mode_z, decimal=2)

  def test_get_zernike_modes_sensorless(self):
    self.AO_func.set_metric('contrast')
    z_steps = np.linspace(self.true_z_min, self.true_z_max, self.true_num_mes)
    full_zernike_applied = np.zeros((self.true_num_mes * self.true_noll_zernike.shape[0], self.planned_n_actuators))
    for noll_ind in self.true_noll_zernike:
      ind = np.where(self.true_noll_zernike == noll_ind)[0][0]
      full_zernike_applied[ind * self.true_num_mes:(ind + 1) * self.true_num_mes, noll_ind - 1] = z_steps

    coef = self.AO_func.get_zernike_modes_sensorless(self.true_multi_mode_measurements, full_zernike_applied,
                                                     self.true_noll_zernike, wavelength=self.test_wavelength,
                                                     NA=self.test_NA, pixel_size=self.test_pixel_size)

    print(coef)
    print(self.true_max_modes_z)
    for noll_ind in self.true_noll_zernike:
      np.testing.assert_almost_equal(-1 * coef[noll_ind-1], self.true_max_modes_z[noll_ind-1],decimal=2)

if __name__ == '__main__':
    unittest.main()
