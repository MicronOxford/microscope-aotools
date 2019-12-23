#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>
##
## microAO is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## microAO is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with microAO.  If not, see <http://www.gnu.org/licenses/>.

#Import required packs
import numpy as np
from scipy.signal import tukey
from skimage.filters import threshold_otsu

def make_OTF_mask(size, y_limits, x_limits):
    inner_y_rad = y_limits[0]
    outer_y_rad = y_limits[1]
    inner_x_rad = x_limits[0]
    outer_x_rad = x_limits[1]

    rad_y = size[0] // 2
    rad_x = size[1] // 2

    outer_mask = np.sqrt(((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1))/outer_y_rad**2) +
                         (np.arange(-rad_x, rad_x) ** 2)/outer_x_rad**2) < 1

    if inner_x_rad != 0 and inner_y_rad != 0:
        inner_mask_neg = np.sqrt(((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1))/inner_y_rad**2) +
                             (np.arange(-rad_x, rad_x) ** 2)/inner_x_rad**2) < 1
        inner_mask = (inner_mask_neg - 1) * -1
    else:
        inner_mask = np.ones(size)
    OTF_mask = outer_mask * inner_mask
    return OTF_mask

def measure_fourier_metric(image, wavelength=500 * 10 ** -9, NA=1.1,
                            pixel_size=0.1193 * 10 ** -6, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_y_rad = (freq_ratio) * (np.shape(image)[0] / 2)
    OTF_outer_x_rad = (freq_ratio) * (np.shape(image)[1] / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[tukey_window.shape[0] // 2 - im_shift.shape[0] // 2:
                                     tukey_window.shape[0] // 2 + im_shift.shape[0] // 2,
                        tukey_window.shape[1] // 2 - im_shift.shape[1] // 2:
                        tukey_window.shape[1] // 2 + im_shift.shape[1] // 2]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = make_OTF_mask(np.shape(image), (0, 1.1 * OTF_outer_y_rad), (0, 1.1 * OTF_outer_x_rad))
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * 1.125

    OTF_mask = make_OTF_mask(np.shape(image), (0.1 * OTF_outer_y_rad, OTF_outer_y_rad),
                              (0.1 * OTF_outer_x_rad, OTF_outer_x_rad))
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask
    metric = np.count_nonzero(freq_above_noise)
    return metric

def measure_contrast_metric(image, no_intensities = 100, **kwargs):
    flattened_image = image.flatten()

    min_ind = max(1, (flattened_image.shape[0]-no_intensities))
    max_ind = flattened_image.shape[0]

    mean_top = np.mean(flattened_image[np.argsort(flattened_image)[min_ind:max_ind]])
    mean_bottom = np.mean(flattened_image[np.argsort(flattened_image)[:min_ind]])
    return mean_top/mean_bottom

def measure_gradient_metric(image, **kwargs):
    image_gradient_x = np.gradient(image, axis=1)
    image_gradient_y = np.gradient(image, axis=0)

    grad_mask_x = image_gradient_x > (threshold_otsu(image_gradient_x) * 1.125)
    grad_mask_y = image_gradient_y > (threshold_otsu(image_gradient_y) * 1.125)

    correction_grad = np.sqrt((image_gradient_x * grad_mask_x) ** 2 + (image_gradient_y * grad_mask_y) ** 2)

    metric = np.mean(correction_grad)
    return metric

def measure_fourier_power_metric(image, wavelength=500 * 10 ** -9, NA=1.1,
                            pixel_size=0.1193 * 10 ** -6, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_y_rad = (freq_ratio) * (np.shape(image)[0] / 2)
    OTF_outer_x_rad = (freq_ratio) * (np.shape(image)[1] / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[tukey_window.shape[0]//2 - im_shift.shape[0]//2:
                                     tukey_window.shape[0]//2 + im_shift.shape[0]//2,
                        tukey_window.shape[1]//2 - im_shift.shape[1]//2:
                        tukey_window.shape[1]//2 + im_shift.shape[1]//2]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = make_OTF_mask(np.shape(image), (0, 1.1 * OTF_outer_y_rad), (0, 1.1 * OTF_outer_x_rad))
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * 1.125

    OTF_mask = make_OTF_mask(np.shape(image), (0.1 * OTF_outer_y_rad, OTF_outer_y_rad),
                             (0.1 * OTF_outer_x_rad, OTF_outer_x_rad))

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    ramp_mask = ((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1))/OTF_outer_y_rad**2) + \
                (np.arange(-rad_x, rad_x) ** 2)/OTF_outer_x_rad**2

    dist = np.sqrt(ramp_mask)
    gamma = abs(dist - 1) * OTF_mask
    omega = 1 - np.exp(-(gamma))

    high_f_amp_mask = 100 * (ramp_mask * omega)/np.max(ramp_mask * omega)

    OTF_mask = make_OTF_mask(np.shape(image), (0.1 * OTF_outer_y_rad, OTF_outer_y_rad),
                             (0.1 * OTF_outer_x_rad, OTF_outer_x_rad))
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask * high_f_amp_mask
    metric = np.sum(freq_above_noise)
    return metric

def measure_second_moment_metric(image, wavelength=500 * 10 ** -9, NA=1.1,
                            pixel_size=0.1193 * 10 ** -6, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_y_rad = (freq_ratio) * (np.shape(image)[0] / 2)
    OTF_outer_x_rad = (freq_ratio) * (np.shape(image)[1] / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[tukey_window.shape[0] // 2 - im_shift.shape[0] // 2:
                                     tukey_window.shape[0] // 2 + im_shift.shape[0] // 2,
                        tukey_window.shape[1] // 2 - im_shift.shape[1] // 2:
                        tukey_window.shape[1] // 2 + im_shift.shape[1] // 2]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    ring_mask = make_OTF_mask(np.shape(image), (0, OTF_outer_y_rad), (0, OTF_outer_x_rad))

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    ramp_mask = (np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) + np.arange(-rad_x, rad_x) ** 2

    dist = np.sqrt(ramp_mask)
    gamma = abs(dist - 1) * ring_mask
    omega = 1 - np.exp(-(gamma))

    metric = np.sum(ring_mask * fftarray_sq_log * ramp_mask * omega)/np.sum(fftarray_sq_log)
    return metric