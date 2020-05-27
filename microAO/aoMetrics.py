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

def make_OTF_mask(size, inner_rad, outer_rad):
    rad_y = int(size[0] / 2)
    rad_x = int(size[1] / 2)

    outer_mask = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                         np.arange(-rad_x, rad_x) ** 2) < outer_rad

    inner_mask_neg = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                         np.arange(-rad_x, rad_x) ** 2) < inner_rad
    inner_mask = (inner_mask_neg - 1) * -1
    ring_mask = outer_mask * inner_mask
    return ring_mask


def measure_fourier_metric(image, wavelength, NA, pixel_size, noise_amp_factor=1.125, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.max(image.shape) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = make_OTF_mask(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) * noise_amp_factor

    OTF_mask = make_OTF_mask(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask
    metric = np.count_nonzero(freq_above_noise)
    return metric

def measure_contrast_metric(image, no_intensities = 100, **kwargs):
    flattened_image = image.flatten()

    flattened_image_list = flattened_image.tolist()
    flattened_image_list.sort()

    mean_top = np.mean(flattened_image_list[-no_intensities:])
    mean_bottom = np.mean(flattened_image[:no_intensities])
    return mean_top/mean_bottom

def measure_gradient_metric(image, **kwargs):
    image_gradient_x = np.gradient(image, axis=1)
    image_gradient_y = np.gradient(image, axis=0)

    grad_mask_x = image_gradient_x > (threshold_otsu(image_gradient_x) * 1.125)
    grad_mask_y = image_gradient_y > (threshold_otsu(image_gradient_y) * 1.125)

    correction_grad = np.sqrt((image_gradient_x * grad_mask_x) ** 2 + (image_gradient_y * grad_mask_y) ** 2)

    metric = np.mean(correction_grad)
    return metric


def measure_fourier_power_metric(image, wavelength, NA, pixel_size, noise_amp_factor=2,
                                 high_f_amp_factor=100, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = freq_ratio * (np.max(np.shape(image)) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    noise_mask = make_OTF_mask(np.shape(image), 0, 1.1 * OTF_outer_rad)
    threshold = np.mean(fftarray_sq_log[noise_mask == 0]) + (noise_amp_factor * np.sqrt(np.var((fftarray_sq_log[noise_mask == 0]))))

    circ_mask = make_OTF_mask(np.shape(image), 0, OTF_outer_rad)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[0]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[1]))
    ramp_mask = x_prime ** 2 + y_prime ** 2

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    dist = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                   np.arange(-rad_x, rad_x) ** 2)
    omega = 1 - np.exp((dist / OTF_outer_rad) - 1)

    high_f_amp_mask = high_f_amp_factor * (ramp_mask * omega) / np.max(ramp_mask * omega)

    OTF_mask = make_OTF_mask(np.shape(image), 0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > threshold) * OTF_mask * high_f_amp_mask
    metric = np.sum(freq_above_noise)
    return metric


def measure_second_moment_metric(image, wavelength, NA, pixel_size, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = freq_ratio * (np.max(np.shape(image)) / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(np.max(im_shift.shape), .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    tukey_window_crop = tukey_window[int(tukey_window.shape[0] / 2 - im_shift.shape[0] / 2):
                                     int(tukey_window.shape[0] / 2 + im_shift.shape[0] / 2),
                        int(tukey_window.shape[1] / 2 - im_shift.shape[1] / 2):
                        int(tukey_window.shape[1] / 2 + im_shift.shape[1] / 2)]
    im_tukey = im_shift * tukey_window_crop
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    ring_mask = make_OTF_mask(np.shape(image), 0, OTF_outer_rad)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[0]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[1]))
    ramp_mask = x_prime ** 2 + y_prime ** 2

    rad_y = int(image.shape[0] / 2)
    rad_x = int(image.shape[1] / 2)
    dist = np.sqrt((np.arange(-rad_y, rad_y) ** 2).reshape((rad_y * 2, 1)) +
                   np.arange(-rad_x, rad_x) ** 2)
    omega = 1 - np.exp((dist/OTF_outer_rad)-1)

    metric = np.sum(ring_mask * fftarray_sq_log * ramp_mask * omega)/np.sum(fftarray_sq_log)
    return metric