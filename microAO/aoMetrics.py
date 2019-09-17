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

def make_ring_mask(size, inner_rad, outer_rad):
    radius = int(size[0] / 2)

    outer_mask = np.sqrt((np.arange(-radius, radius) ** 2).reshape((radius * 2, 1)) + (
            np.arange(-radius, radius) ** 2)) < outer_rad

    inner_mask_neg = np.sqrt((np.arange(-radius, radius) ** 2).reshape((radius * 2, 1)) + (
            np.arange(-radius, radius) ** 2)) < inner_rad
    inner_mask = (inner_mask_neg - 1) * -1
    ring_mask = outer_mask * inner_mask
    return ring_mask

def measure_fourier_metric(image, wavelength=500 * 10 ** -9, NA=1.1,
                            pixel_size=0.1193 * 10 ** -6, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.shape(image)[0] / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(im_shift.shape[0], .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    im_tukey = im_shift * tukey_window
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    cent2corner = np.sqrt(2 * ((image.shape[0] / 2) ** 2))
    rad_to_corner = cent2corner - OTF_outer_rad
    noise_corner_size = int(np.round(np.sqrt((rad_to_corner ** 2) / 2) * 0.9))
    noise = (fftarray_sq_log[0:noise_corner_size, 0:noise_corner_size] +
            fftarray_sq_log[0:noise_corner_size, -noise_corner_size:] +
            fftarray_sq_log[-noise_corner_size:, 0:noise_corner_size] +
            fftarray_sq_log[-noise_corner_size:, -noise_corner_size:]) / 4
    threshold = np.mean(noise) * 1.125

    ring_mask = make_ring_mask(np.shape(image),0.1 * OTF_outer_rad, OTF_outer_rad)
    freq_above_noise = (fftarray_sq_log > threshold) * ring_mask
    metric = np.count_nonzero(freq_above_noise)
    return metric

def measure_contrast_metric(image, **kwargs):
    return (np.max(image) - np.min(image))/(np.max(image) + np.min(image))

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
    OTF_outer_rad = (freq_ratio) * (np.shape(image)[0] / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(im_shift.shape[0], .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    im_tukey = im_shift * tukey_window
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    cent2corner = np.sqrt(2 * ((image.shape[0] / 2) ** 2))
    rad_to_corner = cent2corner - OTF_outer_rad
    noise_corner_size = int(np.round(np.sqrt((rad_to_corner ** 2) / 2) * 0.9))
    noise = (fftarray_sq_log[0:noise_corner_size, 0:noise_corner_size] +
            fftarray_sq_log[0:noise_corner_size, -noise_corner_size:] +
            fftarray_sq_log[-noise_corner_size:, 0:noise_corner_size] +
            fftarray_sq_log[-noise_corner_size:, -noise_corner_size:]) / 4
    threshold = np.mean(noise) * 1.125

    ring_mask = make_ring_mask(np.shape(image),0.1 * OTF_outer_rad, OTF_outer_rad)
    ramp_mask = np.sqrt(np.outer(np.linspace(-150,150,image.shape[0])**2,np.ones(image.shape[1]).T)+
                        np.outer(np.ones(image.shape[0]), (np.linspace(-150,150,image.shape[1])**2).T))
    freq_above_noise = (fftarray_sq_log > threshold) * ring_mask * ramp_mask
    metric = np.sum(freq_above_noise)
    return metric

def measure_second_moment_metric(image, wavelength=500 * 10 ** -9, NA=1.1,
                            pixel_size=0.1193 * 10 ** -6, **kwargs):
    ray_crit_dist = (1.22 * wavelength) / (2 * NA)
    ray_crit_freq = 1 / ray_crit_dist
    max_freq = 1 / (2 * pixel_size)
    freq_ratio = ray_crit_freq / max_freq
    OTF_outer_rad = (freq_ratio) * (np.shape(image)[0] / 2)

    im_shift = np.fft.fftshift(image)
    tukey_window = tukey(im_shift.shape[0], .10, True)
    tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
    im_tukey = im_shift * tukey_window
    fftarray = np.fft.fftshift(np.fft.fft2(im_tukey))

    fftarray_sq_log = np.log(np.real(fftarray * np.conj(fftarray)))

    ring_mask = make_ring_mask(np.shape(image),0, OTF_outer_rad)

    x = np.linspace(0, image.shape[1] - 1, image.shape[1])
    x_p = x - ((image.shape[1] - 1) / 2)
    x_prime = np.outer(np.ones(image.shape[1]), x_p)
    y = np.linspace(0, image.shape[0] - 1, image.shape[0])
    y_p = y - ((image.shape[0] - 1) / 2)
    y_prime = np.outer(y_p, np.ones(image.shape[0]))
    ramp_mask = x_prime**2 + y_prime**2

    metric = np.sum(ring_mask * fftarray_sq_log * ramp_mask)/np.sum(fftarray_sq_log)
    return metric