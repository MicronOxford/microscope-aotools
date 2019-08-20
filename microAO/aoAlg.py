#!/usr/bin/env python
# -*- coding: utf-8 -*-

## Copyright (C) 2018 Nicholas Hall <nicholas.hall@dtc.ox.ac.uk>, Josh Edwards
## <Josh.Edwards222@gmail.com> & Jacopo Antonello
## <jacopo.antonello@dpag.ox.ac.uk>
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
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import tukey, gaussian
from skimage.filters import threshold_otsu
from scipy.optimize import curve_fit
import aotools
import scipy.stats as stats
from skimage.restoration import unwrap_phase
from scipy.integrate import trapz

def gaussian_funcion(x, a, b, c, d):
    return a + (b - a) * np.exp((-(x - c) ** 2) / (2 * d ** 2))

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
                            pixel_size=0.1193 * 10 ** -6):
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

metric_function = {
    'fourier': measure_fourier_metric,
    'contrast': measure_contrast_metric,
}

class AdaptiveOpticsFunctions():

    def __init__(self):
        self.mask = None
        self.fft_filter = None
        self.controlMatrix = None
        self.OTF_ring_mask = None
        self.metric = 'fourier'

    def set_mask(self,mask):
        self.mask = mask
        return

    def set_fft_filter(self,fft_filter):
        self.fft_filter = fft_filter
        return

    def set_controlMatrix(self, controlMatrix):
        self.controlMatrix = controlMatrix
        return

    def set_OTF_ring_mask(self, OTF_ring_mask):
        self.OTF_ring_mask = OTF_ring_mask
        return

    def set_metric(self, metric):
        if metric in metric_function.keys():
            self.metric = metric
        else:
            raise Exception("Error: %s is not a supported image quality metric" %metric)
        return

    def get_metric(self):
        return self.metric

    def make_mask(self, radius):
        diameter = radius * 2
        self.mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius
        return self.mask


    def bin_ndarray(self, ndarray, new_shape, operation='sum'):
        """

        Function acquired from Stack Overflow: https://stackoverflow.com/a/29042041. Stack Overflow or other Stack Exchange
        sites is cc-wiki (aka cc-by-sa) licensed and requires attribution.

        Bins an ndarray in all axes based on the target shape, by summing or
            averaging.

        Number of output dimensions must match number of input dimensions and
            new axes must divide old ones.

        Example
        -------

        m = np.arange(0,100,1).reshape((10,10))
        n = bin_ndarray(m, new_shape=(5,5), operation='sum')
        print(n)

        [[ 22  30  38  46  54]
         [102 110 118 126 134]
         [182 190 198 206 214]
         [262 270 278 286 294]
         [342 350 358 366 374]]

        """
        operation = operation.lower()
        if not operation in ['sum', 'mean']:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                               new_shape))
        compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                      ndarray.shape)]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1*(i+1))
        return ndarray

    def mgcentroid(self, myim, mythr=0.0):
        assert(myim.dtype == np.float)

        myn1, myn2 = myim.shape
        myxx1, myxx2 = np.meshgrid(range(1, myn1 + 1), range(1, myn2 + 1))
        myim[myim < mythr] = 0
        mysum1 = np.sum((myxx1*myim).ravel())
        mysum2 = np.sum((myxx2*myim).ravel())
        mymass = np.sum(myim.ravel())
        return int(np.round(mysum1/mymass)), int(np.round(mysum2/mymass))

    def make_fft_filter(self, image, region=None):
        # Convert image to array and float
        data = np.asarray(image)

        if region is None:
            region = int(data.shape[0]/16)

        # Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1) * tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        # Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        # Remove center section to allow finding of 1st order point
        fftarray = np.fft.fftshift(fftarray)
        find_cent = [int(fftarray.shape[1]/2),int(fftarray.shape[0]/ 2)]
        fftarray[find_cent[1]-region:find_cent[1]+region,find_cent[0]-region:find_cent[0]+region]=0.00001+0j

        # Find approximate position of first order point
        test_point = np.argmax(fftarray)
        test_point = [int(test_point % fftarray.shape[1]), int(test_point / fftarray.shape[1])]

        # Find first order point
        maxpoint = np.zeros(np.shape(test_point), dtype=int)
        maxpoint[:] = test_point[:]
        window = np.zeros((100, 100))

        for ii in range(10):
            try:
                window[:, :] = np.log(
                    abs(fftarray[maxpoint[1] - 50:maxpoint[1] + 50, maxpoint[0] - 50:maxpoint[0] + 50]))
            except ValueError as e:
                raise Exception("Interferometer stripes are too fine. Please make them coarser").with_traceback(e)
            thresh = threshold_otsu(window)
            binaryIm = window > thresh
            windowOtsu = window * binaryIm
            CoM = np.zeros((1, 2))
            CoM[0, :] = np.round(center_of_mass(windowOtsu))
            maxpoint[0] = maxpoint[0] - 50 + int(CoM[0, 1])
            maxpoint[1] = maxpoint[1] - 50 + int(CoM[0, 0])

        self.fft_filter = np.zeros(np.shape(fftarray))
        mask_di = int(data.shape[0]*(5.0/16.0))

        x_shift = np.min((0, abs(maxpoint[0] - mask_di), abs(maxpoint[0] - fftarray.shape[0]) - mask_di))
        y_shift = np.min((0, abs(maxpoint[1] - mask_di), abs(maxpoint[1] - fftarray.shape[1]) - mask_di))

        x = np.sin(np.linspace(0, np.pi, mask_di)) ** 2
        fourier_mask = np.outer(x, x.T)
        y_min = maxpoint[1] - int(np.floor((mask_di / 2.0))) + y_shift
        y_max = maxpoint[1] + int(np.ceil((mask_di / 2.0))) + y_shift
        x_min = maxpoint[0] - int(np.floor((mask_di / 2.0))) + x_shift
        x_max = maxpoint[0] + int(np.ceil((mask_di / 2.0))) + x_shift

        self.fft_filter[y_min:y_max, x_min:x_max] = fourier_mask
        self.fft_filter = np.roll(self.fft_filter, -x_shift, axis=1)
        self.fft_filter = np.roll(self.fft_filter, -y_shift, axis=0)
        return self.fft_filter

    def phase_unwrap(self,image):
        #Convert image to array and float
        data = np.asarray(image)

        #Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1)*tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        #Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        #Apply Fourier filter
        M = np.fft.fftshift(self.fft_filter)
        fftarray_filt = fftarray * M
        fftarray_filt = np.fft.fftshift(fftarray_filt)

        #Roll data to the centre
        centre_y_array, centre_x_array = np.where(self.fft_filter == np.max(self.fft_filter))
        g1 = int(np.round(np.mean(centre_y_array)) - np.round(fftarray_filt.shape[0] // 2))
        g0 = int(np.round(np.mean(centre_x_array)) - np.round(fftarray_filt.shape[0] // 2))
        fftarray_filt = np.roll(fftarray_filt, -g0, axis=1)
        fftarray_filt = np.roll(fftarray_filt, -g1, axis=0)

        #Convert to real space
        fftarray_filt_shift = np.fft.fftshift(fftarray_filt)
        complex_phase = np.fft.fftshift(np.fft.ifft2(fftarray_filt_shift))

        #Find phase data by taking 2d arctan of imaginary and real parts
        phaseorder1 = np.zeros(complex_phase.shape)
        phaseorder1[:,:] = np.arctan2(complex_phase.imag,complex_phase.real)

        #Mask out edge region to allow unwrap to only use correct region
        phaseorder1mask = phaseorder1 * self.mask

        #Perform unwrap
        phaseorder1unwrap = unwrap_phase(phaseorder1mask)
        out = phaseorder1unwrap * self.mask
        return out

    def get_zernike_modes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        #Resize image
        original_dim = int(np.shape(image_unwrap)[0])
        while original_dim%resize_dim is not 0:
            resize_dim -= 1

        if resize_dim < original_dim/resize_dim:
            resize_dim = int(np.round(original_dim/resize_dim))

        image_resize = self.bin_ndarray(image_unwrap, new_shape=(resize_dim,resize_dim), operation='mean')

        #Calculate Zernike mode
        zcoeffs_dbl = []
        num_pixels = np.count_nonzero(aotools.zernike(1, resize_dim))
        for i in range(1,(noZernikeModes+1)):
            intermediate = trapz(image_resize * aotools.zernike(i, resize_dim))
            zcoeffs_dbl.append(trapz(intermediate) / (num_pixels))
        coef = np.asarray(zcoeffs_dbl)
        return coef

    def create_control_matrix(self, zernikeAmps, pokeSteps, numActuators, pupil_ac=None, threshold=0.005):
        if np.any(pupil_ac) == None:
            pupil_ac = np.ones(numActuators)

        noZernikeModes = zernikeAmps.shape[1]

        slopes = np.zeros(noZernikeModes)
        intercepts = np.zeros(noZernikeModes)
        r_values = np.zeros(noZernikeModes)
        p_values = np.zeros(noZernikeModes)
        std_errs = np.zeros(noZernikeModes)

        C_mat = np.zeros((noZernikeModes, numActuators))
        offsets = np.zeros((noZernikeModes, numActuators))
        P_tests = np.zeros((noZernikeModes, numActuators))

        for ii in range(numActuators):
            if pupil_ac[ii] == 1:
                pokeSteps_trimmed = pokeSteps[np.where(pokeSteps[:, ii] != 0)[0], ii]
                zernikeModeAmp = zernikeAmps[np.where(pokeSteps[:, ii] != 0)[0], :]

                # Check that the influence slope for each actuator can actually be calculated
                if len(pokeSteps_trimmed) < 2:
                    raise Exception("Not enough Zernike mode values to calculate slope for actuator %i. "
                                    "Control matrix calculation will fail" % (ii + 1))
                    break

                # Fit a linear regression to get the relationship between actuator position and Zernike mode amplitude
                for kk in range(noZernikeModes):
                    try:
                        slopes[kk], intercepts[kk], r_values[kk], p_values[kk], std_errs[kk] = \
                            stats.linregress(pokeSteps_trimmed, zernikeModeAmp[:, kk])
                    except Exception as e:
                        print(e)

                # Input obtained slopes as the entries in the control matrix
                C_mat[:, ii] = slopes[:]
                offsets[:, ii] = intercepts[:]
                P_tests[:, ii] = p_values[:]
            else:
                print("Actuator %d is not in the pupil and therefore skipped" % (ii))
        print("Computing Control Matrix")
        self.controlMatrix = np.linalg.pinv(C_mat, rcond=threshold)
        print("Control Matrix computed")
        return self.controlMatrix

    def ac_pos_from_zernike(self, applied_z_modes, numActuators):
        if int(np.shape(applied_z_modes)[0]) < int(np.shape(self.controlMatrix)[1]):
            pad_length = int(np.shape(applied_z_modes)[0]) - int(np.shape(self.controlMatrix)[1])
            np.pad(applied_z_modes, (0,pad_length), 'constant')
        elif int(np.shape(applied_z_modes)[0]) > int(np.shape(self.controlMatrix)[1]):
            applied_z_modes = applied_z_modes[:int(np.shape(self.controlMatrix)[1])]
        else:
            pass

        actuator_pos = np.dot(self.controlMatrix, applied_z_modes)

        try:
            assert len(actuator_pos) == numActuators
        except:
            raise Exception

        return actuator_pos

    def find_zernike_amp_sensorless(self, image_stack, zernike_amplitudes, wavelength=500 * 10 ** -9, NA=1.1,
                               pixel_size=0.1193 * 10 ** -6):
        metrics_measured = []
        for ii in range(image_stack.shape[0]):
            print("Measuring metric %i/%i" % (ii + 1, image_stack.shape[0]))
            metric_measured = metric_function[self.metric](image_stack[ii, :, :], wavelength=wavelength, NA=NA,
                                                          pixel_size=pixel_size)
            metrics_measured.append(metric_measured)
        metrics_measured = np.asarray(metrics_measured)

        print("Metrics measured")

        print("Fitting metric polynomial")
        try:
            [offset, normalising, mean, std_dev], pcov = curve_fit(gaussian_funcion, zernike_amplitudes, metrics_measured)
            print("Calculating amplitude present")
            if (normalising - offset) < 0:
                print("Fitting converged on minima. Defaulting to 0 amplitude.")
                mean = 0
        except RuntimeError:
            max_from_mean_var = (np.max(metrics_measured) - np.min(metrics_measured))/np.mean(metrics_measured)
            if max_from_mean_var >= 0.1:
                print("Could not accurately fit metric polynomial. Using maximum metric amplitude")
                mean = zernike_amplitudes[metrics_measured==np.max(metrics_measured)]
            else:
                print("Could not accurately fit metric polynomial. Defaulting to 0 amplitude.")
                mean = 0

        amplitude_present = -1.0 * mean
        print("Amplitude calculated = %f" % amplitude_present)
        return amplitude_present

    def get_zernike_modes_sensorless(self, full_image_stack, full_zernike_applied, nollZernike,
                                     wavelength=500 * 10 ** -9, NA=1.1, pixel_size=0.1193 * 10 ** -6):
        numMes = full_zernike_applied.shape[0]/nollZernike.shape[0]

        coef = np.zeros(full_zernike_applied.shape[1])
        for ii in range(nollZernike.shape[0]):
            image_stack = full_image_stack[ii * numMes:(ii + 1) * numMes,:,:]
            zernike_applied = full_zernike_applied[ii * numMes:(ii + 1) * numMes,nollZernike[ii]-1]
            print("Calculating Zernike amplitude %i/%i" %(ii+1, nollZernike.shape[0]))
            amp = self.find_zernike_amp_sensorless(image_stack, zernike_applied, wavelength=wavelength, NA=NA,
                                                   pixel_size=pixel_size)
            coef[nollZernike[ii]-1] = amp

        return coef