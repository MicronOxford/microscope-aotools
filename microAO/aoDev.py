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

# Import required packs
import functools

import numpy as np
import Pyro4
import time
import aotools
from microAO.aoAlg import AdaptiveOpticsFunctions

# Should fix this with multiple inheritance for this class!
aoAlg = AdaptiveOpticsFunctions()

from microscope.abc import Device
from microscope import TriggerType
from microscope import TriggerMode

import logging

unwrap_method = {
    'interferometry': aoAlg.unwrap_interferometry,
}

_logger = logging.getLogger(__name__)

wavefront_error_modes = ["RMS","Strehl"]


def _with_wavefront_camera_ttype_software(func):
    """Method decorator to set camera with software trigger type."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        ttype = self.wavefront_camera.trigger_type
        tmode = self.wavefront_camera.trigger_mode

        ttype_needs_change = ttype is not TriggerType.SOFTWARE
        try:
            if ttype_needs_change:
                self.wavefront_camera.set_trigger(TriggerType.SOFTWARE, tmode)
            return_value = func(self, *args, **kwargs)
        finally:
            if ttype_needs_change:
                self.wavefront_camera.set_trigger(ttype, tmode)
        return return_value
    return wrapper


class AdaptiveOpticsDevice(Device):
    """Class for the adaptive optics device
    This class requires an adaptive element and a camera.
    Everything else is generated on or after __init__"""

    _CockpitTriggerType_to_TriggerType = {
        "SOFTWARE": TriggerType.SOFTWARE,
        "RISING_EDGE": TriggerType.RISING_EDGE,
        "FALLING_EDGE": TriggerType.FALLING_EDGE,
    }

    _CockpitTriggerModes_to_TriggerModes = {
        "ONCE": TriggerMode.ONCE,
        "START": TriggerMode.START,
    }

    def __init__(self, ao_element_uri, wavefront_uri=None, slm_uri=None, **kwargs):
        # Init will fail if devices it depends on aren't already running, but
        # deviceserver should retry automatically.
        super(AdaptiveOpticsDevice, self).__init__(**kwargs)
        # Adaptive optic element device.
        self.ao_element = Pyro4.Proxy('PYRO:%s@%s:%d' % (ao_element_uri[0].__name__,
                                                         ao_element_uri[1], ao_element_uri[2]))
        # Wavefront sensor. Must support soft_trigger for now.
        if wavefront_uri is not None:
            self.wavefront_camera = Pyro4.Proxy('PYRO:%s@%s:%d' % (wavefront_uri[0].__name__,
                                                                   wavefront_uri[1], wavefront_uri[2]))
        # SLM device
        if slm_uri is not None:
            self.slm = Pyro4.Proxy('PYRO:%s@%s:%d' % (slm_uri[0], slm_uri[1], slm_uri[2]))
        # self.ao_element.set_trigger(TriggerType.RISING_EDGE) #Set trigger type to rising edge
        self.numActuators = self.ao_element.n_actuators
        # Region of interest (i.e. pupil offset and radius) on camera.
        self.roi = None
        # Mask for the interferometric data
        self.mask = None
        # Mask to select phase information
        self.fft_filter = None
        # Phase acquisition method
        self.phase_method = 'interferometry'
        # Control Matrix
        self.controlMatrix = None
        # System correction
        self.flat_actuators_sys = np.zeros(self.numActuators)
        # Last applied actuators values
        self.last_actuator_values = None
        # Last applied actuators pattern
        self.last_actuator_patterns = None
        # Last applied phase image
        self.last_phase_pattern = None

        # Record the trigger type and modes that have been set
        self.last_trigger_type = None
        self.last_trigger_mode = None

        # We might not use all the actuators. Create a mask for the actuators outside
        # the pupil so we can selectively calibrate them. 0 denotes actuators at
        # the edge, i.e. outside the pupil, and 1 denotes actuators in the pupil

        # Preliminary mask for DeepSIM
        self.pupil_ac = np.ones(self.numActuators)

        try:
            assert np.shape(self.pupil_ac)[0] == self.numActuators
        except Exception:
            raise Exception("Length mismatch between pupil mask (%i) and "
                            "number of actuators (%i). Please provide a mask "
                            "of the correct length" % (np.shape(self.pupil_ac)[0],
                                                       self.numActuators))

        self._wavefront_error_mode = self.wavefront_rms_error

    def _do_shutdown(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    @Pyro4.expose
    def enable_camera(self):
        self.wavefront_camera.enable()

    @Pyro4.expose
    def disable_camera(self):
        self.wavefront_camera.disable()

    def generate_isosense_pattern_image(self, shape, dist, wavelength, NA, pixel_size):
        try:
            assert type(shape) is tuple
        except:
            raise Exception("Expected %s instead recieved %s" % (type((512, 512)), type(shape)))

        try:
            assert len(shape) == 2
        except:
            raise Exception("Expected tuple of length 2, instead recieved length %i" % len(shape))

        ray_crit_dist = (1.22 * wavelength) / (2 * NA)
        ray_crit_freq = 1 / ray_crit_dist
        max_freq = 1 / (2 * pixel_size)
        freq_ratio = ray_crit_freq / max_freq
        OTF_outer_radx = freq_ratio * (shape[1] / 2)
        OTF_outer_rady = freq_ratio * (shape[0] / 2)

        pattern_ft = np.zeros(shape)

        f1x = shape[1] // 2
        f1y = shape[0] // 2
        f2x = f1x - int(np.round(0.5 * OTF_outer_radx * dist))
        f2y = f1y - int(np.round(0.5 * OTF_outer_rady * dist))
        f3x = f1x + int(np.round(0.5 * OTF_outer_radx * dist))
        f3y = f1y + int(np.round(0.5 * OTF_outer_rady * dist))
        f4x = f1x - int(np.round(OTF_outer_radx * dist))
        f4y = f1y - int(np.round(OTF_outer_rady * dist))
        f5x = f1x + int(np.round(OTF_outer_radx * dist))
        f5y = f1y + int(np.round(OTF_outer_rady * dist))
        freq_loc_half = (np.asarray([f2y, f2y, f3y, f3y], dtype="int64"),
                         np.asarray([f2x, f3x, f2x, f3x], dtype="int64"))
        freq_loc_quart = (np.asarray([f1y, f1y, f4y, f5y], dtype="int64"),
                          np.asarray([f4x, f5x, f1x, f1x], dtype="int64"))
        pattern_ft[f1y, f1x] = 1
        pattern_ft[freq_loc_half] = 1 / 2
        pattern_ft[freq_loc_quart] = 1 / 4

        pattern_unscaled = abs(np.fft.fft2(np.fft.ifftshift(pattern_ft)))
        pattern = (pattern_unscaled / np.max(pattern_unscaled)) * ((2 ** 16) - 1)
        pattern = pattern.astype("uint16")
        return pattern

    @Pyro4.expose
    def apply_isosense_pattern(self, fill_frac, wavelength, NA, pixel_size):

        if fill_frac < 0 :
            raise ValueError("Fill fraction must be greater than 0")
        elif fill_frac > 100:
            raise ValueError("Fill fraction must be less than 100")
        else:
            pass
        ## Tell the SLM to prepare the pattern sequence.
        dist = fill_frac/100
        shape = self.slm.get_shape()
        pattern = self.generate_isosense_pattern_image(shape=shape, wavelength=wavelength,
                                          dist=dist, NA=NA, pixel_size=pixel_size)
        self.slm.set_custom_sequence(wavelength,[pattern,pattern])

    @Pyro4.expose
    def set_trigger(self, cp_ttype, cp_tmode):
        ttype = self._CockpitTriggerType_to_TriggerType[cp_ttype]
        tmode = self._CockpitTriggerModes_to_TriggerModes[cp_tmode]
        self.ao_element.set_trigger(ttype, tmode)

        self.last_trigger_type = cp_ttype
        self.last_trigger_mode = cp_tmode

    @Pyro4.expose
    def get_trigger(self):
        return self.last_trigger_type, self.last_trigger_mode

    @Pyro4.expose
    def get_pattern_index(self):
        return self.ao_element.get_pattern_index()

    @Pyro4.expose
    def get_n_actuators(self):
        return self.numActuators

    @Pyro4.expose
    def set_pupil_ac(self, pupil_ac):
        try:
            assert np.shape(pupil_ac)[0] == self.numActuators
        except Exception:
            raise Exception("Length mismatch between pupil mask (%i) and "
                            "number of actuators (%i). Please provide a mask "
                            "of the correct length" % (np.shape(pupil_ac)[0],
                                                       self.numActuators))

        self.pupil_ac = pupil_ac

    @Pyro4.expose
    def get_pupil_ac(self):
        return self.pupil_ac

    @Pyro4.expose
    def get_all_unwrap_methods(self):
        return unwrap_method.keys()

    @Pyro4.expose
    def get_unwrap_method(self):
        return self.phase_method

    @Pyro4.expose
    def set_unwrap_method(self, phase_method):
        if not phase_method in unwrap_method:
            raise Exception("TypeError: Not a phase unwrapping method. Check available unwrap methods.")
        else:
            self.phase_method = phase_method

    @Pyro4.expose
    def get_all_wavefront_error_modes(self):
        return wavefront_error_modes

    @Pyro4.expose
    def get_wavefront_error_mode(self):
        if self._wavefront_error_mode is self.wavefront_rms_error:
            mode = wavefront_error_modes[0]
        elif self._wavefront_error_mode is self.wavefront_strehl_ratio:
            mode = wavefront_error_modes[1]
        return mode

    @Pyro4.expose
    def set_wavefront_error_mode(self,mode):
        if not mode in wavefront_error_modes:
            raise Exception("TypeError: Not a valid wavefront error mode")
        else:
            if mode == wavefront_error_modes[0]:
                self._wavefront_error_mode = self.wavefront_rms_error
            elif mode == wavefront_error_modes[1]:
                self._wavefront_error_mode = self.wavefront_strehl_ratio

    @Pyro4.expose
    def set_metric(self,metric):
        aoAlg.set_metric(metric)

    @Pyro4.expose
    def get_metric(self):
        metric = aoAlg.get_metric()
        _logger.info("Current image quality metric is: %s" %metric)
        return metric

    @Pyro4.expose
    def set_system_flat(self, system_flat):
        self.flat_actuators_sys = system_flat

    @Pyro4.expose
    def get_system_flat(self):
        return self.flat_actuators_sys

    # This method is used for AO elements such as DMs which have actuators which require direct signal values to be set.
    @Pyro4.expose
    def send(self, values):
        _logger.info("Sending pattern to AO element")

        ttype, tmode = self.get_trigger()
        if ttype is not "SOFTWARE":
            self.set_trigger(cp_ttype="SOFTWARE", cp_tmode="ONCE")

        # Need to normalise patterns because general DM class expects 0-1 values
        values[values > 1.0] = 1.0
        values[values < 0.0] = 0.0

        try:
            self.ao_element.apply_pattern(values)
        except Exception as e:
            raise e

        self.last_actuator_values = values
        if (ttype, tmode) is not self.get_trigger():
            self.set_trigger(cp_ttype=ttype, cp_tmode=tmode)

    # This method is for AO elements such as SLMs where the phase shape can be applied directly by sending an image of
    # the desired phase.
    @Pyro4.expose
    def apply_phase_pattern(self, wavelength, pattern):
        ao_shape = self.ao_element.get_shape()
        try:
            assert(ao_shape == pattern.shape)
        except:
            raise Exception("AO element shape is (%i,%i), recieved pattern of shape (%i,%i)"
                            %(ao_shape[0], ao_shape[1], pattern.shape[0], pattern.shape[1]))

        self.ao_element.set_custom_sequence(wavelength,[pattern,pattern])
        self.last_phase_pattern = pattern

    @Pyro4.expose
    def get_last_actuator_values(self):
        return self.last_actuator_values

    @Pyro4.expose
    def get_last_phase_pattern(self):
        return self.last_phase_pattern

    @Pyro4.expose
    def queue_patterns(self, patterns):
        _logger.info("Queuing patterns on DM")

        ttype, tmode = self.get_trigger()
        if ttype is not "RISING_EDGE":
            self.set_trigger(cp_ttype="RISING_EDGE", cp_tmode="ONCE")

        # Need to normalise patterns because general DM class expects 0-1 values
        patterns[patterns > 1.0] = 1.0
        patterns[patterns < 0.0] = 0.0

        try:
            self.ao_element.queue_patterns(patterns)
        except Exception as e:
            raise e

        self.last_actuator_patterns = patterns
        if (ttype, tmode) is not self.get_trigger():
            self.set_trigger(cp_ttype=ttype, cp_tmode=tmode)

    @Pyro4.expose
    def get_last_actuator_patterns(self):
        return self.last_actuator_patterns

    @Pyro4.expose
    def set_roi(self, y0, x0, radius):
        self.roi = (int(np.round(y0)), int(np.round(x0)), int(np.round(radius)))
        try:
            assert self.roi is not None
        except:
            raise Exception("ROI assignment failed")

        # Mask will need to be reconstructed as radius has changed
        self.mask = aoAlg.make_mask(self.roi[2])
        try:
            assert self.mask is not None
        except:
            raise Exception("Mask construction failed")

        # Fourier filter should be erased, as it's probably wrong.
        ##Might be unnecessary
        self.fft_filter = None
        return

    @Pyro4.expose
    def get_roi(self):
        if np.any(self.roi) is None:
            raise Exception("No region of interest selected. Please select a region of interest")
        else:
            return self.roi

    @Pyro4.expose
    def get_fourierfilter(self):
        if np.any(self.fft_filter) is None:
            raise Exception("Fourier filter is None. Please construct Fourier filter")
        else:
            return self.fft_filter

    @Pyro4.expose
    def get_controlMatrix(self):
        if np.any(self.controlMatrix) is None:
            raise Exception("No control matrix calculated. Please calibrate the AO element")
        else:
            return self.controlMatrix

    @Pyro4.expose
    def set_controlMatrix(self, controlMatrix):
        self.controlMatrix = controlMatrix
        aoAlg.set_controlMatrix(controlMatrix)
        return

    @Pyro4.expose
    def reset(self):
        _logger.info("Resetting DM")
        last_ac = np.copy(self.last_actuator_values)
        self.send(np.zeros(self.numActuators) + 0.5)
        self.last_actuator_values = last_ac

    @Pyro4.expose
    def make_mask(self, radius):
        self.mask = aoAlg.make_mask(radius)
        return self.mask

    @Pyro4.expose
    @_with_wavefront_camera_ttype_software
    def acquire_raw(self):
        """This method changes trigger type to software.  If something is
        planning on calling this method multiple times in a row it
        should ensure that it sets software trigger type itself
        otherwise the enable/disable cycle that it involves will take
        a lot of time.
        """
        # FIXME: this can loop forever if the camera keeps timing out.
        # It's unlikely that this is the right thing to do.
        while True:
            try:
                data_raw, _ = self.wavefront_camera.grab_next_data()
                break
            except Exception as e:
                # FIXME: this only catches the error from Ximea
                # cameras (I'm not sure it still does).  We should not
                # be trying to handle hardware specific exceptions.
                if str(e) == "ERROR 10: Timeout":
                    _logger.info("Received Timeout error from camera. Waiting to try again...")
                    time.sleep(1)
                else:
                    _logger.info(type(e))
                    _logger.info("Error is: %s" % (e))
                    raise e
        return data_raw

    @Pyro4.expose
    def acquire(self):
        data_raw = self.acquire_raw()
        if np.any(self.roi) is None:
            data = data_raw
        else:
            data_cropped = np.zeros((self.roi[2] * 2, self.roi[2] * 2), dtype=float)
            data_cropped[:, :] = data_raw[self.roi[0] - self.roi[2]:self.roi[0] + self.roi[2],
                                 self.roi[1] - self.roi[2]:self.roi[1] + self.roi[2]]
            if np.any(self.mask) is None:
                self.mask = self.make_mask(self.roi[2])
                data = data_cropped
            else:
                data = data_cropped * self.mask
        return data

    @Pyro4.expose
    def set_fourierfilter(self, test_image, region=None, window_dim=None, mask_di=None):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        try:
            self.fft_filter = aoAlg.make_fft_filter(test_image, region=region, window_dim=window_dim, mask_di=mask_di)
        except Exception as e:
            _logger.info(e)
        return self.fft_filter

    @Pyro4.expose
    def check_unwrap_conditions(self, image=None):
        if self.phase_method == 'interferometry':
            if np.any(self.mask) is None:
                raise Exception("Mask is None. Please construct mask.")
            else:
                pass
            if np.any(self.fft_filter) is None:
                if image is not None:
                    self.set_fourierfilter(image)
                else:
                    raise Exception("Fourier filter is None. Please construct Fourier filter")
            else:
                pass

    @Pyro4.expose
    def phaseunwrap(self, image=None):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        if np.any(image) is None:
            image = self.acquire()

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        out = unwrap_method[self.phase_method](image)
        return out

    @Pyro4.expose
    def getzernikemodes(self, image_unwrap, noZernikeModes, resize_dim=128):
        coef = aoAlg.get_zernike_modes(image_unwrap, noZernikeModes, resize_dim=resize_dim)
        return coef

    @Pyro4.expose
    def createcontrolmatrix(self, imageStack, noZernikeModes, pokeSteps, pupil_ac=None, threshold=0.005):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions(imageStack[0,:,:])

        if np.any(pupil_ac == None):
            pupil_ac = self.pupil_ac
        else:
            pass

        noImages, y, x = np.shape(imageStack)
        numPokeSteps = len(pokeSteps)

        assert x == y
        edge_mask = np.sqrt(
            (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1)) + (np.arange(-x / 2.0, x / 2.0) ** 2)) < ((x / 2.0) - 3)
        all_zernikeModeAmp = []
        all_pokeAmps = []

        curr_calc = 0
        for ac in range(self.numActuators):
            image_stack_cropped = np.zeros((numPokeSteps, y, x))
            unwrapped_stack_cropped = np.zeros((numPokeSteps, y, x))

            # Determine if the current actuator is in the pupil
            if pupil_ac[ac] == 1:
                pokeAc = np.zeros(self.numActuators)
                zernikeModeAmp_list = []

                for im in range(numPokeSteps):
                    curr_calc += 1
                    # Acquire the current poke image
                    poke_image = imageStack[curr_calc - 1, :, :]
                    image_stack_cropped[im, :, :] = poke_image

                    # Unwrap the current image
                    image_unwrap = unwrap_method[self.phase_method](poke_image)
                    unwrapped_stack_cropped[im, :, :] = image_unwrap

                    # Check the current phase map for discontinuities which can interfere with the Zernike mode measurements
                    diff_image = abs(np.diff(np.diff(image_unwrap, axis=1), axis=0)) * edge_mask[:-1, :-1]
                    no_discontinuities = np.shape(np.where(diff_image > 2 * np.pi))[1]
                    if no_discontinuities > (x * y) / 1000.0:
                        _logger.info("Unwrap image %d/%d contained discontinuites" % (curr_calc, noImages))
                        _logger.info("Zernike modes %d/%d not calculated" % (curr_calc, noImages))
                    else:
                        pokeAc[ac] = pokeSteps[im]
                        all_pokeAmps.append(pokeAc.tolist())
                        _logger.info("Calculating Zernike modes %d/%d..." % (curr_calc, noImages))

                    curr_amps = aoAlg.get_zernike_modes(image_unwrap, noZernikeModes)
                    zernikeModeAmp_list.append(curr_amps)
                    all_zernikeModeAmp.append(curr_amps)
            np.save("image_stack_cropped_ac_%i" % ac, image_stack_cropped)
            np.save("unwrap_stack_cropped_ac_%i" % ac, unwrapped_stack_cropped)

        all_zernikeModeAmp = np.asarray(all_zernikeModeAmp)
        all_pokeAmps = np.asarray(all_pokeAmps)

        _logger.info("Computing Control Matrix")
        self.controlMatrix = aoAlg.create_control_matrix(zernikeAmps=all_zernikeModeAmp,
                                                         pokeSteps=all_pokeAmps,
                                                         numActuators=self.numActuators,
                                                         pupil_ac=self.pupil_ac,
                                                         threshold=threshold)
        _logger.info("Control Matrix computed")
        return self.controlMatrix

    @Pyro4.expose
    def acquire_unwrapped_phase(self):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        interferogram = self.acquire()
        interferogram_unwrap = self.phaseunwrap(interferogram)
        _logger.info("Phase unwrapped ")
        return interferogram, interferogram_unwrap

    @Pyro4.expose
    def measure_zernike(self, noZernikeModes):
        interferogram, unwrapped_phase = self.acquire_unwrapped_phase()
        zernike_amps = self.getzernikemodes(unwrapped_phase, noZernikeModes)
        return zernike_amps

    @Pyro4.expose
    def wavefront_rms_error(self, phase_map=None):
        if phase_map is None:
            phase_map = self.acquire_unwrapped_phase()

        if self.mask is None:
            self.make_mask(phase_map.shape//2)

        true_flat = np.zeros(np.shape(phase_map))
        rms_error = np.sqrt(np.mean((true_flat[self.mask] - phase_map[self.mask]) ** 2))
        return rms_error

    @Pyro4.expose
    def wavefront_strehl_ratio(self, phase_map=None):
        if phase_map is None:
            phase_map = self.acquire_unwrapped_phase()

        if self.mask is None:
            self.make_mask(phase_map.shape//2)

        strehl_ratio = np.exp(-np.mean((phase_map[self.mask] - np.mean(phase_map[self.mask])) ** 2))
        return strehl_ratio

    @Pyro4.expose
    @_with_wavefront_camera_ttype_software
    def calibrate(self, numPokeSteps=5, noZernikeModes=69, threshold=0.005):
        self.wavefront_camera.set_exposure_time(0.1)
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        test_image = np.asarray(self.acquire())
        (y, x) = np.shape(test_image)

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions(test_image)

        nzernike = self.numActuators

        poke_min = 0.25
        poke_max = 0.75
        pokeSteps = np.linspace(poke_min, poke_max, numPokeSteps)
        noImages = numPokeSteps * (np.shape(np.where(self.pupil_ac == 1))[1])

        assert x == y
        edge_mask = np.sqrt(
            (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1)) + (np.arange(-x / 2.0, x / 2.0) ** 2)) < ((x / 2.0) - 3)
        all_zernikeModeAmp = []
        all_pokeAmps = []

        actuator_values = np.zeros((noImages, nzernike)) + 0.5
        for ii in range(nzernike):
            for jj in range(numPokeSteps):
                actuator_values[(numPokeSteps * ii) + jj, ii] = pokeSteps[jj]

        curr_calc = 0
        for ac in range(self.numActuators):
            image_stack_cropped = np.zeros((numPokeSteps, y, x))
            unwrapped_stack_cropped = np.zeros((numPokeSteps, y, x))

            # Determine if the current actuator is in the pupil
            if self.pupil_ac[ac] == 1:
                pokeAc = np.zeros(self.numActuators)
                zernikeModeAmp_list = []

                for im in range(numPokeSteps):
                    curr_calc += 1
                    try:
                        self.send(actuator_values[(curr_calc - 1), :])
                    except:
                        _logger.info("Actuator values being sent:")
                        _logger.info(actuator_values[(curr_calc - 1), :])
                        _logger.info("Shape of actuator vector:")
                        _logger.info(np.shape(actuator_values[(curr_calc - 1), :]))
                    _logger.info("Frame %i/%i captured" % (curr_calc, noImages))

                    # Acquire the current poke image
                    poke_image = self.acquire()
                    image_stack_cropped[im, :, :] = poke_image

                    # Unwrap the current image
                    image_unwrap = unwrap_method[self.phase_method](poke_image)
                    unwrapped_stack_cropped[im, :, :] = image_unwrap

                    # Check the current phase map for discontinuities which can interfere with the Zernike mode measurements
                    diff_image = abs(np.diff(np.diff(image_unwrap, axis=1), axis=0)) * edge_mask[:-1, :-1]
                    no_discontinuities = np.shape(np.where(diff_image > 2 * np.pi))[1]
                    if no_discontinuities > (x * y) / 1000.0:
                        _logger.info("Unwrap image %d/%d contained discontinuites" % (curr_calc, noImages))
                        _logger.info("Zernike modes %d/%d not calculated" % (curr_calc, noImages))
                    else:
                        pokeAc[ac] = pokeSteps[im]
                        all_pokeAmps.append(pokeAc.tolist())
                        _logger.info("Calculating Zernike modes %d/%d..." % (curr_calc, noImages))

                    curr_amps = aoAlg.get_zernike_modes(image_unwrap, noZernikeModes)
                    zernikeModeAmp_list.append(curr_amps)
                    all_zernikeModeAmp.append(curr_amps)
            zernikeModeAmp = np.asarray(zernikeModeAmp_list)
            np.save("image_stack_cropped_ac_%i" % ac, image_stack_cropped)
            np.save("unwrap_stack_cropped_ac_%i" % ac, unwrapped_stack_cropped)
            np.save("zernikeModeAmp_ac_%i" % ac, zernikeModeAmp)

        all_zernikeModeAmp = np.asarray(all_zernikeModeAmp)
        all_pokeAmps = np.asarray(all_pokeAmps)
        np.save("all_zernikeModeAmp", all_zernikeModeAmp)
        np.save("all_pokeAmps", all_pokeAmps)
        self.reset()

        _logger.info("Computing Control Matrix")
        self.controlMatrix = aoAlg.create_control_matrix(zernikeAmps=all_zernikeModeAmp,
                                                         pokeSteps=all_pokeAmps,
                                                         numActuators=self.numActuators,
                                                         pupil_ac=self.pupil_ac,
                                                         threshold=threshold)
        _logger.info("Control Matrix computed")
        np.save("control_matrix", self.controlMatrix)
        return self.controlMatrix

    # This method of wavefront flattening should be used when the wavefront sensor defined in __init__ is being used to
    # directly measure the phase wavefront.
    @Pyro4.expose
    @_with_wavefront_camera_ttype_software
    def flatten_phase(self, iterations=1, error_thresh=np.inf, z_modes_ignore=None):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        # Check dimensions match
        numActuators, nzernike = np.shape(self.controlMatrix)
        try:
            assert numActuators == self.numActuators
        except:
            raise Exception("Control Matrix dimension 0 axis and number of "
                            "actuators do not match.")
        # Set which modes to ignore while flattening
        if np.any(z_modes_ignore) is None:
            # By default, ignore piston, tip and tilt
            z_modes_ignore = np.asarray(range(nzernike)) > 2
        else:
            # If we have more Zernike modes to ignore than in the control matrix, crop to the correct number
            if len(z_modes_ignore) > nzernike:
                z_modes_ignore = z_modes_ignore[:nzernike]
            # If we have fewer Zernike modes to ignore than in the control matrix, pad with zeros (i.e. ignore all the
            # Zernike modes the user didn't specify)
            elif len(z_modes_ignore) < nzernike:
                z_modes_ignore = np.pad(z_modes_ignore, (0,len(z_modes_ignore)-nzernike),
                                        mode="constant", constant_values=0)
            else:
                pass

        best_flat_actuators = np.zeros(numActuators) + 0.5
        best_z_amps_corrected = np.zeros(nzernike)
        self.send(best_flat_actuators)
        # Get a measure of the RMS phase error of the uncorrected wavefront
        # The corrected wavefront should be better than this
        image = self.acquire()
        x, y = image.shape
        assert x == y
        best_error = np.inf
        ii = 0
        while (iterations > ii) or (best_error > error_thresh):
            _logger.info("Correction iteration %i/%i" % (ii + 1, iterations))
            # Measure the current wavefront and calculate the Zernike modes to apply to correct
            image = self.acquire()
            correction_wavefront = unwrap_method[self.phase_method](image)
            edge_mask = np.sqrt(
                (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1)) + (np.arange(-x / 2.0, x / 2.0) ** 2)) < (
                                (x / 2.0) - 3)
            diff_correction_wavefront = abs(np.diff(np.diff(correction_wavefront, axis=1), axis=0)) * edge_mask[:-1, :-1]
            no_discontinuities_correction = np.shape(np.where(diff_correction_wavefront > 2 * np.pi))[1]
            z_amps = self.getzernikemodes(correction_wavefront, nzernike)

            ## Here we ignore piston, tip and tilt since they are not true optical aberrations
            correction_wavefront_mptt = correction_wavefront - aotools.phaseFromZernikes(z_amps[0:3], x)
            current_error = self._wavefront_error_mode(correction_wavefront_mptt)
            z_amps = z_amps * z_modes_ignore
            flat_actuators = self.set_phase((-1.0 * z_amps), offset=best_flat_actuators)
            time.sleep(1)

            # Now that the wavefront is corrected, measure it again and calculate RMS deformation
            image = self.acquire()
            corrected_wavefront = unwrap_method[self.phase_method](image)
            diff_corrected_wavefront = abs(np.diff(np.diff(corrected_wavefront, axis=1), axis=0)) * edge_mask[:-1,
                                                                                                      :-1]
            no_discontinuities_corrected = np.shape(np.where(diff_corrected_wavefront > 2 * np.pi))[1]
            z_amps_corrected = self.getzernikemodes(corrected_wavefront, nzernike)

            ## Here we ignore piston, tip and tilt since they are not true optical aberrations
            corrected_wavefront_mptt = corrected_wavefront - aotools.phaseFromZernikes(z_amps_corrected[0:3], x)
            corrected_error = self._wavefront_error_mode(corrected_wavefront_mptt)
            _logger.info("Current wavefront error is %.5f. Wavefront error before correction %.5f."
                         "Best is %.5f" % (corrected_error, current_error, best_error))
            if corrected_error < best_error and corrected_error < current_error:
                if no_discontinuities_corrected > ((x * y) / 1000.0):
                    _logger.info("Too many discontinuities in wavefront unwrap")
                else:
                    best_flat_actuators = np.copy(flat_actuators)
                    best_z_amps_corrected = np.copy(z_amps)
                    best_error = np.copy(corrected_error)
            elif corrected_error < best_error:
                _logger.info("Wavefront error worse than before")
            else:
                _logger.info("No improvement in Wavefront error")
            self.send(best_flat_actuators)
            ii += 1
        self.send(best_flat_actuators)
        return best_flat_actuators, best_z_amps_corrected

    @Pyro4.expose
    def set_phase(self, applied_z_modes, offset=None):
        try:
            actuator_pos = aoAlg.ac_pos_from_zernike(applied_z_modes, self.numActuators)
        except Exception as err:
            _logger.info(err)
            raise err
        if np.any(offset) is None:
            actuator_pos += 0.5
        else:
            actuator_pos += offset
        self.send(actuator_pos)
        return actuator_pos

    @Pyro4.expose
    @_with_wavefront_camera_ttype_software
    def assess_character(self, modes_tba=None):
        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        if modes_tba is None:
            modes_tba = self.controlMatrix.shape[1]
        assay = np.zeros((modes_tba, modes_tba))
        applied_z_modes = np.zeros(modes_tba)
        for ii in range(modes_tba):
            self.reset()
            z_modes_ac0 = self.measure_zernike(modes_tba)
            applied_z_modes[ii] = 1
            self.set_phase(applied_z_modes)
            _logger.info("Appling Zernike mode %i/%i" % (ii+1, modes_tba))
            acquired_z_modes = self.measure_zernike(modes_tba)
            _logger.info("Measured phase")
            assay[:, ii] = acquired_z_modes - z_modes_ac0
            applied_z_modes[ii] = 0.0
        self.reset()
        return assay

    @Pyro4.expose
    def correct_direct_sensing(self, image, z_modes_ignore=None):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure the conditions for phase unwrapping are in satisfied
        self.check_unwrap_conditions()

        # Check dimensions match
        numActuators, nzernike = np.shape(self.controlMatrix)
        try:
            assert numActuators == self.numActuators
        except:
            raise Exception("Control Matrix dimension 0 axis and number of "
                            "actuators do not match.")
        # Set which modes to ignore while flattening
        if np.any(z_modes_ignore) is None:
            # By default, ignore piston, tip and tilt
            z_modes_ignore = np.asarray(range(nzernike)) > 2
        else:
            # If we have more Zernike modes to ignore than in the control matrix, crop to the correct number
            if len(z_modes_ignore) > nzernike:
                z_modes_ignore = z_modes_ignore[:nzernike]
            # If we have fewer Zernike modes to ignore than in the control matrix, pad with zeros (i.e. ignore all the
            # Zernike modes the user didn't specify)
            elif len(z_modes_ignore) < nzernike:
                z_modes_ignore = np.pad(z_modes_ignore, (0, len(z_modes_ignore) - nzernike),
                                        mode="constant", constant_values=0)
            else:
                pass

        x, y = image.shape
        assert x == y

        correction_wavefront = unwrap_method[self.phase_method](image)
        edge_mask = np.sqrt(
            (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1)) + (np.arange(-x / 2.0, x / 2.0) ** 2)) < (
                            (x / 2.0) - 3)
        diff_correction_wavefront = abs(np.diff(np.diff(correction_wavefront, axis=1), axis=0)) * edge_mask[:-1, :-1]
        no_discontinuities_correction = np.shape(np.where(diff_correction_wavefront > 2 * np.pi))[1]
        if no_discontinuities_correction > ((x * y) / 1000.0):
            _logger.info("Too many discontinuities in wavefront unwrap")
            return
        else:
            z_amps = self.getzernikemodes(correction_wavefront, nzernike)
            z_amps = z_amps * z_modes_ignore
            flat_actuators = self.set_phase((-1.0 * z_amps), offset=self.last_actuator_patterns)
            return z_amps, flat_actuators

    @Pyro4.expose
    def measure_metric(self, image, **kwargs):
        metric = aoAlg.measure_metric(image, **kwargs)
        return metric

    @Pyro4.expose
    def correct_sensorless_single_mode(self, image_stack, zernike_applied, nollIndex,
                                       wavelength, NA, pixel_size, offset=None):
        z_amps = np.zeros(self.numActuators)
        amp_to_correct = aoAlg.find_zernike_amp_sensorless(image_stack, zernike_applied, wavelength=wavelength, NA=NA,
                                                           pixel_size=pixel_size)
        if abs(amp_to_correct) <= 1.5 * (np.max(zernike_applied)-np.min(zernike_applied)):
            _logger.info("Amplitude calculated = %f" % amp_to_correct)
        else:
            _logger.info("Amplitude calculated = %f" % amp_to_correct)
            _logger.info("Amplitude magnitude too large. Defaulting to 0.")
            amp_to_correct = 0
        z_amps[nollIndex - 1] = -1.0 * amp_to_correct
        if np.any(offset) == None:
            ac_pos_correcting = self.set_phase(z_amps)
        else:
            ac_pos_correcting = self.set_phase(z_amps, offset=offset)
        return amp_to_correct, ac_pos_correcting

    @Pyro4.expose
    def correct_sensorless_all_modes(self, full_image_stack, full_zernike_applied, nollZernike, wavelength, NA,
                                     pixel_size, offset=None):
        # May change this function later if we hand control of other cameras to the composite device
        coef = aoAlg.get_zernike_modes_sensorless(full_image_stack, full_zernike_applied, nollZernike=nollZernike,
                                                  wavelength=wavelength, NA=NA, pixel_size=pixel_size)
        if np.any(offset) is None:
            ac_pos_correcting = self.set_phase(coef)
        else:
            ac_pos_correcting = self.set_phase(coef, offset=offset)
        return coef, ac_pos_correcting
