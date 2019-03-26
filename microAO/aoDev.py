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
import Pyro4
import time
from microAO.aoAlg import AdaptiveOpticsFunctions

#Should fix this with multiple inheritance for this class!
aoAlg = AdaptiveOpticsFunctions()

from microscope.devices import Device
from microscope.devices import TriggerType
from microscope.devices import TriggerMode

class AdaptiveOpticsDevice(Device):
    """Class for the adaptive optics device

    This class requires a mirror and a camera. Everything else is generated
    on or after __init__"""
    
    _CockpitTriggerType_to_TriggerType = {
    "SOFTWARE" : TriggerType.SOFTWARE,
    "RISING_EDGE" : TriggerType.RISING_EDGE,
    "FALLING_EDGE" : TriggerType.FALLING_EDGE,
    }

    _CockpitTriggerModes_to_TriggerModes = {
    "ONCE" : TriggerMode.ONCE,
    "START" : TriggerMode.START,
    }

    def __init__(self, wavefront_uri, mirror_uri, **kwargs):
        # Init will fail if devices it depends on aren't already running, but
        # deviceserver should retry automatically.
        super(AdaptiveOpticsDevice, self).__init__(**kwargs)
        # Wavefront sensor. Must support soft_trigger for now.
        self.wavefront_camera = Pyro4.Proxy('PYRO:%s@%s:%d' %(wavefront_uri[0].__name__,
                                                    wavefront_uri[1], wavefront_uri[2]))
        self.wavefront_camera.enable()
        # Deformable mirror device.
        self.mirror = Pyro4.Proxy('PYRO:%s@%s:%d' %(mirror_uri[0].__name__,
                                                mirror_uri[1], mirror_uri[2]))
        #self.mirror.set_trigger(TriggerType.RISING_EDGE) #Set trigger type to rising edge
        self.numActuators = self.mirror.n_actuators
        # Region of interest (i.e. pupil offset and radius) on camera.
        self.roi = None
        #Mask for the interferometric data
        self.mask = None
        #Mask to select phase information
        self.fft_filter = None
        #Control Matrix
        self.controlMatrix = None
        #System correction
        self.flat_actuators_sys = np.zeros(self.numActuators)
        #Last applied actuators values
        self.last_actuator_values = None
        # Last applied actuators pattern
        self.last_actuator_patterns = None

        ##We don't use all the actuators. Create a mask for the actuators outside
        ##the pupil so we can selectively calibrate them. 0 denotes actuators at
        ##the edge, i.e. outside the pupil, and 1 denotes actuators in the pupil

        #Use this if all actuators are being used
        #self.pupil_ac = np.ones(self.numActuators)

        #Preliminary mask for DeepSIM
        self.pupil_ac = np.ones(self.numActuators)

        try:
            assert np.shape(self.pupil_ac)[0] == self.numActuators
        except:
            raise Exception("Length mismatch between pupil mask (%i) and "
                            "number of actuators (%i). Please provide a mask "
                            "of the correct length" %(np.shape(self.pupil_ac)[0],
                                                      self.numActuators))

    def _on_shutdown(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    @Pyro4.expose
    def set_trigger(self, cp_ttype, cp_tmode):
        ttype = self._CockpitTriggerType_to_TriggerType[cp_ttype]
        tmode = self._CockpitTriggerModes_to_TriggerModes[cp_tmode]
        self.mirror.set_trigger(ttype, tmode)

    @Pyro4.expose
    def get_pattern_index(self):
        return self.mirror.get_pattern_index()

    @Pyro4.expose
    def get_n_actuators(self):
        return self.numActuators

    @Pyro4.expose
    def send(self, values):
        #Need to normalise patterns because general DM class expects 0-1 values
        values[values > 1.0] = 1.0
        values[values < 0.0] = 0.0

        try:
            self.mirror.apply_pattern(values)
        except Exception as e:
            self._logger.info(e)

        self.last_actuator_values = values

    @Pyro4.expose
    def get_last_actuator_values(self):
        return self.last_actuator_values

    @Pyro4.expose
    def queue_patterns(self, patterns):
        self._logger.info("Queuing patterns on DM")

        # Need to normalise patterns because general DM class expects 0-1 values
        patterns[patterns > 1.0] = 1.0
        patterns[patterns < 0.0] = 0.0

        try:
            self.mirror.queue_patterns(patterns)
        except Exception as e:
            self._logger.info(e)

        self.last_actuator_patterns = patterns

    @Pyro4.expose
    def get_last_actuator_patterns(self):
        return self.last_actuator_patterns

    @Pyro4.expose
    def set_roi(self, y0, x0, radius):
        self.roi = (y0, x0, radius)
        try:
            assert self.roi is not None
        except:
            raise Exception("ROI assignment failed")

        #Mask will need to be reconstructed as radius has changed
        self.mask = aoAlg.make_mask(radius)
        try:
            assert self.mask is not None
        except:
            raise Exception("Mask construction failed")

        #Fourier filter should be erased, as it's probably wrong. 
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
            raise Exception("No Fourier filter created. Please create one.")
        else:
            return self.fft_filter

    @Pyro4.expose
    def get_controlMatrix(self):
        if np.any(self.controlMatrix) is None:
            raise Exception("No control matrix calculated. Please calibrate the mirror")
        else:
            return self.controlMatrix


    @Pyro4.expose
    def set_controlMatrix(self,controlMatrix):
        self.controlMatrix = controlMatrix
        aoAlg.set_controlMatrix(controlMatrix)
        return

    @Pyro4.expose
    def reset(self):
        self._logger.info("Resetting DM")
        self.send(np.zeros(self.numActuators) + 0.5)

    @Pyro4.expose
    def make_mask(self, radius):
        self.mask = aoAlg.make_mask(radius)
        return self.mask

    @Pyro4.expose
    def acquire_raw(self):
        self.acquiring = True
        while self.acquiring == True:
            try:
                data_raw, timestamp = self.wavefront_camera.grab_next_data()
                self.acquiring = False
            except Exception as e:
                if str(e) == str("ERROR 10: Timeout"):
                    self._logger.info("Recieved Timeout error from camera. Waiting to try again...")
                    time.sleep(1)
                else:
                    self._logger.info(type(e))
                    self._logger.info("Error is: %s" %(e))
                    raise e
        return data_raw

    @Pyro4.expose
    def acquire(self):
        self.acquiring = True
        while self.acquiring == True:
            try:
                data_raw, timestamp = self.wavefront_camera.grab_next_data()
                self.acquiring = False
            except Exception as e:
                if str(e) == str("ERROR 10: Timeout"):
                    self._logger.info("Recieved Timeout error from camera. Waiting to try again...")
                    time.sleep(1)
                else:
                    self._logger.info(type(e))
                    self._logger.info("Error is: %s" %(e))
                    raise e
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
    def set_fourierfilter(self, test_image, region=None):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        try:
            self.fft_filter = aoAlg.make_fft_filter(test_image, region=region)
        except Exception as e:
            self._logger.info(e)
        return self.fft_filter

    @Pyro4.expose
    def phaseunwrap(self, image = None):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        if np.any(image) is None:
            image = self.acquire()

        #Ensure the filters has been constructed
        if np.any(self.mask) is None:
            self.mask = self.make_mask(int(np.round(np.shape(image)[0] / 2)))
        else:
            pass

        if np.any(self.fft_filter) is None:
            try:
                self.fft_filter = self.set_fourierfilter(image)
            except:
                raise
        else:
            pass

        self.out = aoAlg.phase_unwrap(image)
        return self.out


    @Pyro4.expose
    def getzernikemodes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        coef = aoAlg.get_zernike_modes(image_unwrap, noZernikeModes, resize_dim = resize_dim)
        return coef

    @Pyro4.expose
    def createcontrolmatrix(self, imageStack, noZernikeModes, pokeSteps, pupil_ac=None, threshold=0.005):
        # Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure the filters has been constructed
        if np.any(self.mask) is None:
            self.mask = self.make_mask(int(np.round(np.shape(imageStack)[1] / 2)))
        else:
            pass

        if np.any(self.fft_filter) is None:
            self.fft_filter = self.set_fourierfilter(imageStack[0, :, :])
        else:
            pass

        if np.any(pupil_ac == None):
            pupil_ac = np.ones(self.numActuators)
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
            if self.pupil_ac[ac] == 1:
                pokeAc = np.zeros(self.numActuators)
                zernikeModeAmp_list = []

                for im in range(numPokeSteps):
                    curr_calc += 1
                    # Acquire the current poke image
                    poke_image = imageStack[curr_calc - 1, :, :]
                    image_stack_cropped[im, :, :] = poke_image

                    # Unwrap the current image
                    image_unwrap = aoAlg.phase_unwrap(poke_image)
                    unwrapped_stack_cropped[im, :, :] = image_unwrap

                    # Check the current phase map for discontinuities which can interfere with the Zernike mode measurements
                    diff_image = abs(np.diff(np.diff(image_unwrap, axis=1), axis=0)) * edge_mask[:-1, :-1]
                    no_discontinuities = np.shape(np.where(diff_image > 2 * np.pi))[1]
                    if no_discontinuities > (x * y) / 1000.0:
                        self._logger.info("Unwrap image %d/%d contained discontinuites" % (curr_calc, noImages))
                        self._logger.info("Zernike modes %d/%d not calculated" % (curr_calc, noImages))
                    else:
                        pokeAc[ac] = pokeSteps[im]
                        all_pokeAmps.append(pokeAc.tolist())
                        self._logger.info("Calculating Zernike modes %d/%d..." % (curr_calc, noImages))

                    curr_amps = aoAlg.get_zernike_modes(image_unwrap, noZernikeModes)
                    zernikeModeAmp_list.append(curr_amps)
                    all_zernikeModeAmp.append(curr_amps)
            np.save("image_stack_cropped_ac_%i" % ac, image_stack_cropped)
            np.save("unwrap_stack_cropped_ac_%i" % ac, unwrapped_stack_cropped)

        all_zernikeModeAmp = np.asarray(all_zernikeModeAmp)
        all_pokeAmps = np.asarray(all_pokeAmps)

        self._logger.info("Computing Control Matrix")
        self.controlMatrix = aoAlg.create_control_matrix(zernikeAmps=all_zernikeModeAmp,
                                                         pokeSteps=all_pokeAmps,
                                                         numActuators=self.numActuators,
                                                         pupil_ac=self.pupil_ac,
                                                         threshold=threshold)
        self._logger.info("Control Matrix computed")
        return self.controlMatrix

    @Pyro4.expose
    def acquire_unwrapped_phase(self):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure a Fourier filter has been constructed
        if np.any(self.fft_filter) is None:
            try:
                test_image = self.acquire()
                self.fft_filter = self.set_fourierfilter(test_image)
            except:
                raise
        else:
            pass

        interferogram = self.acquire()
        interferogram_unwrap = self.phaseunwrap(interferogram)
        self._logger.info("Phase unwrapped ")
        return interferogram, interferogram_unwrap

    @Pyro4.expose
    def measure_zernike(self,noZernikeModes):
        interferogram, unwrapped_phase = self.acquire_unwrapped_phase()
        zernike_amps = self.getzernikemodes(unwrapped_phase,noZernikeModes)
        return zernike_amps

    @Pyro4.expose
    def wavefront_rms_error(self):
        phase_map = self.acquire_unwrapped_phase()
        true_flat = np.zeros(np.shape(phase_map))
        rms_error = np.sqrt(np.mean((true_flat - phase_map)**2))
        return rms_error

    @Pyro4.expose
    def calibrate(self, numPokeSteps = 5, noZernikeModes = 69, threshold = 0.005):
        self.wavefront_camera.set_exposure_time(0.1)
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        test_image = np.asarray(self.acquire())
        (y, x) = np.shape(test_image)

        #Ensure the filters has been constructed
        if np.any(self.mask) is None:
            self._logger.info("Constructing mask")
            self.mask = self.make_mask(self.roi[2])
        else:
            pass
            
        if np.any(self.fft_filter) is None:
            self._logger.info("Constructing Fourier filter")
            self.fft_filter = self.set_fourierfilter(test_image[:, :])
        else:
            pass

        nzernike = self.numActuators

        poke_min = 0.25
        poke_max = 0.75
        pokeSteps = np.linspace(poke_min,poke_max,numPokeSteps)
        noImages = numPokeSteps*(np.shape(np.where(self.pupil_ac == 1))[1])

        assert x == y
        edge_mask = np.sqrt(
            (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1)) + (np.arange(-x / 2.0, x / 2.0) ** 2)) < ((x / 2.0) - 3)
        all_zernikeModeAmp = []
        all_pokeAmps = []

        actuator_values = np.zeros((noImages,nzernike)) + 0.5
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
                        self.send(actuator_values[(curr_calc-1),:])
                    except:
                        self._logger.info("Actuator values being sent:")
                        self._logger.info(actuator_values[(curr_calc-1),:])
                        self._logger.info("Shape of actuator vector:")
                        self._logger.info(np.shape(actuator_values[(curr_calc-1),:]))
                    self._logger.info("Frame %i/%i captured" % (curr_calc, noImages))

                    # Acquire the current poke image
                    poke_image = self.acquire()
                    image_stack_cropped[im, :, :] = poke_image

                    # Unwrap the current image
                    image_unwrap = aoAlg.phase_unwrap(poke_image)
                    unwrapped_stack_cropped[im, :, :] = image_unwrap

                    # Check the current phase map for discontinuities which can interfere with the Zernike mode measurements
                    diff_image = abs(np.diff(np.diff(image_unwrap, axis=1), axis=0)) * edge_mask[:-1, :-1]
                    no_discontinuities = np.shape(np.where(diff_image > 2 * np.pi))[1]
                    if no_discontinuities > (x * y) / 1000.0:
                        self._logger.info("Unwrap image %d/%d contained discontinuites" % (curr_calc, noImages))
                        self._logger.info("Zernike modes %d/%d not calculated" % (curr_calc, noImages))
                    else:
                        pokeAc[ac] = pokeSteps[im]
                        all_pokeAmps.append(pokeAc.tolist())
                        self._logger.info("Calculating Zernike modes %d/%d..." % (curr_calc, noImages))

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

        self._logger.info("Computing Control Matrix")
        self.controlMatrix = aoAlg.create_control_matrix(zernikeAmps = all_zernikeModeAmp,
                                                         pokeSteps = all_pokeAmps,
                                                         numActuators = self.numActuators,
                                                         pupil_ac = self.pupil_ac,
                                                         threshold = threshold)
        self._logger.info("Control Matrix computed")
        np.save("control_matrix", self.controlMatrix)

        # Obtain actuator positions to correct for system aberrations
        # Ignore piston, tip, tilt and defocus
        z_modes_ignore = np.asarray(range(self.numActuators) > 3)
        self.flat_actuators_sys = self.flatten_phase(iterations=25, z_modes_ignore=z_modes_ignore)

        return self.controlMatrix, self.flat_actuators_sys

    @Pyro4.expose
    def flatten_phase(self, iterations = 1, z_modes_ignore = None):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert np.any(self.roi) is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        # Ensure a Fourier filter has been constructed
        if np.any(self.fft_filter) is None:
            try:
                test_image = self.acquire()
                self.fft_filter = self.set_fourierfilter(test_image)
            except:
                raise
        else:
            pass

        #Check dimensions match
        numActuators, nzernike = np.shape(self.controlMatrix)
        try:
            assert numActuators == self.numActuators
        except:
            raise Exception("Control Matrix dimension 0 axis and number of "
                            "actuators do not match.")

        #Set which modes to ignore while flattening
        if np.any(z_modes_ignore) is None:
            #By default, ignore piston, tip and tilt
            z_modes_ignore = np.asarray(range(self.numActuators) > 2)
        else:
            pass

        best_flat_actuators = np.zeros(numActuators) + 0.5
        self.send(best_flat_actuators)

        #Get a measure of the RMS phase error of the uncorrected wavefront
        #The corrected wavefront should be better than this
        interferogram = self.acquire()
        interferogram_unwrap = self.phaseunwrap(interferogram)

        x, y = interferogram_unwrap.shape
        assert x == y

        true_flat = np.zeros(np.shape(interferogram_unwrap))
        best_rms_error = np.sqrt(np.mean((true_flat - interferogram_unwrap)**2))

        for ii in range(iterations):
            interferogram = self.acquire()
            interferogram_unwrap = self.phaseunwrap(interferogram)

            edge_mask = np.sqrt(
                (np.arange(-x / 2.0, x / 2.0) ** 2).reshape((x, 1)) + (np.arange(-x / 2.0, x / 2.0) ** 2)) < (
                                    (x / 2.0) - 3)
            diff_image = abs(np.diff(np.diff(interferogram_unwrap, axis=1), axis=0)) * edge_mask[:-1, :-1]
            no_discontinuities = np.shape(np.where(diff_image > 2 * np.pi))[1]

            z_amps = self.getzernikemodes(interferogram_unwrap, nzernike)

            #We ignore piston, tip and tilt
            z_amps = z_amps * z_modes_ignore
            flat_actuators = self.set_phase((-1.0 * z_amps), offset=best_flat_actuators)

            rms_error = np.sqrt(np.mean((true_flat - interferogram_unwrap)**2))
            if rms_error < best_rms_error:
                if no_discontinuities > (x * y) / 1000.0:
                    self._logger.info("Too many discontinuities in wavefront unwrap")
                else:
                    best_flat_actuators = np.copy(flat_actuators)
                    best_rms_error = np.copy(rms_error)
            elif rms_error > best_rms_error:
                self._logger.info("RMS wavefront error worse than before")
            else:
                self._logger.info("No improvement in RMS wavefront error")
                best_flat_actuators[:] = np.copy(flat_actuators)

        self.send(best_flat_actuators)
        return best_flat_actuators

    @Pyro4.expose
    def set_phase(self, applied_z_modes, offset = None):
        actuator_pos = aoAlg.ac_pos_from_zernike(applied_z_modes, self.numActuators)
        if np.any(offset) is None:
            actuator_pos += 0.5
        else:
            actuator_pos += offset
        self.send(actuator_pos)
        return actuator_pos

    @Pyro4.expose
    def assess_character(self, modes_tba = None):
        #Ensure a Fourier filter has been constructed
        if np.any(self.fft_filter) is None:
            try:
                test_image = self.acquire()
                self.fft_filter = self.set_fourierfilter(test_image)
            except:
                raise
        else:
            pass

        if modes_tba is None:
            modes_tba = self.numActuators
        assay = np.zeros((modes_tba,modes_tba))
        applied_z_modes = np.zeros(modes_tba)
        for ii in range(modes_tba):
            self.reset()
            z_modes_ac0 = self.measure_zernike(modes_tba)
            applied_z_modes[ii] = 1
            self.set_phase(applied_z_modes)
            self._logger.info("Appling Zernike mode %i/%i" %(ii,modes_tba))
            acquired_z_modes = self.measure_zernike(modes_tba)
            self._logger.info("Measured phase")
            assay[:,ii] = acquired_z_modes - z_modes_ac0
            applied_z_modes[ii] = 0.0
        self.reset()
        return assay

    @Pyro4.expose
    def correct_sensorless_single_mode(self, image_stack, zernike_applied, nollIndex, offset = None):
        z_amps = np.zeros(self.numActuators)
        amp_to_correct = aoAlg.find_zernike_amp_sensorless(image_stack,zernike_applied)
        if abs(amp_to_correct) <= 2.5*np.max(abs(zernike_applied)):
           self._logger.info("Amplitude calculated = %f" % amp_to_correct)
        else:
            self._logger.info("Amplitude calculated = %f" % amp_to_correct)
            self._logger.info("Amplitude magnitude too large. Defaulting to 0.")
            amp_to_correct = 0
        z_amps[nollIndex-1] = -1.0*amp_to_correct
        if np.any(offset) == None:
            ac_pos_correcting = self.set_phase(z_amps)
        else:
            ac_pos_correcting = self.set_phase(z_amps, offset=offset)
        return amp_to_correct, ac_pos_correcting

    @Pyro4.expose
    def correct_sensorless_all_modes(self, full_image_stack, full_zernike_applied, nollZernike, wavelength, NA, pixel_size, offset=None):
        #May change this function later if we hand control of other cameras to the composite device
        coef = aoAlg.get_zernike_modes_sensorless(full_image_stack, full_zernike_applied, nollZernike=nollZernike,
                                                  wavelength=wavelength, NA=NA, pixel_size=pixel_size)
        if np.any(offset) == None:
            ac_pos_correcting = self.set_phase(coef)
        else:
            ac_pos_correcting = self.set_phase(coef, offset=offset)
        return coef, ac_pos_correcting
