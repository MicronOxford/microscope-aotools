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

    def __init__(self, camera_uri, mirror_uri, **kwargs):
        # Init will fail if devices it depends on aren't already running, but
        # deviceserver should retry automatically.
        super(AdaptiveOpticsDevice, self).__init__(**kwargs)
        # Camera or wavefront sensor. Must support soft_trigger for now.
        self.camera = Pyro4.Proxy('PYRO:%s@%s:%d' %(camera_uri[0].__name__,
                                                camera_uri[1], camera_uri[2]))
        self.camera.enable()
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

        ##We don't use all the actuators. Create a mask for the actuators outside
        ##the pupil so we can selectively calibrate them. 0 denotes actuators at
        ##the edge, i.e. outside the pupil, and 1 denotes actuators in the pupil

        #Use this if all actuators are being used
        #self.pupil_ac = np.ones(self.numActuators)

        #Preliminary mask for DeepSIM
        #self.pupil_ac = np.asarray([0,0,0,0,0,
        #                            0,0,1,1,1,0,0,
        #                            0,0,1,1,1,1,1,0,0,
        #                            0,1,1,1,1,1,1,1,0,
        #                            0,1,1,1,1,1,1,1,0,
        #                            0,1,1,1,1,1,1,1,0,
        #                            0,0,1,1,1,1,1,0,0,
        #                            0,0,1,1,1,0,0,
        #                            0,0,0,0,0])
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
        self._logger.info("Sending patterns to DM")
        self.mirror.apply_pattern(values)

    @Pyro4.expose
    def queue_patterns(self, patterns):
        self._logger.info("Queuing patterns on DM")
        self.mirror.queue_patterns(patterns)

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
        if self.roi is not None:
            return self.roi
        else:
            raise Exception("No region of interest selected. Please select a region of interest")

    @Pyro4.expose
    def get_fourierfilter(self):
        if self.fft_filter is not None:
            return self.fft_filter
        else:
            raise Exception("No Fourier filter created. Please create one.")

    @Pyro4.expose
    def set_controlMatrix(self,controlMatrix):
        self.controlMatrix = controlMatrix
        return

    @Pyro4.expose
    def get_roi(self):
        if self.roi is not None:
            return self.roi
        else:
            raise Exception("No control matrix calculated. Please calibrate the mirror")

    @Pyro4.expose
    def get_controlMatrix(self):
        if self.controlMatrix is not None:
            return self.controlMatrix
        else:
            raise Exception("No control matrix calculated. Please calibrate the mirror")

    @Pyro4.expose
    def make_mask(self, radius):
        self.mask = aoAlg.make_mask(radius)
        return self.mask

    @Pyro4.expose
    def acquire_raw(self):
        self.acquiring = True
        while self.acquiring == True:
            try:
                self.camera.soft_trigger()
                data_raw = self.camera.get_current_image()
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
                self.camera.soft_trigger()
                data_raw = self.camera.get_current_image()
                self.acquiring = False
            except Exception as e:
                if str(e) == str("ERROR 10: Timeout"):
                    self._logger.info("Recieved Timeout error from camera. Waiting to try again...")
                    time.sleep(1)
                else:
                    self._logger.info(type(e))
                    self._logger.info("Error is: %s" %(e))
                    raise e
        if self.roi is not None:
            #self._logger.info('roi is not None')
            data_cropped = np.zeros((self.roi[2]*2,self.roi[2]*2), dtype=float)
            data_cropped[:,:] = data_raw[self.roi[0]-self.roi[2]:self.roi[0]+self.roi[2],
                            self.roi[1]-self.roi[2]:self.roi[1]+self.roi[2]]
            if self.mask is not None:
                data = data_cropped * self.mask
            else:
                self.mask = self.make_mask(self.roi[2])
                data = data_cropped 
        else:
            data = data_raw
        return data

    @Pyro4.expose
    def set_fourierfilter(self, test_image, region=None):
        #Ensure an ROI is defined so a masked image is obtained
        try:

            assert self.roi is not None
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
            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        if image is None:
            image = self.acquire()

        #Ensure the filters has been constructed
        if self.mask is not None:
            pass
        else:
            self.mask = self.make_mask(int(np.round(np.shape(image)[0]/2)))

        if self.fft_filter is not None:
            pass
        else:
            self.fft_filter = self.set_fourierfilter(image)

        self.out = aoAlg.phase_unwrap(image)
        return self.out


    @Pyro4.expose
    def getzernikemodes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        coef = aoAlg.get_zernike_modes(image_unwrap, noZernikeModes, resize_dim = resize_dim)
        return coef

    @Pyro4.expose
    def createcontrolmatrix(self, imageStack, noZernikeModes, pokeSteps, pupil_ac = None, threshold = 0.005):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        #Ensure the filters has been constructed
        if self.mask is not None:
            pass
        else:
            self.mask = self.make_mask(int(np.round(np.shape(imageStack)[1]/2)))

        if self.fft_filter is not None:
            pass
        else:
            self.fft_filter = self.set_fourierfilter(imageStack[0,:,:])

        if pupil_ac == None:
            pupil_ac = self.pupil_ac
        else:
            pass

        self.controlMatrix = aoAlg.create_control_matrix(self, imageStack,
                                    self.numActuators, noZernikeModes,
                                    pokeSteps, pupil_ac = pupil_ac, threshold = threshold)
        return self.controlMatrix

    @Pyro4.expose
    def acquire_unwrapped_phase(self):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        #Ensure a Fourier filter has been constructed
        if self.fft_filter is not None:
            pass
        else:
            try:
                test_image = self.acquire()
                self.fft_filter = self.set_fourierfilter(test_image)
            except:
                raise
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
    def calibrate(self, numPokeSteps = 5, threshold = 0.005):
        self.camera.set_exposure_time(0.05)
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        test_image = np.asarray(self.acquire())
        (width, height) = np.shape(test_image)
        
        #Ensure the filters has been constructed
        if self.mask is not None:
            pass
        else:
            self._logger.info("Constructing mask")
            self.mask = self.make_mask(self.roi[2])
            
        if self.fft_filter is not None:
            pass
        else:
            self._logger.info("Constructing Fourier filter")
            self.fft_filter = self.set_fourierfilter(test_image[:,:])

        nzernike = self.numActuators

        poke_min = 0.175
        poke_max = 0.825
        pokeSteps = np.linspace(poke_min,poke_max,numPokeSteps)
        noImages = numPokeSteps*(np.shape(np.where(self.pupil_ac == 1))[1])

        image_stack_cropped = np.zeros((noImages,self.roi[2]*2,self.roi[2]*2))

        actuator_values = np.zeros((noImages,nzernike))
        for ii in range(nzernike):
            for jj in range(numPokeSteps):
                actuator_values[(numPokeSteps * ii) + jj, ii] = pokeSteps[jj]

        for ac in range(self.numActuators):
            for im in range(numPokeSteps):
                curr_calc = (ac * numPokeSteps) + im + 1
                self._logger.info("Frame %i/%i captured" %(curr_calc, noImages))
                try:
                    self.mirror.apply_pattern(actuator_values[(curr_calc-1),:])
                except:
                    self._logger.info("Actuator values being sent:")
                    self._logger.info(actuator_values[(curr_calc-1),:])
                    self._logger.info("Shape of actuator vector:")
                    self._logger.info(np.shape(actuator_values[(curr_calc-1),:]))
                poke_image = self.acquire()
                image_stack_cropped[curr_calc-1,:,:] = poke_image

        self._logger.info("Computing Control Matrix")
        self.controlMatrix = aoAlg.create_control_matrix(imageStack = image_stack_cropped,
                                                         numActuators = self.numActuators,
                                                         noZernikeModes = 69,
                                                         pokeSteps = pokeSteps,
                                                         pupil_ac = self.pupil_ac,
                                                         threshold = threshold)
        self._logger.info("Control Matrix computed")
        np.save("image_stack_cropped",image_stack_cropped)
        np.save("control_matrix", self.controlMatrix)

        self.flat_actuators_sys = self.flatten_phase(iterations=25)

        return self.controlMatrix, self.flat_actuators_sys

    @Pyro4.expose
    def flatten_phase(self, iterations = 1):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        #Ensure a Fourier filter has been constructed
        if self.fft_filter is not None:
            pass
        else:
            test_image = self.acquire()
            self.fft_filter = self.set_fourierfilter(test_image)

        numActuators, nzernike = np.shape(self.controlMatrix)
        try:
            assert numActuators == self.numActuators
        except:
            raise Exception("Control Matrix dimension 0 axis and number of "
                            "actuators do not match.")

        flat_actuators = np.zeros(numActuators)
        self.mirror.apply_pattern(flat_actuators)
        previous_flat_actuators = flat_actuators

        z_amps = np.zeros(nzernike)
        previous_z_amps = np.zeros(nzernike)

        previous_rms_error = np.inf

        for ii in range(iterations):
            interferogram = self.acquire()

            interferogram_unwrap = self.phaseunwrap(interferogram)
            z_amps[:] = self.getzernikemodes(interferogram_unwrap, nzernike)
            flat_actuators[:] = -1.0 * aoAlg.ac_pos_from_zernike(z_amps, self.numActuators,
                                                        offset = (-1.0 * previous_flat_actuators))

            flat_actuators[np.where(flat_actuators > 1)] = 1
            flat_actuators[np.where(flat_actuators < 0)] = 0

            self.mirror.apply_pattern(flat_actuators)

            true_flat = np.zeros(np.shape(interferogram_unwrap))
            rms_error = np.sqrt(np.mean((true_flat - interferogram_unwrap)**2))
            if rms_error < previous_rms_error:
                previous_z_amps[:] = z_amps[:]
                previous_flat_actuators[:] = flat_actuators[:]
            else:
                print("Ringing occured after %f iterations") %(ii + 1)

            #try:
            #    assert np.all(abs(flat_actuators)<1)
            #except:
            #    raise Exception("All actuators at max stroke length")




        return flat_actuators

    @Pyro4.expose
    def set_phase(self, applied_z_modes, offset = None):
        actuator_pos = aoAlg.ac_pos_from_zernike(applied_z_modes,
                                    self.numActuators, offset = offset)
        self.mirror.apply_pattern(actuator_pos)
        return

    @Pyro4.expose
    def assess_character(self, modes_tba = None):
        #Ensure a Fourier filter has been constructed
        if self.fft_filter is not None:
            pass
        else:
            try:
                test_image = self.acquire()
                self.fft_filter = self.set_fourierfilter(test_image)
            except:
                raise

        flat_values = self.flatten_phase(iterations=5)

        if modes_tba is None:
            modes_tba = self.numActuators
        assay = np.zeros((modes_tba,modes_tba))
        applied_z_modes = np.zeros(modes_tba)
        for ii in range(modes_tba):
            applied_z_modes[ii] = 1
            self.set_phase(applied_z_modes, offset=flat_values)
            self._logger.info("Appling Zernike mode %i/%i" %(ii,modes_tba))
            acquired_z_modes = self.measure_zernike(modes_tba)
            self._logger.info("Measured phase")
            assay[:,ii] = acquired_z_modes
            applied_z_modes[ii] = 0
        self.mirror.reset()
        return assay
