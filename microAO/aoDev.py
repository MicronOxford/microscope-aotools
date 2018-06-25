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
import aotools
import scipy.stats as stats
from skimage.restoration import unwrap_phase
from scipy.integrate import trapz
import Pyro4
import time
from microscope.devices import Device

class AdaptiveOpticsDevice(Device):
    """Class for the adaptive optics device

    This class requires a mirror and a camera. Everything else is generated
    on or after __init__"""

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
        self.numActuators = self.mirror.get_n_actuators()
        # Region of interest (i.e. pupil offset and radius) on camera.
        self.roi = None
        #Mask for the interferometric data
        self.mask = None
        #Mask to select phase information
        self.fft_filter = None
        #Control Matrix
        self.controlMatrix = None

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
        self.pupil_ac = np.zeros(self.numActuators)

        try:
            assert np.shape(self.pupil_ac) == self.numActuators
        except:
            raise Exception("Length mismatch between pupil mask (%i) and "
                            "number of actuators (%i). Please provide a mask "
                            "of the correct length" %(np.shape(self.pupil_ac),
                                                      self.numActuators))


    def _on_shutdown(self):
        pass

    def initialize(self, *args, **kwargs):
        pass

    @Pyro4.expose
    def get_n_actuators(self):
        return self.numActuators

    @Pyro4.expose
    def send(self, values):
        self.mirror.send(values)

    @Pyro4.expose
    def send_patterns(self, patterns):
        self.mirror.send_patterns(patterns)

    @Pyro4.expose
    def set_roi(self, y0, x0, radius):
        self.roi = (y0, x0, radius)
        try:
            assert self.roi is not None
        except:
            raise Exception("ROI assignment failed")

        #Mask will need to be reconstructed as radius has changed
        self.mask = self.makemask(radius)
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
    def makemask(self, radius):
        diameter = radius * 2
        self.mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius
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
                self.mask = self.makemask(self.roi[2])
                data = data_cropped 
        else:
            data = data_raw
        return data

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

    @Pyro4.expose
    def set_fourierfilter(self, test_image, region=None):
        #Ensure an ROI is defined so a masked image is obtained
        try:

            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        #Convert image to array and float
        data = np.asarray(test_image)

        if region is None:
            region = int(data.shape[0]/8.0)

        #Apply tukey window
        fringes = np.fft.fftshift(data)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1)*tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        #Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        #Remove center section to allow finding of 1st order point
        fftarray = np.fft.fftshift(fftarray)
        find_cent = [int(fftarray.shape[1]/2),int(fftarray.shape[0]/ 2)]
        fftarray[find_cent[1]-region:find_cent[1]+region,find_cent[0]-region:find_cent[0]+region]=0.00001+0j

        #Find approximate position of first order point
        test_point = np.argmax(fftarray)
        test_point= [int(test_point%fftarray.shape[1]),int(test_point/fftarray.shape[1])]

        #Find first order point
        maxpoint = np.zeros(np.shape(test_point),dtype = int)
        maxpoint[:] = test_point[:]
        window = np.zeros((50,50))

        weight_1D = gaussian(50,50)
        weight = np.outer(weight_1D,weight_1D.T)
        weight = weight*(weight>weight[24,49])

        for ii in range(10):
            window[:,:] = np.log(abs(fftarray[maxpoint[1]-25:maxpoint[1]+25,maxpoint[0]-25:maxpoint[0]+25]))
            thresh = np.max(window) - 5
            CoM = np.zeros((1,2))
            window[window < thresh] = 0
            window[:,:] = window[:,:] * weight[:,:]
            CoM[0,:] = np.round(center_of_mass(window))
            maxpoint[0] = maxpoint[0] - 25 + int(CoM[0,1])
            maxpoint[1] = maxpoint[1] - 25 + int(CoM[0,0])

        self.fft_filter = np.zeros(np.shape(fftarray))
        mask_di = min(int(data.shape[0]*(5.0/16.0)), (maxpoint[0]-maxpoint[0]%2), (maxpoint[1]-maxpoint[1]%2))
        #FWHM = int((3.0/8.0) * mask_di)
        #stdv = FWHM/np.sqrt(8 * np.log(2))
        #x = gaussian(mask_di,stdv)
        #gauss = np.outer(x,x.T)
        #fourier_mask = gauss*(gauss>(np.max(x)*np.min(x)))
        
        #mask_rad = mask_di/2
        #fourier_mask = np.sqrt((np.arange(-mask_rad,mask_rad)**2).reshape(
        #              (mask_di,1)) + (np.arange(-mask_rad,mask_rad)**2)) < mask_rad
        
        x = np.sin(np.linspace(0, np.pi, mask_di))**2
        fourier_mask = np.outer(x,x.T)

        y_min = maxpoint[1]-int(np.floor((mask_di/2.0)))
        y_max = maxpoint[1]+int(np.ceil((mask_di/2.0)))
        x_min = maxpoint[0]-int(np.floor((mask_di/2.0)))
        x_max = maxpoint[0]+int(np.ceil((mask_di/2.0)))
        
        self.fft_filter[y_min:y_max,x_min:x_max] = fourier_mask
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
            self.mask = self.makemask(int(np.round(np.shape(image)[0]/2)))

        if self.fft_filter is not None:
            pass
        else:
            self.fft_filter = self.set_fourierfilter(image)

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
        g0, g1 = self.mgcentroid(self.fft_filter) - np.round(fftarray_filt.shape[0]//2)
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
        self.out = phaseorder1unwrap * self.mask
        return self.out


    @Pyro4.expose
    def getzernikemodes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        #Resize image
        original_dim = int(np.shape(image_unwrap)[0])
        while original_dim%resize_dim is not 0:
            resize_dim -= 1
            
        if resize_dim < original_dim/resize_dim:
            resize_dim = original_dim/resize_dim
            
        image_resize = self.bin_ndarray(image_unwrap, new_shape=(resize_dim,resize_dim), operation='mean')

        #Calculate Zernike mode
        zcoeffs_dbl = []
        num_pixels = np.count_nonzero(aotools.zernike(1, resize_dim))
        for i in range(1,(noZernikeModes+1)):
            intermediate = trapz(image_resize * aotools.zernike(i, resize_dim))
            zcoeffs_dbl.append(trapz(intermediate) / (num_pixels))
        coef = np.asarray(zcoeffs_dbl)
        return coef

    @Pyro4.expose
    def createcontrolmatrix(self, imageStack, noZernikeModes, pokeSteps, pupil_ac = None):
        #Ensure an ROI is defined so a masked image is obtained
        try:
            assert self.roi is not None
        except:
            raise Exception("No region of interest selected. Please select a region of interest")

        #Ensure the filters has been constructed
        if self.mask is not None:
            pass
        else:
            self.mask = self.makemask(int(np.round(np.shape(imageStack)[1]/2)))

        if self.fft_filter is not None:
            pass
        else:
            self.fft_filter = self.set_fourierfilter(imageStack[0,:,:])

        if pupil_ac == None:
            pupil_ac = self.pupil_ac
        else:
            pass

        slopes = np.zeros(noZernikeModes)
        intercepts = np.zeros(noZernikeModes)
        r_values = np.zeros(noZernikeModes)
        p_values = np.zeros(noZernikeModes)
        std_errs = np.zeros(noZernikeModes)

        # Define variables
        try:
            assert type(imageStack) is np.ndarray
        except:
            print "Error: Expected numpy.ndarray input data type, got %s" %type(imageStack)
        [noImages, x, y] = np.shape(imageStack)
        numPokeSteps = len(pokeSteps)
        zernikeModeAmp = np.zeros((numPokeSteps,noZernikeModes))

        C_mat = np.zeros((noZernikeModes,self.numActuators))
        all_zernikeModeAmp = np.ones((noImages,noZernikeModes))
        offsets = np.zeros((noZernikeModes,self.numActuators))
        P_tests = np.zeros((noZernikeModes,self.numActuators))
        threshold = 0.005 #Threshold for pinv later

        edge_mask = np.sqrt((np.arange(-self.roi[2],self.roi[2])**2).reshape(
            (self.roi[2]*2,1)) + (np.arange(-self.roi[2],self.roi[2])**2)) < self.roi[2]-3

        # Here the each image in the image stack (read in as np.array), centre and4 diameter should be passed to the unwrap
        # function to obtain the Zernike modes for each one. For the moment a set of random Zernike modes are generated.
        for ii in range(self.numActuators):
            if pupil_ac[ii] == 1:
                pokeSteps_trimmed_list = []
                zernikeModeAmp_list = []
                #Get the amplitudes of each Zernike mode for the poke range of one actuator
                for jj in range(numPokeSteps):
                    curr_calc = (ii * numPokeSteps) + jj + 1
                    print("Calculating Zernike modes %d/%d..." %(curr_calc, noImages))
                    image_unwrap = self.phaseunwrap(imageStack[((ii * numPokeSteps) + jj),:,:])
                    diff_image = abs(np.diff(np.diff(image_unwrap,axis=1),axis=0)) * edge_mask[:-1,:-1]
                    if np.any(diff_image > 2*np.pi):
                        self._logger.info("Unwrap image %d/%d contained discontinuites" %(curr_calc, noImages))
                        self._logger.info("Zernike modes %d/%d not calculates" %(curr_calc, noImages))
                    else:
                        pokeSteps_trimmed_list.append(pokeSteps[jj])
                        self._logger.info("Calculating Zernike modes %d/%d..." %(curr_calc, noImages))
                        curr_amps = self.getzernikemodes(image_unwrap, noZernikeModes)
                        thresh_amps = curr_amps * (abs(curr_amps)>0.5)
                        zernikeModeAmp_list.append(thresh_amps)
                        all_zernikeModeAmp[(curr_calc-1),:] = thresh_amps
                        self._logger.info("Zernike modes %d/%d calculated" %(curr_calc, noImages))

                pokeSteps_trimmed = np.asarray(pokeSteps_trimmed_list)
                zernikeModeAmp = np.asarray(zernikeModeAmp_list)

                #Fit a linear regression to get the relationship between actuator position and Zernike mode amplitude
                for kk in range(noZernikeModes):
                    self._logger.info("Fitting regression %d/%d..." % (kk+1, noZernikeModes))
                    try:
                        slopes[kk],intercepts[kk],r_values[kk],p_values[kk],std_errs[kk] = \
                            stats.linregress(pokeSteps_trimmed,zernikeModeAmp[:,kk])
                    except Exception as e:
                        self._logger.info(e)
                    self._logger.info("Regression %d/%d fitted" % (kk + 1, noZernikeModes))

                #Input obtained slopes as the entries in the control matrix
                C_mat[:,ii] = slopes[:]
                offsets[:,ii] = intercepts[:]
                P_tests[:,ii] = p_values[:]
            else:
                self._logger.info("Actuator %d is not in the pupil and therefore skipped" % (ii))
        print("Computing Control Matrix")
        self.controlMatrix = np.linalg.pinv(C_mat, rcond=threshold)
        print("Control Matrix computed")
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
    def calibrate(self, numPokeSteps = 5):
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
            self.mask = self.makemask(self.roi[2])
            
        if self.fft_filter is not None:
            pass
        else:
            self._logger.info("Constructing Fourier filter")
            self.fft_filter = self.set_fourierfilter(test_image[:,:])

        nzernike = self.numActuators

        poke_min = -0.65
        poke_max = 0.65
        pokeSteps = np.linspace(poke_min,poke_max,numPokeSteps)
        noImages = numPokeSteps*(np.shape(np.where(self.pupil_ac == 1))[1])

        image_stack_cropped = np.zeros((noImages,self.roi[2]*2,self.roi[2]*2))
        unwrapped_stack = np.zeros((noImages,self.roi[2]*2,self.roi[2]*2))

        slopes = np.zeros(nzernike)
        intercepts = np.zeros(nzernike)
        r_values = np.zeros(nzernike)
        p_values = np.zeros(nzernike)
        std_errs = np.zeros(nzernike)

        actuator_values = np.zeros((noImages,nzernike))
        for ii in range(nzernike):
            for jj in range(numPokeSteps):
                actuator_values[(numPokeSteps * ii) + jj, ii] = pokeSteps[jj]

        C_mat = np.zeros((nzernike,self.numActuators))
        all_zernikeModeAmp = np.ones((noImages,nzernike))
        offsets = np.zeros((nzernike,self.numActuators))
        P_tests = np.zeros((nzernike,self.numActuators))
        threshold = 0.005 #Threshold for pinv later

        edge_mask = np.sqrt((np.arange(-self.roi[2],self.roi[2])**2).reshape(
            (self.roi[2]*2,1)) + (np.arange(-self.roi[2],self.roi[2])**2)) < self.roi[2]-3

        for ac in range(self.numActuators):
            if self.pupil_ac[ac] == 1:
                pokeSteps_trimmed_list = []
                zernikeModeAmp_list = []
                for im in range(numPokeSteps):
                    curr_calc = (ac * numPokeSteps) + im + 1
                    self._logger.info("Frame %i/%i captured" %(curr_calc, noImages))
                    try:
                        self.mirror.send(actuator_values[(curr_calc-1),:])
                    except:
                        self._logger.info("Actuator values being sent:")
                        self._logger.info(actuator_values[(curr_calc-1),:])
                        self._logger.info("Shape of actuator vector:")
                        self._logger.info(np.shape(actuator_values[(curr_calc-1),:]))
                    poke_image = self.acquire()
                    image_stack_cropped[curr_calc-1,:,:] = poke_image
                    image_unwrap = self.phaseunwrap(poke_image)
                    unwrapped_stack[curr_calc-1,:,:] = image_unwrap
                    diff_image = abs(np.diff(np.diff(image_unwrap,axis=1),axis=0)) * edge_mask[:-1,:-1]
                    if np.any(diff_image > 2*np.pi):
                        self._logger.info("Unwrap image %d/%d contained discontinuites" %(curr_calc, noImages))
                        self._logger.info("Zernike modes %d/%d not calculates" %(curr_calc, noImages))
                    else:
                        pokeSteps_trimmed_list.append(pokeSteps[im])
                        self._logger.info("Calculating Zernike modes %d/%d..." %(curr_calc, noImages))
                        curr_amps = self.getzernikemodes(image_unwrap, nzernike)
                        thresh_amps = curr_amps * (abs(curr_amps)>0.5)
                        zernikeModeAmp_list.append(thresh_amps)
                        all_zernikeModeAmp[(curr_calc-1),:] = thresh_amps
                        self._logger.info("Zernike modes %d/%d calculated" %(curr_calc, noImages))

                pokeSteps_trimmed = np.asarray(pokeSteps_trimmed_list)
                zernikeModeAmp = np.asarray(zernikeModeAmp_list)

                #Fit a linear regression to get the relationship between actuator position and Zernike mode amplitude
                for kk in range(nzernike):
                    self._logger.info("Fitting regression %d/%d..." % (kk+1, nzernike))
                    try:
                        slopes[kk],intercepts[kk],r_values[kk],p_values[kk],std_errs[kk] = \
                            stats.linregress(pokeSteps_trimmed,zernikeModeAmp[:,kk])
                    except Exception as e:
                        self._logger.info(e)
                    self._logger.info("Regression %d/%d fitted" % (kk + 1, nzernike))

                #Input obtained slopes as the entries in the control matrix
                C_mat[:,ac] = slopes[:]
                offsets[:,ac] = intercepts[:]
                P_tests[:,ac] = p_values[:]
            else:
                self._logger.info("Actuator %d is not in the pupil and therefore skipped" % (ac))

        self._logger.info("Computing Control Matrix")
        self.controlMatrix = np.linalg.pinv(C_mat, rcond=threshold)
        self._logger.info("Control Matrix computed")
        np.save("image_stack_cropped",image_stack_cropped)
        np.save("unwrapped_stack",unwrapped_stack)
        np.save("C_mat", C_mat)

        return self.controlMatrix

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
        self.mirror.send(flat_actuators)
        previous_flat_actuators = np.zeros(numActuators)

        z_amps = np.zeros(nzernike)
        previous_z_amps = np.zeros(nzernike)

        previous_rms_error = np.inf

        for ii in range(iterations):
            interferogram = self.acquire()

            interferogram_unwrap = self.phaseunwrap(interferogram)
            z_amps[:] = self.getzernikemodes(interferogram_unwrap, nzernike)
            flat_actuators[:] = previous_flat_actuators - 1.0 * np.dot(self.controlMatrix, z_amps)

            flat_actuators[np.where(flat_actuators > 1)] = 1
            flat_actuators[np.where(flat_actuators < -1)] = -1

            self.mirror.send(flat_actuators)

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
        if int(np.shape(applied_z_modes)[0]) < int(np.shape(self.controlMatrix)[1]):
            pad_length = int(np.shape(applied_z_modes)[0]) - int(np.shape(self.controlMatrix)[1])
            np.pad(applied_z_modes, (0,pad_length), 'constant')
        elif int(np.shape(applied_z_modes)[0]) > int(np.shape(self.controlMatrix)[1]):
            applied_z_modes = applied_z_modes[:int(np.shape(self.controlMatrix)[1])]
        else:
            pass

        actuator_pos = np.zeros(self.numActuators)
        if offset is not None:
            actuator_pos[:] = np.dot(self.controlMatrix, applied_z_modes) + offset
        else:
            actuator_pos[:] = np.dot(self.controlMatrix, applied_z_modes)

        self.mirror.send(actuator_pos)
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

        flat_values = self.flatten_phase(iterations=10)

        if modes_tba is None:
            modes_tba = self.numActuators
        assay = np.zeros((modes_tba,modes_tba))
        applied_z_modes = np.zeros(modes_tba)
        for ii in range(3,modes_tba):
            applied_z_modes[ii] = 1
            self.set_phase(applied_z_modes, offset=flat_values)
            self._logger.info("Appling Zernike mode %i/%i" %(ii,modes_tba))
            acquired_z_modes = self.measure_zernike(modes_tba)
            self._logger.info("Measured phase")
            assay[:,ii] = acquired_z_modes
            applied_z_modes[ii] = 0
        self.mirror.reset()
        return assay
