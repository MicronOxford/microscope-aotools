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
import numpy.ma as ma
from scipy.ndimage.measurements import center_of_mass
from scipy.signal import tukey, gaussian
import aotools
import scipy.stats as stats
from skimage.restoration import unwrap_phase
from scipy.integrate import trapz

from microscope.devices import Device
from microscope.clients import Client, DataClient

class AdaptiveOpticsDevice(Device):
    """Class for the adaptive optics device

    This class requires a mirror and a camera. Everything else is generated
    on or after __init__"""

    def __init__(self, camera_uri, mirror_uri):
        # Init will fail if devices it depends on aren't already running, but
        # deviceserver should retry automatically.
        # Camera or wavefront sensor. Must support soft_trigger for now.
        self.camera = DataClient(camera_uri)
        # Deformable mirror device.
        self.mirror = Client(mirror_uri)
        # Region of interest (i.e. pupil offset and radius) on camera.
        self.roi = None
        self.mask = None

    def set_roi(self, x0, y0, r):
        self.roi = (x0, y0, r)
        self.mask = self.makemask(r)

    def makemask(self, diameter):
        radius = diameter/2
        mask = np.sqrt((np.arange(-radius,radius)**2).reshape((diameter,1)) + (np.arange(-radius,radius)**2)) < radius
        return mask

    def acquire(self):
        data_raw = self.camera.trigger_and_wait()
        if self.roi is not None:
            data_cropped = np.zeros((self.roi[2],self.roi[2]), dtype=float)
            data_cropped = data_raw[self.roi[0]-int(np.floor(self.roi[2]/2.0)):
            self.roi[0]+int(np.ceil(self.roi[2]/2.0)),
            self.roi[1]-int(np.floor(self.roi[2]/2.0)):
            self.roi[1]+int(np.ceil(self.roi[2]/2.0))]
            data = data_cropped * self.mask
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

    def getfourierfilter(self, image, mask, middle, diameter, region=30):
        #Convert image to array and float
        data = np.asarray(image)
        data = data[::-1]
        data = data.astype(float)

        #Mask image to remove extraneous data from edges
        data_cropped = np.zeros((diameter,diameter), dtype=float)
        data_cropped = data[middle[0]-int(np.floor(diameter/2.0)):middle[0]+int(np.ceil(diameter/2.0)),middle[1]-int(
            np.floor(diameter/2.0)):middle[1]+int(np.ceil(diameter/2.0))]
        data_cropped = data_cropped * mask

        #Apply tukey window
        fringes = np.fft.fftshift(data_cropped)
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

        fft_filter = np.zeros(np.shape(fftarray))
        gauss_dim = min(int(image.shape[0]*(5.0/16.0)), (maxpoint[0]-maxpoint[0]%2), (maxpoint[1]-maxpoint[1]%2))
        FWHM = int((3.0/8.0) * gauss_dim)
        stdv = FWHM/np.sqrt(8 * np.log(2))
        x = gaussian(gauss_dim,stdv)
        gauss = np.outer(x,x.T)
        gauss = gauss*(gauss>(np.max(x)*np.min(x)))

        fft_filter[(maxpoint[1]-(gauss_dim/2)):(maxpoint[1]+(gauss_dim/2)),(maxpoint[0]-(gauss_dim/2)):(maxpoint[0]+(gauss_dim/2))] = gauss
        return fft_filter

    def phaseunwrap(self, image, mask, fft_filter, middle, diameter):
        #Convert image to array and float
        data = np.asarray(image)
        data = data[::-1]
        data = data.astype(float)

        #Mask image to remove extraneous data from edges
        data_cropped = np.zeros((diameter,diameter), dtype=float)
        data_cropped = data[middle[0]-int(np.floor(diameter/2.0)):middle[0]+int(np.ceil(diameter/2.0)),middle[1]-int(
            np.floor(diameter/2.0)):middle[1]+int(np.ceil(diameter/2.0))]
        data_cropped = data_cropped * mask

        #Apply tukey window
        fringes = np.fft.fftshift(data_cropped)
        tukey_window = tukey(fringes.shape[0], .10, True)
        tukey_window = np.fft.fftshift(tukey_window.reshape(1, -1)*tukey_window.reshape(-1, 1))
        fringes_tukey = fringes * tukey_window

        #Perform fourier transform
        fftarray = np.fft.fft2(fringes_tukey)

        #Apply Fourier filter
        M = np.fft.fftshift(fft_filter)
        fftarray_filt = fftarray * M
        fftarray_filt = np.fft.fftshift(fftarray_filt)

        #Roll data to the centre
        g0, g1 = self.mgcentroid(fft_filter) - np.round(fftarray_filt.shape[0]//2)
        fftarray_filt = np.roll(fftarray_filt, -g0, axis=1)
        fftarray_filt = np.roll(fftarray_filt, -g1, axis=0)

        #Convert to real space
        fftarray_filt_shift = np.fft.fftshift(fftarray_filt)
        complex_phase = np.fft.fftshift(np.fft.ifft2(fftarray_filt_shift))

        #Find phase data by taking 2d arctan of imaginary and real parts
        phaseorder1 = np.zeros(complex_phase.shape)
        phaseorder1 = np.arctan2(complex_phase.imag,complex_phase.real)

        #Mask out edge region to allow unwrap to only use correct region
        phaseorder1mask = ma.masked_where(mask == 0,phaseorder1)

        #Perform unwrap
        phaseorder1unwrap = unwrap_phase(phaseorder1mask)
        out = np.ma.filled(phaseorder1unwrap, 0)
        return out


    def getzernikemodes(self, image_unwrap, noZernikeModes, resize_dim = 128):
        #Resize image
        original_dim = int(np.shape(image_unwrap)[0])
        while original_dim%resize_dim is not 0:
            resize_dim -= 1
        image_resize = self.bin_ndarray(image_unwrap, new_shape=(resize_dim,resize_dim), operation='mean')

        #Calculate Zernike mode
        zcoeffs_dbl = []
        num_pixels = np.count_nonzero(aotools.zernike(1, resize_dim))
        for i in range(1,(noZernikeModes+1)):
            intermediate = trapz(image_resize * aotools.zernike(i, resize_dim))
            zcoeffs_dbl.append(trapz(intermediate) / (num_pixels))
        coef = np.asarray(zcoeffs_dbl)
        return coef

    def createcontrolmatrix(self, imageStack, numActuators, noZernikeModes, centre, diameter):

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
        image_unwrap = np.shape((x,y))
        numPokeSteps = noImages/numActuators
        pokeSteps = np.linspace(-0.6,0.6,numPokeSteps)
        zernikeModeAmp = np.zeros((numPokeSteps,noZernikeModes))
        C_mat = np.zeros((noZernikeModes,numActuators))
        all_zernikeModeAmp = np.ones((noImages,noZernikeModes))
        offsets = np.zeros((noZernikeModes,numActuators))
        P_tests = np.zeros((noZernikeModes,numActuators))

        mask = self.makemask(diameter)
        fft_filter = self.getfourierfilter(imageStack[0,:,:], mask, middle=centre, diameter=diameter)

        # Here the each image in the image stack (read in as np.array), centre and4 diameter should be passed to the unwrap
        # function to obtain the Zernike modes for each one. For the moment a set of random Zernike modes are generated.
        for ii in range(numActuators):

            #Get the amplitudes of each Zernike mode for the poke range of one actuator
            for jj in range(numPokeSteps):
                curr_calc = (ii * numPokeSteps) + jj + 1
                print("Calculating Zernike modes %d/%d..." %(curr_calc, noImages))
                image_unwrap = self.phaseunwrap(imageStack[((ii * numPokeSteps) + jj),:,:], mask, fft_filter, middle=centre, diameter=diameter)
                zernikeModeAmp[jj,:] = self.getzernikemodes(image_unwrap, noZernikeModes)
                all_zernikeModeAmp[((ii * numPokeSteps) + jj),:] = zernikeModeAmp[jj,:]
                print("Zernike modes %d/%d calculated" %(curr_calc, noImages))

            #Fit a linear regression to get the relationship between actuator position and Zernike mode amplitude
            for kk in range(noZernikeModes):
                print("Fitting regression %d/%d..." % (kk+1, noZernikeModes))
                slopes[kk], intercepts[kk], r_values[kk], p_values[kk], std_errs[kk] = stats.linregress(pokeSteps,zernikeModeAmp[:,kk])
                print("Regression %d/%d fitted" % (kk + 1, noZernikeModes))


            #Input obtained slopes as the entries in the control matrix
            C_mat[:,ii] = slopes[:]
            offsets[:,ii] = intercepts[:]
            P_tests[:,ii] = p_values[:]
        print("Computing Control Matrix")
        controlMatrix = np.linalg.pinv(C_mat)
        print("Control Matrix computed")
        return controlMatrix

    def calibrate(self, acquire, mirror, camera, centre, diameter, numPokeSteps = 10):
        numActuators = mirror.n_actuators()
        nzernike = numActuators

        pokeSteps = np.linspace(0.5,0.95,numPokeSteps)
        noImages = numPokeSteps*nzernike

        actuator_values = np.zeros((noImages,nzernike))
        for ii in range(nzernike):
            for jj in range(numPokeSteps):
                actuator_values[(numPokeSteps * ii) + jj, ii] = pokeSteps[jj]

        (width, height) = camera.get_sensor_shape()
        imStack = np.zeros(noImages, height, width)
        for im in range(noImages):
            imStack[im, :, :] = acquire(actuator_values[im])

        controlMatrix, flat_values = self.createcontrolmatrix(imStack, numActuators, nzernike, centre, diameter)

        return controlMatrix, flat_values

    def flatten_phase(self, acquire, mirror, controlMatrix, centre, diameter, iterations = 1):
        numActuators, nzernike = np.shape(controlMatrix)

        mask = self.makemask(diameter)
        interferogram = acquire()
        fft_filter = self.getfourierfilter(interferogram, mask, centre, diameter)

        flat_actuators = np.zeros(numActuators)
        previous_flat_actuators = np.zeros(numActuators)
        z_amps = np.zeros(nzernike)
        previous_z_amps = np.zeros(nzernike)

        for ii in range(iterations):
            interferogram = acquire()

            z_amps[:] = self.getzernikemodes(interferogram, mask, fft_filter, nzernike, centre, diameter)
            flat_actuators[:] = -1.0 * np.dot(controlMatrix, z_amps)

            mirror.apply_pattern(flat_actuators)

            ##We need some test here for ringing in our solution

            previous_z_amps[:] = z_amps[:]
            previous_flat_actuators[:] = flat_actuators[:]

        return flat_actuators
