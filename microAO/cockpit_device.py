#!/usr/bin/python
# -*- coding: utf-8

## Copyright (C) 2021 David Miguel Susano Pinto <david.pinto@bioch.ox.ac.uk>
## Copyright (C) 2019 Ian Dobbie <ian.dobbie@bioch.ox.ac.uk>
## Copyright (C) 2019 Mick Phillips <mick.phillips@gmail.com>
## Copyright (C) 2019 Nick Hall <nicholas.hall@dtc.ox.ac.uk>
##
## This file is part of Cockpit.
##
## Cockpit is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## Cockpit is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Cockpit.  If not, see <http://www.gnu.org/licenses/>.

"""Cockpit Device file for a Composite AO device as constructed by Microscope-AOtools.

This file provides the cockpit end of the driver for a deformable
mirror as currently mounted on DeepSIM in Oxford.

"""

import os
import time
import typing
from collections import OrderedDict

import aotools
import cockpit.devices
import cockpit.gui.device
import cockpit.handlers.executor
import cockpit.interfaces.imager
import cockpit.interfaces.stageMover
import cockpit.util
import numpy as np
import Pyro4
import wx
from cockpit import depot, events
from cockpit.devices import device
from cockpit.util import userConfig
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from wx.lib.floatcanvas.FloatCanvas import FloatCanvas


_ROI_MIN_RADIUS = 8


def _np_grey_img_to_wx_image(np_img: np.ndarray) -> wx.Image:
    img_min = np.min(np_img)
    img_max = np.max(np_img)
    scaled_img = (np_img - img_min) / (img_max - img_min)

    uint8_img = (scaled_img * 255).astype("uint8")
    scaled_img_rgb = np.require(
        np.stack((uint8_img,) * 3, axis=-1), requirements="C"
    )

    wx_img = wx.Image(
        scaled_img_rgb.shape[0], scaled_img_rgb.shape[1], scaled_img_rgb,
    )
    return wx_img


class _ROISelect(wx.Frame):
    """Display a window that allows the user to select a circular area.

    This is a window for selecting the ROI for interferometry.
    """

    def __init__(self, input_image: np.ndarray, scale_factor=1) -> None:
        super().__init__(None, title="ROI selector")
        self._panel = wx.Panel(self)
        self._img = _np_grey_img_to_wx_image(input_image)
        self._scale_factor = scale_factor

        # What, if anything, is being dragged.
        # XXX: When we require Python 3.8, annotate better with
        # `typing.Literal[None, "xy", "r"]`
        self._dragging: typing.Optional[str] = None

        # Canvas
        self.canvas = FloatCanvas(self._panel, size=self._img.GetSize())
        self.canvas.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)
        self.bitmap = self.canvas.AddBitmap(self._img, (0, 0), Position="cc")
        self.circle = self.canvas.AddCircle(
            (0, 0), 128, LineColor="cyan", LineWidth=2
        )

        # Save button
        saveBtn = wx.Button(self._panel, label="Save ROI")
        saveBtn.Bind(wx.EVT_BUTTON, self.OnSave)

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self.canvas)
        panel_sizer.Add(saveBtn, wx.SizerFlags().Border())
        self._panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._panel)
        self.SetSizerAndFit(frame_sizer)

    @property
    def ROI(self):
        """Convert circle parameters to ROI x, y and radius"""
        roi_x, roi_y = self.canvas.WorldToPixel(self.circle.XY)
        roi_r = max(self.circle.WH)
        return (roi_x, roi_y, roi_r)

    def OnSave(self, event: wx.CommandEvent) -> None:
        roi = [x * self._scale_factor for x in self.ROI]
        userConfig.setValue("dm_circleParams", (roi[1], roi[0], roi[2]))
        print("Save ROI button pressed. Current ROI: (%i, %i, %i)" % self.ROI)

    def MoveCircle(self, pos: wx.Point, r) -> None:
        """Set position and radius of circle with bounds checks."""
        x, y = pos
        _x, _y, _r = self.ROI
        xmax, ymax = self._img.GetSize()
        if r == _r:
            x_bounded = min(max(r, x), xmax - r)
            y_bounded = min(max(r, y), ymax - r)
            r_bounded = r
        else:
            r_bounded = max(_ROI_MIN_RADIUS, min(xmax - x, x, ymax - y, y, r))
            x_bounded = min(max(r_bounded, x), xmax - r_bounded)
            y_bounded = min(max(r_bounded, y), ymax - r_bounded)
        self.circle.SetPoint(self.canvas.PixelToWorld((x_bounded, y_bounded)))
        self.circle.SetDiameter(2 * r_bounded)
        if any((x_bounded != x, y_bounded != y, r_bounded != r)):
            self.circle.SetColor("magenta")
        else:
            self.circle.SetColor("cyan")

    def OnMouse(self, event: wx.MouseEvent) -> None:
        pos = event.GetPosition()
        x, y, r = self.ROI
        if event.LeftDClick():
            # Set circle centre
            self.MoveCircle(pos, r)
        elif event.Dragging():
            # Drag circle centre or radius
            drag_r = np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            if self._dragging is None:
                # determine what to drag
                if drag_r < 0.5 * r:
                    # closer to center
                    self._dragging = "xy"
                else:
                    # closer to edge
                    self._dragging = "r"
            elif self._dragging == "r":
                # Drag circle radius
                self.MoveCircle((x, y), drag_r)
            elif self._dragging == "xy":
                # Drag circle centre
                self.MoveCircle(pos, r)

        if not event.Dragging():
            # Stop dragging
            self._dragging = None
            self.circle.SetColor("cyan")

        self.canvas.Draw(Force=True)


class _PhaseViewer(wx.Frame):
    """This is a window for selecting the ROI for interferometry."""

    def __init__(self, input_image, image_ft, RMS_error):
        super().__init__(None, title="Phase View")
        self._panel = wx.Panel(self)

        wx_img_real = _np_grey_img_to_wx_image(input_image)
        wx_img_fourier = _np_grey_img_to_wx_image(image_ft)

        self._canvas = FloatCanvas(self._panel, size=wx_img_real.GetSize())
        self._real_bmp = self._canvas.AddBitmap(
            wx_img_real, (0, 0), Position="cc"
        )
        self._fourier_bmp = self._canvas.AddBitmap(
            wx_img_fourier, (0, 0), Position="cc"
        )
        # By default, show real and hide the fourier transform.
        self._fourier_bmp.Hide()

        save_btn = wx.ToggleButton(self._panel, label="Show Fourier")
        save_btn.Bind(wx.EVT_TOGGLEBUTTON, self.OnToggleFourier)

        rms_txt = wx.StaticText(
            self._panel, label="RMS difference: %.05f" % (RMS_error)
        )

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(self._canvas)

        bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bottom_sizer.Add(save_btn, wx.SizerFlags().Center().Border())
        bottom_sizer.Add(rms_txt, wx.SizerFlags().Center().Border())
        panel_sizer.Add(bottom_sizer)

        self._panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(self._panel)
        self.SetSizerAndFit(frame_sizer)

    def OnToggleFourier(self, event: wx.CommandEvent) -> None:
        show_fourier = event.IsChecked()
        # These bmp are wx.lib.floatcanvas.FCObjects.Bitmap and not
        # wx.Bitmap.  Their Show method does not take show argument
        # and therefore we can't do `Show(show_fourier)`.
        if show_fourier:
            self._fourier_bmp.Show()
            self._real_bmp.Hide()
        else:
            self._real_bmp.Show()
            self._fourier_bmp.Hide()
        self._canvas.Draw(Force=True)


class _CharacterisationAssayViewer(wx.Frame):
    def __init__(self, parent, characterisation_assay):
        super().__init__(parent, title="Characterisation Asssay")
        root_panel = wx.Panel(self)

        figure = Figure()

        img_ax = figure.add_subplot(1, 2, 1)
        img_ax.imshow(characterisation_assay)

        diag_ax = figure.add_subplot(1, 2, 2)
        assay_diag = np.diag(characterisation_assay)
        diag_ax.plot(assay_diag)

        canvas = FigureCanvas(root_panel, wx.ID_ANY, figure)

        info_txt = wx.StaticText(
            root_panel,
            label=(
                "Mean Zernike reconstruction accuracy: %0.5f"
                % np.mean(assay_diag)
            ),
        )

        panel_sizer = wx.BoxSizer(wx.VERTICAL)
        panel_sizer.Add(info_txt, wx.SizerFlags().Centre().Border())
        panel_sizer.Add(canvas, wx.SizerFlags(1).Expand())
        root_panel.SetSizer(panel_sizer)

        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        frame_sizer.Add(root_panel, wx.SizerFlags().Expand())
        self.SetSizerAndFit(frame_sizer)


class MicroscopeAOCompositeDevice(device.Device):
    def __init__(self, name, aoComp_config={}):
        super(self.__class__, self).__init__(name, aoComp_config)
        self.proxy = None
        self.sendImage = False
        self.curCamera = None

        self.buttonName = "AO composite device"

        ## Connect to the remote program

    def initialize(self):
        self.proxy = Pyro4.Proxy(self.uri)
        self.proxy.set_trigger(cp_ttype="FALLING_EDGE", cp_tmode="ONCE")
        self.no_actuators = self.proxy.get_n_actuators()
        self.config_dir = wx.GetApp().Config["global"].get("config-dir")

        # Need initial values for system flat calculations
        self.sys_flat_num_it = 10
        self.sys_error_thresh = np.inf
        self.sysFlatNollZernike = np.linspace(
            start=4, stop=68, num=65, dtype=int
        )
        self.sys_flat_values = None

        # Need intial values for sensorless AO
        self.numMes = 9
        self.num_it = 2
        self.z_max = 1.5
        self.z_min = -1.5
        self.nollZernike = np.asarray([11, 22, 5, 6, 7, 8, 9, 10])

        # Excercise the DM to remove residual static and then set to 0 position
        for ii in range(50):
            self.proxy.send(np.random.rand(self.no_actuators))
            time.sleep(0.01)
        self.proxy.reset()

        # Load values from config
        try:
            self.parameters = userConfig.getValue("dm_circleParams")
            self.proxy.set_roi(
                self.parameters[0], self.parameters[1], self.parameters[2]
            )
        except:
            pass

        try:
            self.controlMatrix = np.asarray(
                userConfig.getValue("dm_controlMatrix")
            )
            self.proxy.set_controlMatrix(self.controlMatrix)
        except:
            pass

        # subscribe to enable camera event to get access the new image queue
        events.subscribe(
            "camera enable", lambda c, isOn: self.enablecamera(c, isOn)
        )

    def finalizeInitialization(self):
        # A mapping of context-menu entries to functions.
        # Define in tuples - easier to read and reorder.
        menuTuples = (
            ("Fourier metric", "fourier"),
            ("Contrast metric", "contrast"),
            ("Fourier Power metric", "fourier_power"),
            ("Gradient metric", "gradient"),
            ("Second Moment metric", "second_moment"),
            (
                "Set System Flat Calculation Paramterers",
                self.set_sys_flat_param,
            ),
            ("Set Sensorless Parameters", self.set_sensorless_param),
        )
        # Store as ordered dict for easy item->func lookup.
        self.menuItems = OrderedDict(menuTuples)

    ### Context menu and handlers ###
    def menuCallback(self, index, item):
        try:
            self.menuItems[item]()
        except TypeError:
            return self.proxy.set_metric(self.menuItems[item])

    def set_sys_flat_param(self):
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            None,
            "Set the parameters for Sensorless Adaptive Optics routine",
            [
                "Number of iterations",
                "Error threshold",
                "System Flat Noll indeces",
            ],
            (
                self.sys_flat_num_it,
                self.sys_error_thresh,
                self.sysFlatNollZernike.tolist(),
            ),
        )
        self.sys_flat_num_it = int(inputs[0])
        if inputs[1] == "inf":
            self.sys_error_thresh = np.inf
        else:
            self.sys_error_thresh = int(inputs[1])
        if inputs[-1][1:-1].split(", ") == [""]:
            self.sysFlatNollZernike = None
        else:
            self.sysFlatNollZernike = np.asarray(
                [int(z_ind) for z_ind in inputs[-1][1:-1].split(", ")]
            )

    def set_sensorless_param(self):
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            None,
            "Set the parameters for Sensorless Adaptive Optics routine",
            [
                "Aberration range minima",
                "Aberration range maxima",
                "Number of measurements",
                "Number of repeats",
                "Noll indeces",
            ],
            (
                self.z_min,
                self.z_max,
                self.numMes,
                self.num_it,
                self.nollZernike.tolist(),
            ),
        )
        self.z_min, self.z_max, self.numMes, self.num_it = [
            i for i in inputs[:-1]
        ]
        self.z_min = float(self.z_min)
        self.z_max = float(self.z_max)
        self.numMes = int(self.numMes)
        self.num_it = int(self.num_it)
        self.nollZernike = np.asarray(
            [int(z_ind) for z_ind in inputs[-1][1:-1].split(", ")]
        )

    def onRightMouse(self, event):
        menu = cockpit.gui.device.Menu(
            self.menuItems.keys(), self.menuCallback
        )
        menu.show(event)

    def takeImage(self):
        wx.GetApp().Imager.takeImage()

    def enablecamera(self, camera, isOn):
        self.curCamera = camera
        # Subscribe to new image events only after canvas is prepared.

    ### UI functions ###
    def makeUI(self, parent):
        self.panel = wx.Panel(parent)
        self.panel.SetDoubleBuffered(True)
        sizer = wx.BoxSizer(wx.VERTICAL)
        label_setup = cockpit.gui.device.Label(
            parent=self.panel, label="AO set-up"
        )
        sizer.Add(label_setup)
        rowSizer = wx.BoxSizer(wx.VERTICAL)
        self.elements = OrderedDict()

        # Button to select the interferometer ROI
        selectCircleButton = wx.Button(self.panel, label="Select ROI")
        selectCircleButton.Bind(wx.EVT_BUTTON, self.onSelectCircle)
        self.elements["selectCircleButton"] = selectCircleButton

        # Visualise current interferometric phase
        visPhaseButton = wx.Button(self.panel, label="Visualise Phase")
        visPhaseButton.Bind(wx.EVT_BUTTON, lambda evt: self.onVisualisePhase())
        self.elements["visPhaseButton"] = visPhaseButton

        # Button to calibrate the DM
        calibrateButton = wx.Button(self.panel, label="Calibrate")
        calibrateButton.Bind(wx.EVT_BUTTON, lambda evt: self.onCalibrate())
        self.elements["calibrateButton"] = calibrateButton

        characteriseButton = wx.Button(self.panel, label="Characterise")
        characteriseButton.Bind(
            wx.EVT_BUTTON, lambda evt: self.onCharacterise()
        )
        self.elements["characteriseButton"] = characteriseButton

        sysFlatCalcButton = wx.Button(
            self.panel, label="Calculate System Flat"
        )
        sysFlatCalcButton.Bind(wx.EVT_BUTTON, lambda evt: self.onSysFlatCalc())
        self.elements["sysFlatCalcButton"] = sysFlatCalcButton

        label_use = cockpit.gui.device.Label(parent=self.panel, label="AO use")
        self.elements["label_use"] = label_use

        # Reset the DM actuators
        resetButton = wx.Button(self.panel, label="Reset DM")
        resetButton.Bind(wx.EVT_BUTTON, lambda evt: self.proxy.reset())
        self.elements["resetButton"] = resetButton

        # Apply the actuator values correcting the system aberrations
        applySysFlat = wx.Button(self.panel, label="System Flat")
        applySysFlat.Bind(wx.EVT_BUTTON, lambda evt: self.onApplySysFlat())
        self.elements["applySysFlat"] = applySysFlat

        # Apply last actuator values
        applyLastPatternButton = wx.Button(
            self.panel, label="Apply last pattern"
        )
        applyLastPatternButton.Bind(
            wx.EVT_BUTTON, lambda evt: self.onApplyLastPattern()
        )
        self.elements["applyLastPatternButton"] = applyLastPatternButton

        # Button to perform sensorless correction
        sensorlessAOButton = wx.Button(self.panel, label="Sensorless AO")
        sensorlessAOButton.Bind(
            wx.EVT_BUTTON, lambda evt: self.displaySensorlessAOMenu()
        )
        self.elements["Sensorless AO"] = sensorlessAOButton

        self.panel.Bind(wx.EVT_CONTEXT_MENU, self.onRightMouse)

        for e in self.elements.values():
            rowSizer.Add(e, 0, wx.EXPAND)
        sizer.Add(rowSizer, 0, wx.EXPAND)
        self.panel.SetSizerAndFit(sizer)
        self.hasUI = True
        return self.panel

    def bin_ndarray(self, ndarray, new_shape, operation="sum"):
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
        if not operation in ["sum", "mean"]:
            raise ValueError("Operation not supported.")
        if ndarray.ndim != len(new_shape):
            raise ValueError(
                "Shape mismatch: {} -> {}".format(ndarray.shape, new_shape)
            )
        compression_pairs = [
            (d, c // d) for d, c in zip(new_shape, ndarray.shape)
        ]
        flattened = [l for p in compression_pairs for l in p]
        ndarray = ndarray.reshape(flattened)
        for i in range(len(new_shape)):
            op = getattr(ndarray, operation)
            ndarray = op(-1 * (i + 1))
        return ndarray

    def onSelectCircle(self, event):
        image_raw = self.proxy.acquire_raw()
        if np.max(image_raw) > 10:
            original_dim = int(np.shape(image_raw)[0])
            resize_dim = 512

            while original_dim % resize_dim is not 0:
                resize_dim -= 1

            if resize_dim < original_dim / resize_dim:
                resize_dim = int(np.round(original_dim / resize_dim))

            scale_factor = original_dim / resize_dim
            temp = self.bin_ndarray(
                image_raw, new_shape=(resize_dim, resize_dim), operation="mean"
            )
            self.createCanvas(temp, scale_factor)
        else:
            print("Detecting nothing but background noise")

    def createCanvas(self, temp, scale_factor):
        temp = np.require(temp, requirements="C")
        frame = _ROISelect(input_image=temp, scale_factor=scale_factor)
        frame.Show()

    def onCalibrate(self):
        self.parameters = userConfig.getValue("dm_circleParams")
        self.proxy.set_roi(
            self.parameters[0], self.parameters[1], self.parameters[2]
        )

        try:
            self.proxy.get_roi()
        except Exception as e:
            try:
                self.parameters = userConfig.getValue("dm_circleParams")
                self.proxy.set_roi(
                    self.parameters[0], self.parameters[1], self.parameters[2]
                )
            except:
                raise e

        try:
            self.proxy.get_fourierfilter()
        except Exception as e:
            try:
                test_image = self.proxy.acquire()
                self.proxy.set_fourierfilter(
                    test_image=test_image,
                    window_dim=50,
                    mask_di=int((2 * self.parameters[2]) * (3.0 / 16.0)),
                )
            except:
                raise e

        controlMatrix = self.proxy.calibrate(numPokeSteps=5)
        userConfig.setValue(
            "dm_controlMatrix", np.ndarray.tolist(controlMatrix)
        )
        contol_matrix_file_path = os.path.join(
            self.config_dir, "control_matrix.txt"
        )
        np.savetxt(contol_matrix_file_path, controlMatrix)

    def onCharacterise(self):
        self.parameters = userConfig.getValue("dm_circleParams")
        self.proxy.set_roi(
            self.parameters[0], self.parameters[1], self.parameters[2]
        )

        try:
            self.proxy.get_roi()
        except Exception as e:
            try:
                self.parameters = userConfig.getValue("dm_circleParams")
                self.proxy.set_roi(
                    self.parameters[0], self.parameters[1], self.parameters[2]
                )
            except:
                raise e

        try:
            self.proxy.get_fourierfilter()
        except Exception as e:
            try:
                test_image = self.proxy.acquire()
                self.proxy.set_fourierfilter(
                    test_image=test_image,
                    window_dim=50,
                    mask_di=int((2 * self.parameters[2]) * (3.0 / 16.0)),
                )
            except:
                raise e

        try:
            self.proxy.get_controlMatrix()
        except Exception as e:
            try:
                self.controlMatrix = np.asarray(
                    userConfig.getValue("dm_controlMatrix")
                )
                self.proxy.set_controlMatrix(self.controlMatrix)
            except:
                raise e
        assay = self.proxy.assess_character()

        if np.mean(assay[1:, 1:]) < 0:
            cm = self.proxy.get_controlMatrix()
            self.proxy.set_controlMatrix((-1 * cm))
            assay = assay * -1
            userConfig.setValue("dm_controlMatrix", np.ndarray.tolist(cm))
            contol_matrix_file_path = os.path.join(
                self.config_dir, "control_matrix.txt"
            )
            np.savetxt(contol_matrix_file_path, cm)

        file_path = os.path.join(self.config_dir, "characterisation_assay")
        np.save(file_path, assay)
        # The default system corrections should be for the zernike modes we can accurately recreate
        self.sysFlatNollZernike = (np.where(np.diag(assay) > 0.75)[0]) + 1

        # Show characterisation assay, excluding piston
        frame = _CharacterisationAssayViewer(assay[1:, 1:])
        frame.Show()

    def onSysFlatCalc(self):
        self.parameters = userConfig.getValue("dm_circleParams")
        self.proxy.set_roi(
            self.parameters[0], self.parameters[1], self.parameters[2]
        )

        # Check we have the interferogram ROI
        try:
            self.proxy.get_roi()
        except Exception as e:
            try:
                param = np.asarray(userConfig.getValue("dm_circleParams"))
                self.proxy.set_roi(y0=param[0], x0=param[1], radius=param[2])
            except:
                raise e

        # Check we have a Fourier filter
        try:
            self.proxy.get_fourierfilter()
        except:
            try:
                test_image = self.proxy.acquire()
                self.proxy.set_fourierfilter(
                    test_image=test_image,
                    window_dim=50,
                    mask_di=int((2 * self.parameters[2]) * (3.0 / 16.0)),
                )
            except Exception as e:
                raise e

        # Check the DM has been calibrated
        try:
            self.proxy.get_controlMatrix()
        except Exception as e:
            try:
                self.controlMatrix = np.asarray(
                    userConfig.getValue("dm_controlMatrix")
                )
                self.proxy.set_controlMatrix(self.controlMatrix)
            except:
                raise e

        z_ignore = np.zeros(self.no_actuators)
        if self.sysFlatNollZernike is not None:
            z_ignore[self.sysFlatNollZernike - 1] = 1
        self.sys_flat_values, best_z_amps_corrected = self.proxy.flatten_phase(
            iterations=self.sys_flat_num_it,
            error_thresh=self.sys_error_thresh,
            z_modes_ignore=z_ignore,
        )

        userConfig.setValue(
            "dm_sys_flat", np.ndarray.tolist(self.sys_flat_values)
        )
        print("Zernike modes amplitudes corrected:\n", best_z_amps_corrected)
        print("System flat actuator values:\n", self.sys_flat_values)

    def onVisualisePhase(self):
        self.parameters = userConfig.getValue("dm_circleParams")
        self.proxy.set_roi(
            self.parameters[0], self.parameters[1], self.parameters[2]
        )

        try:
            self.proxy.get_roi()
        except Exception as e:
            try:
                param = np.asarray(userConfig.getValue("dm_circleParams"))
                self.proxy.set_roi(y0=param[0], x0=param[1], radius=param[2])
            except:
                raise e

        try:
            self.proxy.get_fourierfilter()
        except:
            try:
                test_image = self.proxy.acquire()
                self.proxy.set_fourierfilter(
                    test_image=test_image,
                    window_dim=50,
                    mask_di=int((2 * self.parameters[2]) * (3.0 / 16.0)),
                )
            except Exception as e:
                raise e

        interferogram, unwrapped_phase = self.proxy.acquire_unwrapped_phase()
        z_amps = self.proxy.getzernikemodes(unwrapped_phase, 3)
        unwrapped_phase_mptt = unwrapped_phase - aotools.phaseFromZernikes(
            z_amps[0:3], unwrapped_phase.shape[0]
        )
        unwrapped_RMS_error = self.proxy.wavefront_rms_error(
            unwrapped_phase_mptt
        )

        interferogram_file_path = os.path.join(
            self.config_dir, "interferogram"
        )
        np.save(interferogram_file_path, interferogram)

        interferogram_ft = np.fft.fftshift(np.fft.fft2(interferogram))
        interferogram_ft_file_path = os.path.join(
            self.config_dir, "interferogram_ft"
        )
        np.save(interferogram_ft_file_path, interferogram_ft)

        unwrapped_phase_file_path = os.path.join(
            self.config_dir, "unwrapped_phase"
        )
        np.save(unwrapped_phase_file_path, unwrapped_phase)

        unwrapped_phase = np.require(unwrapped_phase, requirements="C")
        power_spectrum = np.require(
            np.log(abs(interferogram_ft)), requirements="C"
        )

        frame = _PhaseViewer(
            unwrapped_phase, power_spectrum, unwrapped_RMS_error
        )
        frame.Show()

    def onApplySysFlat(self):
        if self.sys_flat_values is None:
            self.sys_flat_values = np.asarray(
                userConfig.getValue("dm_sys_flat")
            )

        self.proxy.send(self.sys_flat_values)

    def onApplyLastPattern(self):
        last_ac = self.proxy.get_last_actuator_values()
        self.proxy.send(last_ac)

    ### Sensorless AO functions ###

    ## Display a menu to the user letting them choose which camera
    # to use to perform sensorless AO. Of course, if only one camera is
    # available, then we just perform sensorless AO.
    def displaySensorlessAOMenu(self):
        self.showCameraMenu(
            "Perform sensorless AO with %s camera", self.correctSensorlessSetup
        )

    ## Generate a menu where the user can select a camera to use to perform
    # some action.
    # \param text String template to use for entries in the menu.
    # \param action Function to call with the selected camera as a parameter.
    def showCameraMenu(self, text, action):
        cameras = depot.getActiveCameras()
        if len(cameras) == 1:
            action(cameras[0])
        else:
            menu = wx.Menu()
            for i, camera in enumerate(cameras):
                menu.Append(i + 1, text % camera.descriptiveName)
                self.panel.Bind(
                    wx.EVT_MENU,
                    lambda event, camera=camera: action(camera),
                    id=i + 1,
                )
            cockpit.gui.guiUtils.placeMenuAtMouse(self.panel, menu)

    def correctSensorlessSetup(self, camera):
        print("Performing sensorless AO setup")
        # Note: Default is to correct Primary and Secondary Spherical aberration and both
        # orientations of coma, astigmatism and trefoil
        print("Checking for control matrix")
        try:
            self.proxy.get_controlMatrix()
        except Exception as e:
            try:
                self.controlMatrix = np.asarray(
                    userConfig.getValue("dm_controlMatrix")
                )
                self.proxy.set_controlMatrix(self.controlMatrix)
            except:
                raise e

        self.controlMatrix = self.proxy.get_controlMatrix()
        if self.controlMatrix is None:
            raise Exception(
                "No control matrix exists. Please calibrate the DM or load a control matrix"
            )

        print("Setting Zernike modes")

        self.actuator_offset = None

        self.sensorless_correct_coef = np.zeros(self.no_actuators)

        print("Subscribing to camera events")
        # Subscribe to camera events
        self.camera = camera
        events.subscribe(
            events.NEW_IMAGE % self.camera.name, self.correctSensorlessImage
        )

        # Get pixel size
        self.objectives = wx.GetApp().Objectives.GetCurrent().lens_ID
        self.pixelSize = wx.GetApp().Objectives.GetPixelSize() * 10 ** -6

        # Initialise the Zernike modes to apply
        print("Initialising the Zernike modes to apply")
        self.z_steps = np.linspace(self.z_min, self.z_max, self.numMes)

        for ii in range(self.num_it):
            it_zernike_applied = np.zeros(
                (self.numMes * self.nollZernike.shape[0], self.no_actuators)
            )
            for noll_ind in self.nollZernike:
                ind = np.where(self.nollZernike == noll_ind)[0][0]
                it_zernike_applied[
                    ind * self.numMes : (ind + 1) * self.numMes, noll_ind - 1
                ] = self.z_steps
            if ii == 0:
                self.zernike_applied = it_zernike_applied
            else:
                self.zernike_applied = np.concatenate(
                    (self.zernike_applied, it_zernike_applied)
                )

        # Initialise stack to store correction iumages
        print("Initialising stack to store correction images")
        self.correction_stack = []

        print("Applying the first Zernike mode")
        # Apply the first Zernike mode
        print(self.zernike_applied[len(self.correction_stack), :])
        self.proxy.set_phase(
            self.zernike_applied[len(self.correction_stack), :],
            offset=self.actuator_offset,
        )

        # Take image. This will trigger the iterative sensorless AO correction
        wx.CallAfter(self.takeImage)

    def correctSensorlessImage(self, image, timestamp):
        if len(self.correction_stack) < self.zernike_applied.shape[0]:
            print(
                "Correction image %i/%i"
                % (
                    len(self.correction_stack) + 1,
                    self.zernike_applied.shape[0],
                )
            )
            # Store image for current applied phase
            self.correction_stack.append(np.ndarray.tolist(image))
            wx.CallAfter(self.correctSensorlessProcessing)
        else:
            print("Error in unsubscribing to camera events. Trying again")
            events.unsubscribe(
                events.NEW_IMAGE % self.camera.name,
                self.correctSensorlessImage,
            )

    def correctSensorlessProcessing(self):
        print("Processing sensorless image")
        if len(self.correction_stack) < self.zernike_applied.shape[0]:
            if len(self.correction_stack) % self.numMes == 0:
                # Find aberration amplitudes and correct
                ind = int(len(self.correction_stack) / self.numMes)
                nollInd = (
                    np.where(
                        self.zernike_applied[len(self.correction_stack) - 1, :]
                        != 0
                    )[0][0]
                    + 1
                )
                print("Current Noll index being corrected: %i" % nollInd)
                current_stack = np.asarray(self.correction_stack)[
                    (ind - 1) * self.numMes : ind * self.numMes, :, :
                ]
                (
                    amp_to_correct,
                    ac_pos_correcting,
                ) = self.proxy.correct_sensorless_single_mode(
                    image_stack=current_stack,
                    zernike_applied=self.z_steps,
                    nollIndex=nollInd,
                    offset=self.actuator_offset,
                    wavelength=500 * 10 ** -9,
                    NA=1.1,
                    pixel_size=self.pixelSize,
                )
                self.actuator_offset = ac_pos_correcting
                self.sensorless_correct_coef[nollInd - 1] += amp_to_correct
                print("Aberrations measured: ", self.sensorless_correct_coef)
                print("Actuator positions applied: ", self.actuator_offset)

                # Advance counter by 1 and apply next phase
                self.proxy.set_phase(
                    self.zernike_applied[len(self.correction_stack), :],
                    offset=self.actuator_offset,
                )

                # Take image, but ensure it's called after the phase is applied
                wx.CallAfter(self.takeImage)
            else:
                # Advance counter by 1 and apply next phase
                self.proxy.set_phase(
                    self.zernike_applied[len(self.correction_stack), :],
                    offset=self.actuator_offset,
                )

                # Take image, but ensure it's called after the phase is applied
                time.sleep(0.1)
                wx.CallAfter(self.takeImage)
        else:
            # Once all images have been obtained, unsubscribe
            print("Unsubscribing to camera %s events" % self.camera.name)
            events.unsubscribe(
                events.NEW_IMAGE % self.camera.name,
                self.correctSensorlessImage,
            )

            # Save full stack of images used
            self.correction_stack = np.asarray(self.correction_stack)
            correction_stack_file_path = os.path.join(
                self.config_dir,
                "sensorless_AO_correction_stack_%i%i%i_%i%i"
                % (
                    time.gmtime()[2],
                    time.gmtime()[1],
                    time.gmtime()[0],
                    time.gmtime()[3],
                    time.gmtime()[4],
                ),
            )
            np.save(correction_stack_file_path, self.correction_stack)
            zernike_applied_file_path = os.path.join(
                self.config_dir,
                "sensorless_AO_zernike_applied_%i%i%i_%i%i"
                % (
                    time.gmtime()[2],
                    time.gmtime()[1],
                    time.gmtime()[0],
                    time.gmtime()[3],
                    time.gmtime()[4],
                ),
            )
            np.save(zernike_applied_file_path, self.zernike_applied)
            nollZernike_file_path = os.path.join(
                self.config_dir,
                "sensorless_AO_nollZernike_%i%i%i_%i%i"
                % (
                    time.gmtime()[2],
                    time.gmtime()[1],
                    time.gmtime()[0],
                    time.gmtime()[3],
                    time.gmtime()[4],
                ),
            )
            np.save(nollZernike_file_path, self.nollZernike)

            # Find aberration amplitudes and correct
            ind = int(len(self.correction_stack) / self.numMes)
            nollInd = (
                np.where(
                    self.zernike_applied[len(self.correction_stack) - 1, :]
                    != 0
                )[0][0]
                + 1
            )
            print("Current Noll index being corrected: %i" % nollInd)
            current_stack = np.asarray(self.correction_stack)[
                (ind - 1) * self.numMes : ind * self.numMes, :, :
            ]
            (
                amp_to_correct,
                ac_pos_correcting,
            ) = self.proxy.correct_sensorless_single_mode(
                image_stack=current_stack,
                zernike_applied=self.z_steps,
                nollIndex=nollInd,
                offset=self.actuator_offset,
                wavelength=500 * 10 ** -9,
                NA=1.1,
                pixel_size=self.pixelSize,
            )
            self.actuator_offset = ac_pos_correcting
            self.sensorless_correct_coef[nollInd - 1] += amp_to_correct
            print("Aberrations measured: ", self.sensorless_correct_coef)
            print("Actuator positions applied: ", self.actuator_offset)
            sensorless_correct_coef_file_path = os.path.join(
                self.config_dir,
                "sensorless_correct_coef_%i%i%i_%i%i"
                % (
                    time.gmtime()[2],
                    time.gmtime()[1],
                    time.gmtime()[0],
                    time.gmtime()[3],
                    time.gmtime()[4],
                ),
            )
            np.save(
                sensorless_correct_coef_file_path, self.sensorless_correct_coef
            )
            ac_pos_sensorless_file_path = os.path.join(
                self.config_dir,
                "ac_pos_sensorless_%i%i%i_%i%i"
                % (
                    time.gmtime()[2],
                    time.gmtime()[1],
                    time.gmtime()[0],
                    time.gmtime()[3],
                    time.gmtime()[4],
                ),
            )
            np.save(ac_pos_sensorless_file_path, self.actuator_offset)

            log_file_path = os.path.join(
                self.config_dir, "sensorless_AO_logger.txt"
            )
            log_file = open(log_file_path, "a+")
            log_file.write(
                "Time stamp: %i:%i:%i %i/%i/%i\n"
                % (
                    time.gmtime()[3],
                    time.gmtime()[4],
                    time.gmtime()[5],
                    time.gmtime()[2],
                    time.gmtime()[1],
                    time.gmtime()[0],
                )
            )
            log_file.write("Aberrations measured: ")
            log_file.write(str(self.sensorless_correct_coef))
            log_file.write("\n")
            log_file.write("Actuator positions applied: ")
            log_file.write(str(self.actuator_offset))
            log_file.write("\n")
            log_file.close()

            print("Actuator positions applied: ", self.actuator_offset)
            self.proxy.send(self.actuator_offset)
            wx.CallAfter(self.takeImage)
