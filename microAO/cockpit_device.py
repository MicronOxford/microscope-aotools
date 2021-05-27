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

"""Device file for a microscope-aotools AdaptiveOpticsDevice.

This file provides the cockpit end of the driver for a deformable
mirror as currently mounted on DeepSIM in Oxford.

"""

import os
import time
import typing

import aotools
import cockpit.devices
import cockpit.devices.device
import cockpit.gui.device
import cockpit.interfaces.imager
import numpy as np
import Pyro4
import wx
from cockpit import depot, events
from cockpit.util import logger, userConfig
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from wx.lib.floatcanvas.FloatCanvas import FloatCanvas

from microAO.aoDev import AdaptiveOpticsDevice


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


def _bin_ndarray(ndarray, new_shape):
    """Bins an ndarray in all axes based on the target shape by averaging.


    Number of output dimensions must match number of input dimensions
    and new axes must divide old ones.

    Example
    -------

    m = np.arange(0,100,1).reshape((10,10))
    n = bin_ndarray(m, new_shape=(5,5))
    print(n)
    [[ 5.5  7.5  9.5 11.5 13.5]
     [25.5 27.5 29.5 31.5 33.5]
     [45.5 47.5 49.5 51.5 53.5]
     [65.5 67.5 69.5 71.5 73.5]
     [85.5 87.5 89.5 91.5 93.5]]

    Function acquired from Stack Overflow at
    https://stackoverflow.com/a/29042041. Stack Overflow or other
    Stack Exchange sites is cc-wiki (aka cc-by-sa) licensed and
    requires attribution.

    """
    if ndarray.ndim != len(new_shape):
        raise ValueError(
            "Shape mismatch: {} -> {}".format(ndarray.shape, new_shape)
        )
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        ndarray = ndarray.mean(-1 * (i + 1))
    return ndarray


def _computeUnwrappedPhaseMPTT(unwrapped_phase):
    # XXX: AdaptiveOpticsDevice.getzernikemodes method does not
    # actually make use of its instance.  It should have been a free
    # function or at least a class method.  Using it like this means
    # we can compute it client-side instead of having send the data.
    # This should be changed in microscope-aotools.
    z_amps = AdaptiveOpticsDevice.getzernikemodes(None, unwrapped_phase, 3)
    phase = aotools.phaseFromZernikes(z_amps[0:3], unwrapped_phase.shape[0])
    return unwrapped_phase - phase


def _computePowerSpectrum(interferogram):
    interferogram_ft = np.fft.fftshift(np.fft.fft2(interferogram))
    power_spectrum = np.log(abs(interferogram_ft))
    return power_spectrum


def _np_save_with_timestamp(data, basename_prefix):
    dirname = wx.GetApp().Config["log"].getpath("dir")
    timestamp = time.strftime("%Y%m%d_%H%M", time.gmtime())
    basename = basename_prefix + "_" + timestamp
    np.save(os.path.join(dirname, basename), data)


class _ROISelect(wx.Frame):
    """Display a window that allows the user to select a circular area.

    This is a window for selecting the ROI for interferometry.
    """

    def __init__(
        self, parent, input_image: np.ndarray, initial_roi, scale_factor=1
    ) -> None:
        super().__init__(parent, title="ROI selector")
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
            self.canvas.PixelToWorld(initial_roi[:2]),
            initial_roi[2] * 2,
            LineColor="cyan",
            LineWidth=2,
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
        del event
        roi = [x * self._scale_factor for x in self.ROI]
        userConfig.setValue("dm_circleParams", (roi[1], roi[0], roi[2]))

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

    def __init__(self, parent, input_image, image_ft, RMS_error):
        super().__init__(parent, title="Phase View")
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


def log_correction_applied(
    correction_stack,
    zernike_applied,
    nollZernike,
    sensorless_correct_coef,
    actuator_offset,
):
    # Save full stack of images used
    _np_save_with_timestamp(
        np.asarray(correction_stack), "sensorless_AO_correction_stack",
    )

    _np_save_with_timestamp(
        zernike_applied, "sensorless_AO_zernike_applied",
    )

    _np_save_with_timestamp(nollZernike, "sensorless_AO_nollZernike")
    _np_save_with_timestamp(
        sensorless_correct_coef, "sensorless_correct_coef",
    )

    _np_save_with_timestamp(actuator_offset, "ac_pos_sensorless")

    ao_log_filepath = os.path.join(
        wx.GetApp().Config["log"].getpath("dir"), "sensorless_AO_logger.txt",
    )
    with open(ao_log_filepath, "a+") as fh:
        fh.write(
            "Time stamp: %s\n" % time.strftime("%Y/%m/%d %H:%M", time.gmtime())
        )
        fh.write("Aberrations measured: %s\n" % (sensorless_correct_coef))
        fh.write("Actuator positions applied: %s\n" % (str(actuator_offset)))


class MicroscopeAOCompositeDevicePanel(wx.Panel):
    def __init__(self, parent, device):
        super().__init__(parent)
        self.SetDoubleBuffered(True)

        self._device = device

        label_setup = cockpit.gui.device.Label(parent=self, label="AO setup")

        # Button to select the interferometer ROI
        selectCircleButton = wx.Button(self, label="Select ROI")
        selectCircleButton.Bind(wx.EVT_BUTTON, self.OnSelectROI)

        # Visualise current interferometric phase
        visPhaseButton = wx.Button(self, label="Visualise Phase")
        visPhaseButton.Bind(wx.EVT_BUTTON, self.OnVisualisePhase)

        # Button to calibrate the DM
        calibrateButton = wx.Button(self, label="Calibrate")
        calibrateButton.Bind(wx.EVT_BUTTON, self.OnCalibrate)

        characteriseButton = wx.Button(self, label="Characterise")
        characteriseButton.Bind(wx.EVT_BUTTON, self.OnCharacterise)

        sysFlatCalcButton = wx.Button(self, label="Calculate System Flat")
        sysFlatCalcButton.Bind(wx.EVT_BUTTON, self.OnCalcSystemFlat)

        label_use = cockpit.gui.device.Label(parent=self, label="AO use")

        # Reset the DM actuators
        resetButton = wx.Button(self, label="Reset DM")
        resetButton.Bind(wx.EVT_BUTTON, self.OnResetDM)

        # Apply the actuator values correcting the system aberrations
        applySysFlat = wx.Button(self, label="System Flat")
        applySysFlat.Bind(wx.EVT_BUTTON, self.OnSystemFlat)

        # Apply last actuator values
        applyLastPatternButton = wx.Button(self, label="Apply last pattern")
        applyLastPatternButton.Bind(wx.EVT_BUTTON, self.OnApplyLastPattern)

        # Button to perform sensorless correction
        sensorlessAOButton = wx.Button(self, label="Sensorless AO")
        sensorlessAOButton.Bind(wx.EVT_BUTTON, self.OnSensorlessAO)

        # Right click button to select the metric and other
        # parameters.
        # FIXME: this is horrible UI with very low discoverability.
        # Change to have a proper menu?
        self.Bind(wx.EVT_CONTEXT_MENU, self.OnContextMenu)

        self._menu_item_id_to_metric: typing.Dict[int, str] = {}
        self._menu_item_id_to_callback: typing.Dict[
            int, typing.Callable[[], None]
        ] = {}

        self._context_menu = wx.Menu()

        for label, metric in [
            ("Fourier metric", "fourier"),
            ("Contrast metric", "contrast"),
            ("Fourier Power metric", "fourier_power"),
            ("Gradient metric", "gradient"),
            ("Second Moment metric", "second_moment"),
        ]:
            menu_item = self._context_menu.AppendRadioItem(wx.ID_ANY, label)
            self._menu_item_id_to_metric[menu_item.GetId()] = metric
            self._context_menu.Bind(
                wx.EVT_MENU,
                self.OnContextMenuSelectMetric,
                id=menu_item.GetId(),
            )
        self._context_menu.AppendSeparator()

        for label, callback in [
            (
                "Set System Flat Calculation Parameters",
                self.SetSystemFlatCalculationParameters,
            ),
            ("Set Sensorless Parameters", self.SetSensorlessParameters),
        ]:
            menu_item = self._context_menu.Append(wx.ID_ANY, label)
            self._menu_item_id_to_callback[menu_item.GetId()] = callback
            self._context_menu.Bind(
                wx.EVT_MENU, self.OnContextMenuCallback, id=menu_item.GetId()
            )

        sizer = wx.BoxSizer(wx.VERTICAL)
        for btn in [
            label_setup,
            selectCircleButton,
            visPhaseButton,
            calibrateButton,
            characteriseButton,
            sysFlatCalcButton,
            label_use,
            resetButton,
            applySysFlat,
            applyLastPatternButton,
            sensorlessAOButton,
        ]:
            sizer.Add(btn, wx.SizerFlags(0).Expand())
        self.SetSizer(sizer)

    def OnContextMenu(self, event: wx.ContextMenuEvent) -> None:
        cockpit.gui.guiUtils.placeMenuAtMouse(
            event.GetEventObject(), self._context_menu
        )

    def OnContextMenuSelectMetric(self, event: wx.CommandEvent) -> None:
        metric = self._menu_item_id_to_metric[event.GetId()]
        self._device.proxy.set_metric(metric)

    def OnContextMenuCallback(self, event: wx.CommandEvent) -> None:
        callback = self._menu_item_id_to_callback[event.GetId()]
        callback()

    def OnSelectROI(self, event: wx.CommandEvent) -> None:
        del event
        image_raw = self._device.acquireRaw()
        if np.max(image_raw) > 10:
            original_dim = np.shape(image_raw)[0]
            resize_dim = 512

            while original_dim % resize_dim != 0:
                resize_dim -= 1

            if resize_dim < original_dim / resize_dim:
                resize_dim = int(np.round(original_dim / resize_dim))

            scale_factor = original_dim / resize_dim
            img = _bin_ndarray(image_raw, new_shape=(resize_dim, resize_dim))
            img = np.require(img, requirements="C")

            last_roi = userConfig.getValue("dm_circleParams",)
            # We need to check if getValue() returns None, instead of
            # passing a default value to getValue().  The reason is
            # that if there is no ROI at the start, by the time we get
            # here the device initialize has called updateROI which
            # also called getValue() which has have the side effect of
            # setting its value to None.  And we can't set a sensible
            # default at that time because we have no method to get
            # the wavefront camera sensor size.
            if last_roi is None:
                last_roi = (
                    *[d // 2 for d in image_raw.shape],
                    min(image_raw.shape) // 4,
                )

            last_roi = (
                last_roi[1] / scale_factor,
                last_roi[0] / scale_factor,
                last_roi[2] / scale_factor,
            )

            frame = _ROISelect(self, img, last_roi, scale_factor)
            frame.Show()
        else:
            wx.MessageBox(
                "Detected nothing but background noise.",
                caption="No good image acquired",
                style=wx.ICON_ERROR | wx.OK | wx.CENTRE,
            )

    def OnVisualisePhase(self, event: wx.CommandEvent) -> None:
        del event
        self._device.updateROI()
        self._device.checkFourierFilter()

        interferogram, unwrapped_phase = self._device.acquireUnwrappedPhase()
        power_spectrum = _computePowerSpectrum(interferogram)
        unwrapped_phase_mptt = _computeUnwrappedPhaseMPTT(unwrapped_phase)

        unwrapped_RMS_error = self._device.wavefrontRMSError(
            unwrapped_phase_mptt
        )

        frame = _PhaseViewer(
            self, unwrapped_phase, power_spectrum, unwrapped_RMS_error
        )
        frame.Show()

    def OnCalibrate(self, event: wx.CommandEvent) -> None:
        del event
        self._device.calibrate()

    def OnCharacterise(self, event: wx.CommandEvent) -> None:
        del event
        assay = self._device.characterise()
        # Show characterisation assay, excluding piston.
        frame = _CharacterisationAssayViewer(self, assay[1:, 1:])
        frame.Show()

    def OnCalcSystemFlat(self, event: wx.CommandEvent) -> None:
        del event
        sys_flat_values, best_z_amps_corrected = self._device.sysFlatCalc()
        logger.log.debug(
            "Zernike modes amplitudes corrected:\n %s", best_z_amps_corrected
        )
        logger.log.debug("System flat actuator values:\n%s", sys_flat_values)

    def OnResetDM(self, event: wx.CommandEvent) -> None:
        del event
        self._device.reset()

    def OnSystemFlat(self, event: wx.CommandEvent) -> None:
        del event
        self._device.applySysFlat()

    def OnApplyLastPattern(self, event: wx.CommandEvent) -> None:
        del event
        self._device.applyLastPattern()

    def OnSensorlessAO(self, event: wx.CommandEvent) -> None:
        # Perform sensorless AO but if there is more than one camera
        # available display a menu letting the user choose a camera.
        del event

        action = self._device.correctSensorlessSetup

        cameras = depot.getActiveCameras()
        if not cameras:
            wx.MessageBox(
                "There are no cameras enabled.", caption="No cameras active"
            )
        elif len(cameras) == 1:
            action(cameras[0])
        else:
            menu = wx.Menu()
            for camera in cameras:
                menu_item = menu.Append(
                    wx.ID_ANY,
                    "Perform sensorless AO with %s camera"
                    % camera.descriptiveName,
                )
                self.Bind(
                    wx.EVT_MENU,
                    lambda event, camera=camera: action(camera),
                    menu_item,
                )
            cockpit.gui.guiUtils.placeMenuAtMouse(self, menu)

    def SetSystemFlatCalculationParameters(self) -> None:
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Set system flat parameters",
            [
                "Number of iterations",
                "Error threshold",
                "System Flat Noll indeces",
            ],
            (
                self._device.sys_flat_num_it,
                self._device.sys_error_thresh,
                self._device.sysFlatNollZernike.tolist(),
            ),
        )
        self._device.sys_flat_num_it = int(inputs[0])
        self._device.sys_error_thresh = np.float(inputs[1])

        # FIXME: we should probably do some input checking here and
        # maybe not include a space in `split(", ")`
        if inputs[2] == "":
            self._device.sysFlatNollZernike = None
        else:
            self._device.sysFlatNollZernike = np.asarray(
                [int(z_ind) for z_ind in inputs[-1][1:-1].split(", ")]
            )

    def SetSensorlessParameters(self) -> None:
        inputs = cockpit.gui.dialogs.getNumberDialog.getManyNumbersFromUser(
            self,
            "Set sensorless AO parameters",
            [
                "Aberration range minima",
                "Aberration range maxima",
                "Number of measurements",
                "Number of repeats",
                "Noll indeces",
            ],
            (
                self._device.z_min,
                self._device.z_max,
                self._device.numMes,
                self._device.num_it,
                self._device.nollZernike.tolist(),
            ),
        )
        self._device.z_min = float(inputs[0])
        self._device.z_max = float(inputs[1])
        self._device.numMes = int(inputs[2])
        self._device.num_it = int(inputs[3])
        self._device.nollZernike = np.asarray(
            [int(z_ind) for z_ind in inputs[4][1:-1].split(", ")]
        )


class MicroscopeAOCompositeDevice(cockpit.devices.device.Device):
    def __init__(self, name: str, config={}) -> None:
        super().__init__(name, config)
        self.proxy = None

    def initialize(self):
        self.proxy = Pyro4.Proxy(self.uri)
        self.proxy.set_trigger(cp_ttype="FALLING_EDGE", cp_tmode="ONCE")
        self.no_actuators = self.proxy.get_n_actuators()

        # Need initial values for system flat calculations
        self.sys_flat_num_it = 10
        self.sys_error_thresh = np.inf
        self.sysFlatNollZernike = np.linspace(
            start=4, stop=68, num=65, dtype=int
        )

        # Need intial values for sensorless AO
        self.numMes = 9
        self.num_it = 2
        self.z_max = 1.5
        self.z_min = -1.5
        self.nollZernike = np.asarray([11, 22, 5, 6, 7, 8, 9, 10])

        # Shared state for the new image callbacks during sensorless
        self.actuator_offset = None
        self.camera = None
        self.correction_stack = []
        self.sensorless_correct_coef = np.zeros(self.no_actuators)
        self.z_steps = np.linspace(self.z_min, self.z_max, self.numMes)
        self.zernike_applied = None

        # Excercise the DM to remove residual static and then set to 0 position
        for _ in range(50):
            self.proxy.send(np.random.rand(self.no_actuators))
            time.sleep(0.01)
        self.reset()

        # Load values from config
        try:
            self.updateROI()
        except Exception:
            pass

        try:
            controlMatrix = np.asarray(userConfig.getValue("dm_controlMatrix"))
            self.proxy.set_controlMatrix(controlMatrix)
        except Exception:
            pass

    def makeUI(self, parent):
        return MicroscopeAOCompositeDevicePanel(parent, self)

    def acquireRaw(self):
        return self.proxy.acquire_raw()

    def acquireUnwrappedPhase(self):
        return self.proxy.acquire_unwrapped_phase()

    def getZernikeModes(self, image_unwrap, noZernikeModes):
        return self.proxy.getzernikemodes(image_unwrap, noZernikeModes)

    def wavefrontRMSError(self, phase_map):
        return self.proxy.wavefront_rms_error(phase_map)

    def updateROI(self):
        circle_parameters = userConfig.getValue("dm_circleParams")
        self.proxy.set_roi(*circle_parameters)

        # Check we have the interferogram ROI
        try:
            self.proxy.get_roi()
        except Exception as e:
            try:
                self.proxy.set_roi(*circle_parameters)
            except Exception:
                raise e

    def checkFourierFilter(self):
        circle_parameters = userConfig.getValue("dm_circleParams")
        try:
            self.proxy.get_fourierfilter()
        except Exception as e:
            try:
                test_image = self.proxy.acquire()
                self.proxy.set_fourierfilter(
                    test_image=test_image,
                    window_dim=50,
                    mask_di=int((2 * circle_parameters[2]) * (3.0 / 16.0)),
                )
            except Exception:
                raise e

    def checkIfCalibrated(self):
        try:
            self.proxy.get_controlMatrix()
        except Exception as e:
            try:
                controlMatrix = np.asarray(
                    userConfig.getValue("dm_controlMatrix")
                )
                self.proxy.set_controlMatrix(controlMatrix)
            except Exception:
                raise e

    def calibrate(self):
        self.updateROI()
        self.checkFourierFilter()

        controlMatrix = self.proxy.calibrate(numPokeSteps=5)
        userConfig.setValue(
            "dm_controlMatrix", np.ndarray.tolist(controlMatrix)
        )

    def characterise(self):
        self.updateROI()
        self.checkFourierFilter()
        self.checkIfCalibrated()
        assay = self.proxy.assess_character()

        if np.mean(assay[1:, 1:]) < 0:
            controlMatrix = self.proxy.get_controlMatrix()
            self.proxy.set_controlMatrix((-1 * controlMatrix))
            assay = assay * -1
            userConfig.setValue(
                "dm_controlMatrix", np.ndarray.tolist(controlMatrix)
            )

        # The default system corrections should be for the zernike
        # modes we can accurately recreate.
        self.sysFlatNollZernike = ((np.diag(assay) > 0.75).nonzero()[0]) + 1

        return assay

    def sysFlatCalc(self):
        self.updateROI()
        self.checkFourierFilter()
        self.checkIfCalibrated()

        z_ignore = np.zeros(self.no_actuators)
        if self.sysFlatNollZernike is not None:
            z_ignore[self.sysFlatNollZernike - 1] = 1
        sys_flat_values, best_z_amps_corrected = self.proxy.flatten_phase(
            iterations=self.sys_flat_num_it,
            error_thresh=self.sys_error_thresh,
            z_modes_ignore=z_ignore,
        )

        userConfig.setValue("dm_sys_flat", np.ndarray.tolist(sys_flat_values))
        return sys_flat_values, best_z_amps_corrected

    def reset(self):
        self.proxy.reset()

    def applySysFlat(self):
        sys_flat_values = np.asarray(userConfig.getValue("dm_sys_flat"))
        self.proxy.send(sys_flat_values)

    def applyLastPattern(self):
        last_ac = self.proxy.get_last_actuator_values()
        self.proxy.send(last_ac)

    def correctSensorlessSetup(self, camera):
        logger.log.info("Performing sensorless AO setup")
        # Note: Default is to correct Primary and Secondary Spherical
        # aberration and both orientations of coma, astigmatism and
        # trefoil.

        self.checkIfCalibrated()

        # Shared state for the new image callbacks during sensorless
        self.actuator_offset = None
        self.camera = camera
        self.correction_stack = []  # list of corrected images
        self.sensorless_correct_coef = np.zeros(self.no_actuators)
        # Zernike modes to apply
        self.z_steps = np.linspace(self.z_min, self.z_max, self.numMes)
        self.zernike_applied = np.zeros((0, self.no_actuators))

        logger.log.debug("Subscribing to camera events")
        # Subscribe to camera events
        events.subscribe(
            events.NEW_IMAGE % self.camera.name, self.correctSensorlessImage
        )

        for ii in range(self.num_it):
            it_zernike_applied = np.zeros(
                (self.numMes * self.nollZernike.shape[0], self.no_actuators)
            )
            for noll_ind in self.nollZernike:
                ind = np.where(self.nollZernike == noll_ind)[0][0]
                it_zernike_applied[
                    ind * self.numMes : (ind + 1) * self.numMes, noll_ind - 1
                ] = self.z_steps

            self.zernike_applied = np.concatenate(
                [self.zernike_applied, it_zernike_applied]
            )

        logger.log.info("Applying the first Zernike mode")
        # Apply the first Zernike mode
        logger.log.debug(self.zernike_applied[len(self.correction_stack), :])
        self.proxy.set_phase(
            self.zernike_applied[len(self.correction_stack), :],
            offset=self.actuator_offset,
        )

        # Take image. This will trigger the iterative sensorless AO correction
        wx.CallAfter(wx.GetApp().Imager.takeImage)

    def correctSensorlessImage(self, image, timestamp):
        del timestamp
        if len(self.correction_stack) < self.zernike_applied.shape[0]:
            logger.log.info(
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
            logger.log.error(
                "Failed to unsubscribe to camera events. Trying again."
            )
            events.unsubscribe(
                events.NEW_IMAGE % self.camera.name,
                self.correctSensorlessImage,
            )

    def findAbberationAndCorrect(self):
        pixelSize = wx.GetApp().Objectives.GetPixelSize() * 10 ** -6

        # Find aberration amplitudes and correct
        ind = int(len(self.correction_stack) / self.numMes)
        nollInd = (
            np.where(self.zernike_applied[len(self.correction_stack) - 1, :])[
                0
            ][0]
            + 1
        )
        logger.log.debug("Current Noll index being corrected: %i" % nollInd)
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
            pixel_size=pixelSize,
        )
        self.actuator_offset = ac_pos_correcting
        self.sensorless_correct_coef[nollInd - 1] += amp_to_correct
        logger.log.debug(
            "Aberrations measured: ", self.sensorless_correct_coef
        )
        logger.log.debug("Actuator positions applied: ", self.actuator_offset)

    def correctSensorlessProcessing(self):
        logger.log.info("Processing sensorless image")
        if len(self.correction_stack) < self.zernike_applied.shape[0]:
            if len(self.correction_stack) % self.numMes == 0:
                self.findAbberationAndCorrect()

            # Advance counter by 1 and apply next phase
            self.proxy.set_phase(
                self.zernike_applied[len(self.correction_stack), :],
                offset=self.actuator_offset,
            )

        else:
            # Once all images have been obtained, unsubscribe
            logger.log.debug(
                "Unsubscribing to camera %s events" % self.camera.name
            )
            events.unsubscribe(
                events.NEW_IMAGE % self.camera.name,
                self.correctSensorlessImage,
            )

            self.findAbberationAndCorrect()

            log_correction_applied(
                self.correction_stack,
                self.zernike_applied,
                self.nollZernike,
                self.sensorless_correct_coef,
                self.actuator_offset,
            )

            logger.log.debug(
                "Actuator positions applied: %s", self.actuator_offset
            )
            self.proxy.send(self.actuator_offset)

        # Take image, but ensure it's called after the phase is applied
        time.sleep(0.1)
        wx.CallAfter(wx.GetApp().Imager.takeImage)
