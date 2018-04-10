#!/usr/bin/python
# -*- coding: utf-8
#
# Copyright 2017 Mick Phillips (mick.phillips@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Display a window that allows the user to select a circular area."""
import Tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import microAO.AdaptiveOpticsDevice.camera as cam

class App(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.pack()
        self.ratio = 4
        self.offset = [45, 50]
        self.create_widgets()

    def create_widgets(self):
        self.canvas = Canvas(self, width=600, height=600)
        self.array = cam.get_current_image()
        self.convert = Image.fromarray(self.array)
        self.convert_flip = self.convert.transpose(Image.FLIP_TOP_BOTTOM)
        self.image = ImageTk.PhotoImage(image = self.convert_flip)
        print(np.min(self.array))
        print(np.max(self.array))
        self.canvas.create_image(self.offset[0], self.offset[1], anchor = tk.NW, image = self.image)
        self.canvas.pack()

        self.btn_quit = tk.Button(self, text="Quit", command=self.quit)
        self.btn_quit.pack()

class Canvas(tk.Canvas):
    def __init__(self, *args, **kwargs):
        tk.Canvas.__init__(self, *args, **kwargs)
        self.bind("<Button-1>", self.on_click)
        self.bind("<Button-3>", self.on_click)
        self.bind("<B1-Motion>", self.circle_resize)
        self.bind("<B3-Motion>", self.circle_drag)
        self.bind("<ButtonRelease>", self.on_release)
        self.circle = None
        self.p_click = None
        self.bbox_click = None
        self.centre = [0,0]
        self.diameter = 0
        self.ratio = 4
        self.offset = [45, 50]

    def on_release(self, event):
        self.p_click = None
        self.bbox_click = None

    def on_click(self, event):
        if self.circle == None:
            self.circle = self.create_oval((event.x-1, event.y-1, event.x+1, event.y+1))
            self.centre[0] = (event.x - self.offset[0]) * self.ratio
            self.centre[1] = (event.y - self.offset[1]) * self.ratio
            self.diameter = (event.x+1 - event.x+1 + 1) * self.ratio
            np.save("alpao_circleParams", np.asarray([self.centre[0], self.centre[1], self.diameter]))

    def circle_resize(self, event):
        if self.circle is None:
            return
        if self.p_click is None:
            self.p_click = (event.x, event.y)
            self.bbox_click = self.bbox(self.circle)
            return
        bbox = self.bbox(self.circle)
        unscaledCentre = ((bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2)
        r0 = ((self.p_click[0] - unscaledCentre[0])**2 + (self.p_click[1] - unscaledCentre[1])**2)**0.5
        r1 = ((event.x - unscaledCentre[0])**2 + (event.y - unscaledCentre[1])**2)**0.5
        scale = r1 / r0
        self.scale(self.circle, unscaledCentre[0], unscaledCentre[1], scale, scale)
        self.p_click= (event.x, event.y)
        self.diameter = (self.bbox(self.circle)[2] - self.bbox(self.circle)[0]) * self.ratio
        np.save("alpao_circleParams", np.asarray([self.centre[0], self.centre[1], self.diameter]))

    def circle_drag(self, event):
        if self.circle is None:
            return
        if self.p_click is None:
            self.p_click = (event.x, event.y)
            return
        self.move(self.circle,
                  event.x - self.p_click[0],
                  event.y - self.p_click[1])
        self.p_click = (event.x, event.y)
        bbox = self.bbox(self.circle)
        unscaledCentre = ((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2)
        self.centre[0] = (unscaledCentre[0] - self.offset[0]) * self.ratio
        self.centre[1] = (unscaledCentre[1] - self.offset[1]) * self.ratio
        np.save("alpao_circleParams", np.asarray([self.centre[0], self.centre[1], self.diameter]))
        self.update()
