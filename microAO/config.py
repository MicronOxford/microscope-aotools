"""Config file for devicebase.

Import device classes, then define entries in DEVICES as:
    devices(CLASS, HOST, PORT, other_args)
"""
# Function to create record for each device.
from microscope.devices import device
# Import device modules/classes here.
from microscope.testsuite.devices import TestCamera
from microscope.testsuite.devices import TestDeformableMirror
from microAO.aoDev import AdaptiveOpticsDevice

DEVICES = [
   device(TestCamera, '127.0.0.1', 8005),
   device(TestDeformableMirror, '127.0.0.1', 8006),
   device(AdaptiveOpticsDevice, '127.0.0.1', 8007,
          camera_uri = 'PYRO:TestCamera@127.0.0.1:8005',
          mirror_uri = 'PYRO:TestMirror@127.0.0.1:8006')]
