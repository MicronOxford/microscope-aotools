# AOTools: A microscope add-on for adaptive optics

A composite device for controlling and using deformable mirror adaptive
optic elements. This composite device can be constructed from any
python-microscope compatible deformable mirrors and cameras (see 
python-microscope compatibility list) as well as certain spatial light
modulators. The functions presented allow a user to calibrate and 
characterise a deformable mirror using an interferometric wavefront 
sensor, to set control matrices determined by other calibration methods 
and perform sensorless adaptive optics correction with a variety of metrics.

##Set-up

AOTools is set-up like a regular python-microscope device and served with
the python-microscope device server (see device  server documentation: 
https://www.python-microscope.org/examples.html). The device layout is 
as follows:

    device(AdaptiveOpticsDevice, [ip_address], [port],
           {mirror_uri:mirror_args,
           wavefront_uri:wavefront_args,
           slm_uri:slm_args})

Where the various `_args` are lists in the following format:
    
    [microscope_device_name, ip_address, port]
    
Note that all microscope_device_name variables need to be imported from
python-microscope, ip_address variables must be a string and port variables 
must be an int.
