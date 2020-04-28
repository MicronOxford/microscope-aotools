# AOTools: A microscope add-on for adaptive optics

A composite device for controlling and using deformable mirror adaptive
optic elements. This composite device can be constructed from any
python-microscope compatible deformable mirrors and cameras (see 
python-microscope compatibility list) as well as certain spatial light
modulators. The functions presented allow a user to calibrate and 
characterise a deformable mirror using an interferometric wavefront 
sensor, to set control matrices determined by other calibration methods 
and perform sensorless adaptive optics correction with a variety of metrics.

**Set-up:**

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

**Adding new image quality metrics:**

The list of currently implemented image quality assessment techniques are
stored in the *aoMetrics.py* file. They are all functions of the naming 
convention *"measure_"* and have the same input and output forms. The each 
of these functions, and any new functions a user may wish to add, expects 
3 inputs:
	
* An image to assess the quality of
* Any key word variable, if any, required for the specific image quality assessment technique
* The \*\*kwargs syntax
	
The final \*\*kwarg syntax is included to catch any key word variables required 
for other image quality assessment techniques which higher level processes 
which utilise Microscope-AOtools may pass to the functions, depending on how 
the metric switching is enabled at those levels. The function should output a 
single value, which can be an integer or a floating point number. 
	
Once the new image quality assessment technique has been added to *aoMetrics.py* 
it should be added to the *metric_function* dictionary in *aoAlg.py* with an 
appropriate string key i.e. the contrast image quality metric has the string key 
*"contrast"*. At this point the new image quality metric can be used in any of 
the Use Case methods in the same manner as the existing image quality assessment 
techniques. 
