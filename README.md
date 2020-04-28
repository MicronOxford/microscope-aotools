# AOTools: A microscope add-on for adaptive optics

A composite device for controlling and using deformable mirror adaptive
optic elements. This composite device can be constructed from any
python-microscope compatible adaptive optics elements and cameras (see 
python-microscope compatibility list) as well as certain spatial light
modulators. The functions presented allow a user to calibrate and 
characterise a adaptive optics elements using a variety of wavefront 
sensing techniques, to set control matrices determined by other   
calibration methods and perform sensorless adaptive optics correction  
with a variety of metrics.

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

**Adding new wavefront sensing techniques:**

The currently implemented wavefront sensing techniques are stored in the
*aoAlg.py* file as functions of the AdaptiveOpticsFunctions class with 
the naming convention *"unwrap_"* Each of these functions, and any new 
functions a user may wish to add, should be of the form:
	
	def unwrap_*new_technique*(self, image, (N key word variables), **kwargs):
		#The actual function definition
		return phase_image
	
There may be no key word variables required. If any are required, they
should be placed between the image key word variable and \*\*kwargs.
The final \*\*kwarg syntax is included to catch any key word variables required 
for other wavefont sensing techniques which higher level processes which
utilise Microscope-AOtools may pass to the functions, depending on how the 
wavefont sensing switching is enabled at those levels. The function should 
output a phase image as a real valued numpy array.

Many wavefront sensing techniques have pre-requisites that, once set, do not
need to be calculated again e.g. the inteferometric wavefront sensing technique 
requires a mask to isolate the phase information in the Fourier transform. Once
this mask has been constructed for one image, it can be used for all subsequent 
wavefront sensing images. These pre-requisites are stored as attributes of the 
AdaptiveOpticsFunctions class. Methods for constructing these pre-requisites
and attributes for storing them should be added to the AdaptiveOpticsFunctions 
class. The attributes should be set as *None* in the *\_\_init\_\_* function.

Once the new wavefont sensing technique has been added to *aoAlg.py* it
should be added to the *unwrap_function* dictionary in *aoDev.py* with an 
appropriate string key i.e. the interferometry wavefront sensing technique
has the string key *"interferometry"*. Checks for the appropriate pre-requisites 
should be added to the *check_unwrap_conditions* function in *aoDev.py*. At 
this point the new wavefont sensing technique can be used in any of the Set-up 
methods in the same manner as the existing wavefont sensing techniques. 

**Adding new image quality metrics:**

The currently implemented image quality assessment techniques are
stored in the *aoMetrics.py* file as functions with the naming 
convention *"measure_"* and have the same input and output forms. Each 
of these functions, and any new functions a user may wish to add, should
be of the form:
	
	def measure_*new_technique*(image, (N key word variables), **kwargs):
		#The actual function definition
		return metric
	
There may be no key word variables required. If any are required, they
should be placed between the image key word variable and \*\*kwargs.
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
