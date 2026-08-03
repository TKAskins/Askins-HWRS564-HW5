

HYDRUS is run using HYDRUSmain_basic.py
There are several options/methods on what kind of simulations to run.
method 1: runs a single simulation
method 2: runs multiple simulations, changing basic parameters
method 3: runs multiple simulations, but is more complex changing soil parameters, initial profile heads, etc.

You will mainly use method 3.

results will be saved in the results_dir and results_folder. The method currently saves the water content, heads, and flux for all nodes and times in the NODINF file.

soil textures are retrieved by calling the getAllTexParams() function.
prct: specify the sand, silt, clay increment either 'one' or 'two',
model: which ROSETTA model 'old' or 'new'
lmean: provides the mean of ROSETTA soil hydraulic paramters
llog: log of the hydraulic conductivities

The wc object (using the WC() constructor/class) is for setting the heads in profile.dat based on the soil texture.

Boundary conditions can be changed using the atmo_in object (using the ATMOSPHIN constructor/class)
Times can be added using addLines and specifying the times, the conditions can be set/changed using setData
Any propery/parameter value can be changed using setData including times
Times can be removed by specifying the number of times to delete.

Profile values are modified using the profile object (using the PROFILE constructor/class)
Add layers using addLayers and specifying the new depths and heads (either as a constant or list)
Layers are removed by specifying the number to removed
Parameters and properties are also changed using the setData method

Selector in parameters are changed using the SELECTORIN constructor/class
This is done the script multiple times by calling the setSelectorParams() function
The function requires that you provide a dictionary of parameter names and values, as before all named parameters can be modified.

HYDRUS itself is run by calling run_hydrus method, the output is then stored using saveOutput() based on the directory and trial num/name provided. The saveOutput method currently saves the water content, heads, and flux for all nodes and times in the NODINF file. You can choose to only save heads and flux by changing the variables get_hdata, get_fdata





