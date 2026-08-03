Derek Groenendyk
4/30/2012
Notes on files for running HYDRUS in batch mode and other associated functions.

Notes:
You will see that the folder is broken in two different sets of files.  They each
represent a specific project.  The main files in each are generally the same. 
They might have slightly varying implementations but are both based off of the 
same framework of code/files. So below I have commented in general on each of 
the files.

** HYDRUSmain.py **************************************************************
This file simply is the "main" program controlling and running the batch of 
simulations.  You can run in batch mode or single mode depending on your 
preference.  In order to run it loads the various classes to access the
different objects and methods associated with the respective file or program.

The implementation of a class, essentially creates an object and with that 
object each class provides methods upon which functions, routines, etc. can be
called.


** HYDRUS_Class.py ************************************************************
HYDRUS can be called in a batch mode by simply calling hd1calc.exe, the actual
program responsible for performing the calculations, while Hydrus1D.exe is just
the GUI provided for manipulating and calling hd1calc.exe.  When called in 
batch mode hd1calc.exe looks for a file called LEVEL_01.DIR to provide the 
experiment directory.  However this can be completed avoided if you provide an 
argument with the respective location of the experiment you wish to run.

HYDRUS_Class.py is a file that defines and creates the class HYDRUS.  The class
is basically an object that knows things about itself or has intrinsic values.  
The class also has methods and functions that it knows.  These can do various 
things depending on the implemenation of the class.  Generally speaking they 
are to either get or set the data or information of the class.

You can see in this file that the class knows its experiment directory, the 
name of the experiment as well as a list of corresponding file extentions.  It
can also run the HYDRUS program, update the LEVEL_01.DIR, and print out results
from a simulation.


** IN_Class.py/OUT_Class.py ***************************************************
These files are used to create object for each of the respective type of files.
In each file there are multiple classes representing each of the different 
files that are of interest for either inputing or outputting data in HYDRUS.

Notice again the structure of the class and their corresponding methods.
You can look at the HYDRUSmain.py file for implementation of these classes.
