#pandas-patch


[![Travis-CI Build Status](https://travis-ci.org/ericfourrier/pandas-patch.svg?branch=master)](https://travis-ci.org/ericfourrier/pandas-patch)

##CONTENTS OF THIS FILE


 * Introduction
 * Installation
 * Version 
 * Using This Module
 * Design Decisions
 * Thanks


##INTRODUCTION


This quick pandas monkey patch module provide functions designed for several common task of 
statistician, data scientist, quantitative analyst...

This module has two packages dependencies as pandas numpy 

In this module you have :

 * A README !

 * A example folder to find examples and a ipython notebook

 * A unique package folder pandas_patch with main and utils 

 * A test file (tou can run the test with `$python -m unittest -v test`)

##INSTALLATION


These is a simple monkey-patch of pandas. So it is simply dynamic 
creation of methods for the class Dataframe at runtime.

 1. Clone the project on your local computer.

 2. Run the following command 

 	* `$ python setup.py install`

##VERSION


The current version is 0.1 (early release test version).
The module will be improved over time.


##USING THE MODULE


This code is designed to help you in your python data analysis especially
if you need quick and result writing few lines of code.

For each function you have a small documentation.

For now there are no automated tests so there are maybe bugs, feel free to correct them.

You can copy the function and modify the function for your personal use.


##DESIGN DECISIONS


 * We do not pretend to replace beautiful package as pandas,numpy,scipy
 They offer way more flexibility than the functions of this module. Fell free 
 to explore deeper this packages.


##THANKS 


Thanks to all the creator and contributors of the package we are using.

