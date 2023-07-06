# Processing pipeline with drift correction for an experiment with 5 visual stimuli
A set of notebooks customized for a numerosity experiment with 5 visual stimuli : 1, 2, 3, 4 and 5 dots on the screen , separated by a blank screen.

## !NOTE: 
These notebooks use [ANTsPy library](https://github.com/ANTsX/ANTsPy) for drift correction. 
ANTsPy only works on Mac and Linux,  If you are on Windows, 
you will need to create an enviromens on WSL (Windows Subsystem for Linux) 
and install both numan and ants into the came conda enviroment on WSL.
You will be able to choose this enviroment as your kernel from Jupyter Notebook on Windows.

After you have installed WSL, follow the instructions here:

Requires numan version 1.0.9, and vodex version 1.0.19
upgrade the packages if you have older versions with :
```
pip install --upgrade vodex
pip install --upgrade numan
```
Or, even better, create a new virtual enviroment and install numan there.
If you are using conda:
```
conda create -n numan python=3.10
conda activate numan
pip install numan==1.0.9
```
***You also need to install ANTs separately by running:***
```
pip install antspyx
```

This should install everything you need (it will get vodex and any other packages).


Run these notebooks in order, read the comments :)

Notebooks 01, 02, 03 do the preprocessing of the movies and take a pretty long time to run .. the rest of the notebooks extract the signals and produce nice plots and cell masks. 
Notebook 09 explores the effect of the covariates.

### You will need to provide the csv files with all the annotations and an offset image from camera calibration. 
