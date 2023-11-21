# Photo-Colour-Correction
Python script for systematically colour-correcting photographs of plastic debris. using reference card

This script imports images that contain the X-Rite/Calibrite colour reference card and uses it to determine the necessary colour correction to bring RGB values in line with their 'true' values, independent of lighting conditions.

Written by Dr Joseph Razzell Hollis on 2023-07-24. Details and assessment of the underlying methodology were published by Razzell Hollis et al. in the journal Methods in Ecology and Evolution in 2023 (DOI: TBC). Please cite the methods paper if you use this code in your analysis.

Any updates to this script will be made available online at www.github.com/Jobium/Photo-Colour-Correction/

# Before starting:
ensure you have installed Python 3.7 (or higher) and the following packages:
    os
    glob
    numpy
    lmfit
    colour
    matplotlib
    PIL
    scipy
    skimage
guidance on installing python and packages can be found online at LINK

# Instructions:
1) ensure that your photos have the colour reference card in a consistent position and orientation
    - photos with very different positions will need to be processed separately
2) set the variable "Photo_dir" to the directory of the photo/s to be imported. For multiple photos, use '*' as a wildcard
    - the script can only accept the following image formats: TIFF, JPG
3) set the variable "ccard_coords" to define a rectangle (in pixel coordinates) that only contains the colour card and nothing else
    - if any of the reference colour squares are not inside this box, the script will fail to detect them properly and produce an inaccurate colour correction
    - the presence of other objects in this box may interfere with the card detection and produce an inaccurate colour correction
4) set the variable "Orientation" to either horizontal or vertical to match the orientation of the colour card in the photos
5) set the variable "Figure_dir" to define the folder where colour correction summary figures should be saved
6) set the variable "Output_dir" to define the folder where colour-corrected photos should be saved
7) run the script in your Python environment
8) check the colour correction worked by looking at the corrected image, and the summary figures

# Notes:
1) colour reference values are derived from the file "ColorChecker_sRGB_from_Lab_D50_AfterNov2014.tiff", an X-Rite/Calibrite reference image post-2014 that was downloaded from www.babelcolor.com/colorchecker-2 on 2023-03-03
2) the script will automatically duplicate the subfolder structure from "Photo_dir", and will save any output images or figures in the appropriate subfolders.
3) the script can process many photos at once, as long as they are roughly consistent in the position and orientation of the colour card

# Citations:
If you use this script, please cite Razzell Hollis et al., Methods in Ecology & Evolution (2023).
