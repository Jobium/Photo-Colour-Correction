"""
====================================================

This script imports images that contain the X-Rite/Calibrite colour reference card and uses it to determine the necessary colour correction to bring RGB values in line with their 'true' values, independent of lighting conditions.

Written by Dr Joseph Razzell Hollis on 2023-07-24. Details and assessment of the underlying methodology were published by Razzell Hollis et al. in the journal XXX on DATE (DOI: XXX). Please cite the methods paper if you use this code in your analysis.

Any updates to this script will be made available online at www.github.com/Jobium/XXXXX

Before starting:
# ensure you have installed Python 3.7 (or higher) and the following packages:
    os
    glob
    numpy
    lmfit
    colour
    matplotlib
    PIL
    scipy
    skimage
# guidance on installing python and packages can be found online at LINK

Instructions:
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

Notes:
# colour reference values are derived from the file "ColorChecker_sRGB_from_Lab_D50_AfterNov2014.tiff", an X-Rite/Calibrite reference image post-2014 that was downloaded from www.babelcolor.com/colorchecker-2 on 2023-03-03
# the script will automatically duplicate the subfolder structure from "Photo_dir", and will save any output images or figures in the appropriate subfolders.
# the script can process many photos at once, as long as they are roughly consistent in the position and orientation of the colour card

Citations:
# if you use this script, please cite Razzell Hollis et al. (2023), JOURNAL, INFO. DOI: XXX

====================================================
"""

# ==================================================
# import necessary Python packages

import os                       # for creating output folders
import glob                     # for handling file directories
import numpy as np              # for array manipulation
import lmfit as lmfit           # for fitting colour card img with grid  
import colour as Colour         # for colour correction analysis
import matplotlib.pyplot as plt # for plotting results

from PIL import Image           # for importing images
from scipy import stats         # for doing linear least squares regression
from skimage import color       # for conversion betweeen colour spaces
from skimage import filters, measure, morphology     # for detecting colour card in image

# ==================================================
# user-defined variables

# directory of folder containing images to be imported and processed
Photo_dir = './photos/**/*.tif'     # use '*' as a wildcard, or '**' for multiple layers of subfolders

# define the area of the image (in pixels) that should contains the colour card and ideally nothing else
ccard_coords = [1600, 3500, 2200, -1]   # format: [left, right, top, bottom], -1 == far edge of image

# define orientation of the colour card ('horizontal', or 'vertical')
Orientation = 'horizontal'

# parameters for colour correction fitting
Correction_method = 'Vandermonde'   # choose from 'Cheung 2004', 'Finlayson 2015', 'Vandermonde' (recommended: Vandermonde)
Degree = 3                          # polynomial order (recommended: 3)

# directory for folder where summary figures will be saved
Figure_dir = './colour_correction_summaries/'

# directory for folder where processed images will be saved
Output_dir = './colour_corrected_images/'

# the script can also output a summary figure of colour reproduction errors between photos
Plot_errors = True                 # True or False

"""
# ==================================================
# define functions for handling images
# ==================================================
"""

def f_grid(params, length, height):
    # function for generating the x,y coords for a uniform grid of given length, height from fitted parameters
    x_ind, y_ind = np.mgrid[:height, :length]
    x_coords = x_ind * params['x_spacing']
    y_coords = y_ind * params['y_spacing']
    x_rotated = params['x_0'] + np.cos(params['angle'])*x_coords - np.sin(params['angle'])*y_coords
    y_rotated = params['y_0'] + np.sin(params['angle'])*x_coords + np.cos(params['angle'])*y_coords
    return np.array([x_rotated, y_rotated])

def fit_grid(params, length, height, input_coords):
    # function for getting distances between ideal grid and actual points
    grid = f_grid(params, length, height)
    nearest_dists = []
    for i in range(0, np.size(input_coords, axis=1)):
        dists = np.sqrt((input_coords[0,i]-grid[0])**2 + (input_coords[1,i]-grid[1])**2)
        nearest_dists.append(np.amin(dists))
    return nearest_dists
    
def fit_grid_script(input_coords, length=6, height=4, debug=False, plot=False):
    # function for fitting an incomplete set of points using a grid, returns fitted grid coords
    if np.size(input_coords, axis=0) != 2:
        # makes sure input coords are correctly arranged
        input_coords = input_coords.transpose()
    # set up initial parameters for grid
    if debug == True:
        print("input coords shape:", np.shape(input_coords))
        print("x steps:", height)
        print("x min/max: %0.f / %0.f" % (np.amin(input_coords[0]), np.amax(input_coords[0])))
        print("x spacing: %0.f" % ((np.amax(input_coords[0])-np.amin(input_coords[0]))/(height-1)))
        print("y steps:", length)
        print("y min/max: %0.f / %0.f" % (np.amin(input_coords[1]), np.amax(input_coords[1])))
        print("y spacing: %0.f" % ((np.amax(input_coords[1])-np.amin(input_coords[1]))/(length-1)))
    params = lmfit.Parameters()
    params.add('angle', value=0., min=-0.1, max=0.1)
    params.add('x_0', value=np.amin(input_coords[0]))
    params.add('y_0', value=np.amin(input_coords[1]))
    params.add('x_spacing', value=(np.amax(input_coords[0])-np.amin(input_coords[0]))/(height-1), min=0)
    params.add('y_spacing', value=(np.amax(input_coords[1])-np.amin(input_coords[1]))/(length-1), min=0)
    if debug == True:
        print("        initial parameters:")
        print(params.pretty_print())
    grid = f_grid(params, length, height)
    if plot == True:
        plt.scatter(grid[1], grid[0], label='init')
    # run the fitting algorithm
    fit_output = lmfit.minimize(fit_grid, params, args=(length, height, input_coords))
    if debug == True:
        print("        fitted parameters: ")
        print(fit_output.params.pretty_print())
    fitted_coords = f_grid(fit_output.params, length, height)
    if debug == True:
        print(np.shape(fitted_coords))
    return np.reshape(fitted_coords, (2, -1)), fit_output.params

def get_colour_grid(img, length=6, height=4, debug=False, plot=False):
    # function for analysing a colour card image and returning average RGB values from the 24 reference squares]
    # compress image to grayscale and use Otsu thresholding to separate objects from background based on intensity
    threshold = filters.threshold_otsu(np.mean(img, axis=2))
    mask = np.mean(img, axis=2) > threshold
    # remove any small objects and holes
    mask = morphology.remove_small_holes(mask, 50)
    mask = morphology.remove_small_objects(mask, 20000)
    if plot == True:
        plt.imshow(img)
        plt.show()
        plt.imshow(mask)
    # find objects and measure their properties
    labels = measure.label(mask)
    props = measure.regionprops(labels)
    mean_coords = []
    if debug == True:
        print("squares found:", np.amax(labels), "/24") # not every square will be detected, this is normal
    for index in range(0, np.amax(labels)):
        # iterate over objects, record mean position
        label_i = props[index].label
        coords = measure.find_contours(labels == label_i, 0.5)[0]
        mean_coords.append(np.mean(coords, axis=0))
        if debug == True:
            print("object %s: %4.f coords, area=%5.f px" % (index, np.size(coords, axis=0), props[index].area))
        if plot == True:
            plt.plot(coords[:,1], coords[:,0])
    # because not every square will be detected, we need to figure out where the missing squares should be
    # create a complete grid and fit to mean positions of detected squares
    mean_coords = np.asarray(mean_coords)
    fitted_coords, fitted_params = fit_grid_script(mean_coords, length, height, debug=debug, plot=plot)
    fitted_coords = fitted_coords.transpose()
    if plot == True:
        plt.scatter(mean_coords[:,1], mean_coords[:,0], label='centers')
        plt.scatter(fitted_coords[:,1], fitted_coords[:,0], label='fitted')
        plt.legend()
        plt.show()
    if debug == True:
        print("fitted grid coords:", np.shape(fitted_coords))
    # get mean colour for each grid point
    mean_colours = []
    temp = np.zeros_like(img, dtype=float)
    x_grid, y_grid = np.mgrid[:np.size(img, axis=0), :np.size(img, axis=1)]
    rad = (fitted_params['x_spacing']+fitted_params['y_spacing'])/8
    for i in range(0, np.size(fitted_coords, axis=0)):
        # iterate over grid points, mask img and find mean colour using circular mask
        mask = np.sqrt((x_grid - fitted_coords[i,0])**2 + (y_grid - fitted_coords[i,1])**2) <= rad
        mean_colours.append(np.mean(img[mask], axis=0)/255)
        if debug == True:
            print("grid point %s mean colour:" % i, mean_colours[-1])
        if plot == True:
            t = np.linspace(-np.pi, np.pi, 100)
            cx = fitted_coords[i,0] + rad*np.cos(t)
            cy = fitted_coords[i,1] + rad*np.sin(t)
            plt.plot(cy, cx, 'w:')
            plt.plot(fitted_coords[i,1], fitted_coords[i,0], mfc=mean_colours[-1], marker='o', mec='w', markersize=15)
            plt.text(fitted_coords[i,1]+rad/3, fitted_coords[i,0]+rad/3, i)
        temp[mask] = mean_colours[-1]
    mean_colours = np.asarray(mean_colours)
    fitted_coords = np.asarray(fitted_coords)
    if plot == True:
        plt.imshow(img)
        plt.show()
    labs = color.rgb2lab(mean_colours[:,:3], illuminant='D50')
    if np.argmin(labs[:,0]) < 6 and np.argmax(labs[:,0]) < 6:
        # card is rotated 180 degrees, return values in reverse order to get correct sequence
        if debug == True:
            print("reversing order...")
        return fitted_params, mean_colours[::-1,:], fitted_coords[::-1,:]
    else:
        return fitted_params, mean_colours, fitted_coords

"""
# ==================================================
import colour card reference image
# ==================================================
"""

print()
print("STARTING SCRIPT")

print()
print()
print("importing reference colour card values from reference image...")

# import reference image of ColorChecker
ref_img = plt.imread("./ColorChecker_sRGB_from_Lab_D50_AfterNov2014.tif")

# get colours from card squares, convert to L*a*b*
fit_params, ref_rgbs, fitted_coords = get_colour_grid(ref_img, debug=False, plot=False)
ref_labs = color.rgb2lab(ref_rgbs)

# find squares with lowest, highest L*
black_i = np.argmin(ref_labs[:,0])
white_i = np.argmax(ref_labs[:,0])

if black_i < 6 and white_i < 6:
    # rotate card to line up with ref values
    ref_rgbs = np.flip(ref_rgbs, axis=0)
    ref_labs = np.flip(ref_labs, axis=0)
    
print()
print("ref RGB/LAB arrays from reference image:", np.shape(ref_rgbs), np.shape(ref_labs))

"""
# ==================================================
# set up data storage dicts
# ==================================================
"""

img_data = {
    'file_name': [],
    'file_format': [],
    'subfolders': [],
    'metadata': [],
    'outdir': [],
    'figdir': [],
    'full_img': [],
    'ccard_img': [],
    'ccard_coords': [],
    'ccard_rgb': [],
    'ccard_lab': [],
    'full_img_corr': [],
    'ccard_img_corr': [],
    'ccard_rgb_corr': [],
    'ccard_lab_corr': []
}

"""
# ==================================================
# import photos to be colour corrected
# ==================================================
"""

print()
print()
print("finding photos for import...")

img_dirs = sorted(glob.glob(Photo_dir, recursive=True))

print()
print("photos found:", len(img_dirs))

for img_dir in img_dirs:
    while True:
        try:
            file_name = img_dir.split("/")[-1]
            subfolders = "/".join(img_dir.split("/")[2:-1])
            print()
            print(f"importing {file_name}...")
            print(f"    {img_dir}")
            
            # import image (default data format: (y, x, c), dtype: integer 0-255)
            img =  Image.open(img_dir)
            file_format = img.format
            print("    imported image shape: ", np.shape(img))
            print("             image format:", file_format)
            
            # check if image is in RGBA format
            if np.size(img, axis=2) > 3:
                # remove A channel if necessary
                img = np.asarray(img)[:,:,:3]
            
            # trim image to area containing colour card based on "ccard_coords"
            ccard_img = np.asarray(img)[ccard_coords[2]:ccard_coords[3], ccard_coords[0]:ccard_coords[1]]
            print("    trimmed image shape:", np.shape(ccard_img))
            
            # if necessary rotate colour card image to horizontal
            if Orientation.lower() in ['vertical', 'vert', 'v', 'portrait']:
                ccard_img = np.rot90(ccard_img, k=1, axes=(0, 1))
                print("        rotating image to be horizontal. new shape:", np.shape(ccard_img))
            
            # get mean colours from card squares
            fit_params, card_rgbs, fitted_coords = get_colour_grid(ccard_img, debug=False, plot=False)
            
            # rescale RGB values to 0-1 and then convert to L*a*b* colour space
            temp = np.copy(card_rgbs)
            if np.amax(temp) > 1:
                temp /= 255
            card_labs = color.rgb2lab(temp, illuminant='D50')
            print(f"    colour card L* range: {np.amin(card_labs):0.1f} - {np.amax(card_labs):0.1f}")
            
            # find squares with lowest, highest L* (brightness)
            black_i = np.argmin(card_labs[:,0])
            white_i = np.argmax(card_labs[:,0])
            
            # rearrange into grid, extract greyscale square L* values
            grid_Ls = np.reshape(card_labs[:,0], (4,6))
            greyscale_Ls = grid_Ls[3,:]
            
            if black_i < 6 and white_i < 6:
                # rotate card to line up with ref values (greyscale squares should be 18-23)
                card_rgbs = np.flip(card_rgbs, axis=0)
                card_labs = np.flip(card_labs, axis=0)
                black_i = np.argmin(card_labs[:,0])
                white_i = np.argmax(card_labs[:,0])
                grid_Ls = np.rot90(grid_Ls, k=2)
                greyscale_Ls = grid_Ls[3,:]
                
            # create directory for corrected image
            out_dir = "%s%s/" % (Output_dir, subfolders)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            # create directory for summary figure
            fig_dir = "%s%s/" % (Figure_dir, subfolders)
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
                
            # save isolated colour card image to output folder
            im = Image.fromarray(ccard_img, mode="RGB")
            im.save('%s%s_colourcard_corr.%s' % (fig_dir, file_name, file_format))
            
            # add to data storage array
            img_data['file_name'].append(file_name)
            img_data['file_format'].append(file_format)
            img_data['subfolders'].append(subfolders)
            img_data['figdir'].append(fig_dir)
            img_data['outdir'].append(out_dir)
            img_data['full_img'].append(img)
            img_data['ccard_img'].append(ccard_img)
            img_data['ccard_coords'].append(fitted_coords)
            img_data['ccard_rgb'].append(card_rgbs)
            img_data['ccard_lab'].append(card_labs)
                
            print("successfully imported!")
            break
        except Exception as e:
            # if anything goes wrong, print exception error and move on to next image
            print("something went wrong! Exception:", e)
            break

"""
# ==================================================
# determine and apply colour correction
# ==================================================
"""

print()
print()
print("calculating transformation needed to correct colour values...")

for i in range(len(img_data['file_name'])):
    print()
    print(f"correcting image {img_data['file_name'][i]}")
    
    # for each image, take the colour card RGB values, the colour card image, and the full image and do colour correction to each
    for img_subset in ['ccard_rgb', 'ccard_img', 'full_img']:
        # take the RGB values reshape the array to 2D for colour correction transform
        temp = np.reshape(img_data[img_subset][i], (-1,3))

        # reduce data from 0-255 range to 0-1
        if np.amax(temp) > 1:
            temp = np.asarray(temp)/255.

        # determine transform needed to align RGB values with ref values
        temp_corr = Colour.colour_correction(temp, img_data['ccard_rgb'][i], ref_rgbs, method=Correction_method, degree=Degree)

        # report results
        print("    %s values before/after correction:" % img_subset)
        print("        temp array shapes:", np.shape(temp), np.shape(temp_corr))
        print("        R range: %0.2f - %0.2f -> %0.2f - %0.2f" % (np.amin(temp[:,0]), np.amax(temp[:,0]), np.amin(temp_corr[:,0]), np.amax(temp_corr[:,0])))
        print("        G range: %0.2f - %0.2f -> %0.2f - %0.2f" % (np.amin(temp[:,1]), np.amax(temp[:,1]), np.amin(temp_corr[:,1]), np.amax(temp_corr[:,1])))
        print("        B range: %0.2f - %0.2f -> %0.2f - %0.2f" % (np.amin(temp[:,2]), np.amax(temp[:,2]), np.amin(temp_corr[:,2]), np.amax(temp_corr[:,2])))
        print("        %3.1f%% of corrected values out of 0-1 range" % (100*np.count_nonzero(np.logical_or(temp_corr < 0, temp_corr > 1))/np.size(temp_corr)))

        # trim out-of-bounds values (<0, >1)
        temp_corr[temp_corr > 1.] = 1.
        temp_corr[temp_corr < 0.] = 0.

        # return value array to its original shape and add to img_data array
        img_data[img_subset+"_corr"].append(np.reshape(temp_corr, np.shape(img_data[img_subset][i])))
        
        if img_subset == 'ccard_rgb':
            # convert corrected RGB values to L*a*b* space
            temp_corr_lab = color.rgb2lab(temp_corr, illuminant='D50')
            img_data['ccard_lab_corr'].append(np.reshape(temp_corr_lab, np.shape(img_data[img_subset][i])))
        
        
        
    # convert corrected version of colour card image to 0-255 format and save to figure folder
    temp = np.copy(img_data['ccard_img_corr'][i])
    if np.amax(temp) <= 1:
        temp *= 255
    temp = np.asarray(temp, dtype=np.uint8)
    im = Image.fromarray(temp, mode="RGB")
    im.save('%s%s_colourcard_corr.%s' % (img_data['figdir'][i], img_data['file_name'][i], img_data['file_format'][i]))
    
    # convert corrected version of full image to 0-255 format and save to figure folder
    temp = np.copy(img_data['full_img_corr'][i])
    if np.amax(temp) <= 1:
        temp *= 255
    temp = np.asarray(temp, dtype=np.uint8)
    im = Image.fromarray(temp, mode="RGB")
    im.save('%s%s_corr.%s' % (img_data['outdir'][i], img_data['file_name'][i], img_data['file_format'][i]))
    
    
    
    # plot summary figure for colour correction
    plt.figure(figsize=(12,8))
    plt.suptitle(f'Colour correction of {file_name}')
    
    # ax1: original colour card image with overlaid spots showing ref colours
    ax1 = plt.subplot2grid((2,6), (0,0), colspan=3)
    ax1.imshow(img_data['ccard_img'][i])
    ax1.scatter(img_data['ccard_coords'][i][:,1], img_data['ccard_coords'][i][:,0], c=ref_rgbs, s=256, edgecolors='k')
    ax1.text(0.5, 0.9, "Original Image", color='w', ha='center', transform = ax1.transAxes)
    ax1.text(0.05, 0.1, "Circles show reference colours", color='w', transform = ax1.transAxes)
    
    # ax2: corrected colour card image with overlaid spots showing ref colours
    ax2 = plt.subplot2grid((2,6), (0,3), colspan=3)
    ax2.imshow(img_data['ccard_img_corr'][i])
    ax2.scatter(img_data['ccard_coords'][i][:,1], img_data['ccard_coords'][i][:,0], c=ref_rgbs, s=256, edgecolors='k')
    ax2.text(0.5, 0.9, "Corrected Image", color='w', ha='center', transform = ax2.transAxes)
    
    # ax3: scatter plot of brightness (L*) values for colour card squares
    ax3 = plt.subplot2grid((2,6), (1,0), colspan=2)
    ax3.set_title("L* values")
    ax3.set_xlabel("Square #")
    ax3.set_ylabel("CIE L*")
    ax3.scatter(range(24), img_data['ccard_lab'][i][:,0], c=img_data['ccard_rgb'][i], marker='o', zorder=2, label='orig.')
    ax3.scatter(range(24), ref_labs[:,0], c='w', edgecolors=ref_rgbs, marker='D', zorder=1, label='ref.')
    ax3.scatter(range(24), img_data['ccard_lab_corr'][i][:,0], marker='o', c='w', edgecolors=img_data['ccard_rgb_corr'][i], label='corr.')
    
    # ax4: scatter plot of colour (a*, b*) values for colour card squares
    ax4 = plt.subplot2grid((2,6), (1,2), colspan=2)
    ax4.set_title("a* , b* values")
    ax4.set_xlabel("CIE a*")
    ax4.set_ylabel("CIE b*")
    ax4.scatter(img_data['ccard_lab'][i][:,1], img_data['ccard_lab'][i][:,2], c=img_data['ccard_rgb'][i], marker='o', zorder=2)
    ax4.scatter(ref_labs[:,1], ref_labs[:,2], c='w', edgecolors=ref_rgbs, marker='D', zorder=1)
    ax4.scatter(img_data['ccard_lab_corr'][i][:,1], img_data['ccard_lab_corr'][i][:,2], marker='o', c='w', edgecolors=img_data['ccard_rgb_corr'][i])
    
    # ax5: scatter plot of brightness (L*) values vs reference values
    ax5 = plt.subplot2grid((2,6), (1,4), colspan=2)
    ax5.set_title("Luminance")
    ax5.set_xlabel("Reference L*")
    ax5.set_ylabel("Measured L*")
    x_temp = np.linspace(0,100,10)
    ax5.plot(x_temp, x_temp, 'k:', zorder=1, label='ref')
    ax5.scatter(ref_labs[:,0], img_data['ccard_lab'][i][:,0], c='b', zorder=3, label='orig.')
    ax5.scatter(ref_labs[:,0], img_data['ccard_lab_corr'][i][:,0], c='r', zorder=4, label='corr.')
    result = stats.linregress(ref_labs[:,0], img_data['ccard_lab_corr'][i][:,0])
    y_temp = result.slope * x_temp + result.intercept
    ax5.plot(x_temp, y_temp, 'r:', zorder=2)
    ax5.legend()
    for i2 in range(24):
        ax3.plot([i2, i2, i2], [img_data['ccard_lab'][i][i2,0], ref_labs[i2,0], img_data['ccard_lab_corr'][i][i2,0]], c=ref_rgbs[i2], zorder=0)
        ax4.plot([img_data['ccard_lab'][i][i2,1], ref_labs[i2,1], img_data['ccard_lab_corr'][i][i2,1]], [img_data['ccard_lab'][i][i2,2], ref_labs[i2,2], img_data['ccard_lab_corr'][i][i2,2]], c=ref_rgbs[i2], zorder=0)
    ax3.legend()
    
    # finish figure and save to figure folder
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.tight_layout()
    plt.savefig("%s%s_colour_correction.png" % (img_data['figdir'][i], img_data['file_name'][i]), dpi=300)
    plt.show()
    
"""
# ==================================================
# report colour reproduction errors
# ==================================================
"""

# colour reproduction error is defined as the Euclidean distance between the apparent colour and the true colour in (0-1) RGB space.

if Plot_errors == True:
    print()
    print("calculating colour reproduction errors...")
    
    # convert colour card RGB values to arrays for faster calculation
    img_data['ccard_rgb'] = np.asarray(img_data['ccard_rgb'])
    img_data['ccard_rgb_corr'] = np.asarray(img_data['ccard_rgb_corr'])
    
    # repeat ref value array to match shape of colour card array, for array mathematics
    temp = np.tile(ref_rgbs, (len(img_data['file_name']),1,1))
    
    # calculate Euclidean distance between original colour card RGB coordinates and reference coordinates
    img_data['ccard_rgb_error'] = np.sqrt(np.sum((img_data['ccard_rgb'] - temp)**2, axis=2))
    
    # report average error before correction
    print()
    print(f"average error before correction: {np.mean(img_data['ccard_rgb_error']):0.2f} +/- {np.std(img_data['ccard_rgb_error']):0.2f}")
    
    # calculate Euclidean distance between corrected colour card RGB coordinates and reference coordinates
    img_data['ccard_rgb_corr_error'] = np.sqrt(np.sum((img_data['ccard_rgb_corr'] - temp)**2, axis=2))
    
    # report average error after correction
    print(f"               after correction: {np.mean(img_data['ccard_rgb_corr_error']):0.2f} +/- {np.std(img_data['ccard_rgb_corr_error']):0.2f}")
    
    print()
    # report average error for each photo
    for i in range(len(img_data['file_name'])):
        length = len(img_data['file_name'][i])
        print(f"photo {img_data['file_name'][i]} average error before correction: {np.mean(img_data['ccard_rgb_error'][i]):0.2f} +/- {np.std(img_data['ccard_rgb_error'][i]):0.2f}")
        print(f"      {' '*length}                after correction: {np.mean(img_data['ccard_rgb_corr_error'][i]):0.2f} +/- {np.std(img_data['ccard_rgb_corr_error'][i]):0.2f}")
    
    # create error figure
    plt.figure(figsize=(12,6))
    
    # ax2: boxplot of errors for pre-correction images
    ax1 = plt.subplot2grid((2,6), (0,0), colspan=5)
    ax1.set_title("Colour Reproduction Error of Colour Reference Card")
    ax1.set_ylabel("$\sqrt{(\Delta R)^2 + (\Delta G)^2 + (\Delta B)^2}$")
    ax1.boxplot(img_data['ccard_rgb_error'].transpose())
    ax1.set_xticklabels(range(1, len(img_data['file_name'])+1), rotation=90)
    ax1.text(0.02, 0.9, 'Before correction', transform = ax1.transAxes)
    
    # ax2: boxplot of errors for post-correction images
    ax2 = plt.subplot2grid((2,6), (1,0), colspan=5, sharey=ax1)
    ax2.set_ylabel("$\sqrt{(\Delta R)^2 + (\Delta G)^2 + (\Delta B)^2}$")
    ax2.boxplot(img_data['ccard_rgb_corr_error'].transpose())
    ax2.set_xticklabels(range(1, len(img_data['file_name'])+1), rotation=90)
    ax2.text(0.02, 0.9, 'After correction', transform = ax2.transAxes)
    
    #ax 3: histogram of all pre-corr errors
    ax3 = plt.subplot2grid((2,6), (0,5), sharey=ax1)
    ax3.hist(np.ravel(img_data['ccard_rgb_error']), orientation='horizontal', range=ax1.get_ylim(), bins=20)
    
    #ax 4: histogram of all post-corr errors
    ax4 = plt.subplot2grid((2,6), (1,5), sharey=ax1, sharex=ax3)
    ax4.hist(np.ravel(img_data['ccard_rgb_corr_error']), orientation='horizontal', range=ax2.get_ylim(), bins=20)
    
    # finish and save figure
    plt.tight_layout()
    plt.savefig("%sRGB_errors.png" % (Figure_dir), dpi=300)
    plt.show()
    
print()
print()
print("SCRIPT FINISHED")