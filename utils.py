import numpy as np
import tensorflow as tf
from PIL import Image
import scipy.ndimage
from scipy import special
import os


##########################################################
# DEFINE PARAMETERS OF SPECKLE AND NORMALIZATION FACTOR
# DEFAULT VALUES 

# M = 10.089038980848645
# m = -1.429329123112601
# L = 1
# c = (1 / 2) * (special.psi(L) - np.log(L))
# cn = c / (M - m)  # normalized (0,1) mean of log speckle

##########################################################

M = 11
m = -5
L = 1
# c = (1 / 2) * (special.psi(L) - np.log(L))
# cn = c / (M - m)  # normalized (0,1) mean of log speckle

def normalize_sar(im):
    return ((np.log(im + np.spacing(1)) - m) * 255 / (M - m)).astype('float32')

def denormalize_sar(im):
    return np.exp((M - m) * np.clip((np.squeeze(im)).astype('float32'),0,1) + m)

def load_sar_images(filelist):
    if not isinstance(filelist, list):
        im = np.load(filelist)
        im = normalize_sar(im)
        return np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1)
    data = []
    for file in filelist:
        im = np.load(file)
        im = normalize_sar(im)
        data.append(np.array(im).reshape(1, np.size(im, 0), np.size(im, 1), 1))
    return data

def store_data_and_plot(im, threshold, filename):
    im = np.clip(im, 0, threshold)
    im = im / threshold * 255
    im = Image.fromarray(im.astype('float64')).convert('L')
    im.save(filename.replace('npy','tif'), compression='tiff_lzw')
    print("    [*] Dumped " + filename.replace('npy','tif'))


def save_sar_images(denoised, noisy, imagename, save_dir, save_npy):
    threshold = np.mean(noisy)+3*np.std(noisy)

    denoisedfilename = os.path.join(os.path.join(save_dir, 'denoised'), imagename)
    if(save_npy):
        np.save(denoisedfilename, denoised)
        print("    [*] Dumped "+ denoisedfilename)
    store_data_and_plot(denoised, threshold, denoisedfilename)

    noisyfilename = os.path.join(os.path.join(save_dir, 'noisy'), imagename)
    if(save_npy):
        np.save(noisyfilename, noisy)
        print("    [*] Dumped "+ noisyfilename)
    store_data_and_plot(noisy, threshold, noisyfilename)
