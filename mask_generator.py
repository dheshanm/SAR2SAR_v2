from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.fftpack as fp
from glob import glob
from pathlib import Path
import numpy as np
import cv2 as cv
import argparse
import os
import time

from PIL import Image, UnidentifiedImageError
Image.MAX_IMAGE_PIXELS = None  # Override PIL's DecompressionBombError

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', dest='img_dir', default=os.path.join(os.path.join(os.getcwd(),'output'), 'denoised'), help='Image files that need to be proessed are stored here')
parser.add_argument('--mask_data', dest='mask_data', default=os.path.join(os.path.join(os.getcwd(),'output'), 'mask_data'), help='Output directory to store masks')
parser.add_argument('--single', dest='single', help='Path where the file for which the mask to be generated is')
args = parser.parse_args()

def generate_masks(files):
    print(f'    [] Overridding PILs DecompressionBombError')
    Image.MAX_IMAGE_PIXELS = None  # Override PIL's DecompressionBombError
    dump_loc = args.mask_data
    print(f'    [] Found {len(files)} files')
    
    for idx in range(len(files)):
        timer_start = time.time()
        print(f'  [*] Processing file {files[idx]}')
        try:
            img = imread(files[idx])
        except UnidentifiedImageError:
            print('   [*] Not an Image SKIPPING [UnidentifiedImageError]')
            continue
        except ValueError:
            print(f'   [*] Not an Image SKIPPING []')
            continue
    
        print('   [*] Performing Fourier Transforms...')
        F1 = fp.fft2((img).astype(float))
        F2 = fp.fftshift(F1)

        (w, h) = img.shape
        half_w, half_h = int(w/2), int(h/2)

        n = 100
        F2[half_w-n:half_w+n+1,half_h-n:half_h+n+1] = 0 # select all but the first 100x100 (low) frequencies

        img = fp.ifft2(fp.ifftshift(F2)).real
        img = Image.fromarray(img).convert('P')
        
        filename = os.path.join(args.mask_data, Path(files[idx]).stem + '_mask.png')
        print(f'   [*] Dumping mask at {filename}')
        
        img.save(filename)
        img = cv.imread(filename, 0)
        
        print(f'   [*] Performing Thresholding...')
        ret,img = cv.threshold(img,127,255,cv.THRESH_BINARY)
        img = Image.fromarray(img).convert('P')
        print(f'   [*] Dumping Thresholded mask at {filename}')
        img.save(filename)
        
        img = cv.imread(filename, 0) 
        print(f'   [*] Dialating thresholded mask...')
        # Taking a matrix of size 5 as the kernel 
        print(f'     [*] Generating 5x5 kernel...')
        kernel = np.ones((5,5), np.uint8) 
        print(f'     [*] Dialating...')
        img = cv.dilate(img, kernel, iterations=1) 
        
        print(f'   [*] Dumping Thresholded amd Dialated mask at {filename}')
        img = Image.fromarray(img).convert('P')
        img.save(filename)
        timer_end = time.time()
        print('   [*] process completed in %.2fs' % (timer_end-timer_start))

if __name__ == '__main__':
    print(f'[*] Starting Mask Generation')
    if args.single:
        print(f'    [] Processing {args.single}')
        test_files = glob(args.single)
    else:
        print(f'    [] Processing files from {args.img_dir}')
        test_files = glob((args.img_dir+'/*').format('float32'))
    generate_masks(test_files)
    print(f'[*] Script Succeeded')