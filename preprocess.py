from PIL import Image, UnidentifiedImageError
from glob import glob
from pathlib import Path
import numpy as np
import os
import argparse
import time

parser = argparse.ArgumentParser(description='Script to Preprocess Image files')
parser.add_argument('--img_dir', dest='img_dir', default=os.path.join(os.getcwd(),'input_img'), help='Path to the images to be preprocessed')
parser.add_argument('--data_dir', dest='data_dir', default=os.path.join(os.getcwd(),'data'), help='Path where the ouput is to be dumped')
parser.add_argument('--single', dest='single', help='Path where the file to be preprocessed is')
args = parser.parse_args()

def preprocess(files):
    print(f'    [] Overridding PILs DecompressionBombError')
    Image.MAX_IMAGE_PIXELS = None  # Override PIL's DecompressionBombError
    dump_loc = args.data_dir
    print(f'    [] Found {len(files)} files')
    
    for idx in range(len(files)):
        timer_start = time.time()
        print(f'  [*] Processing file {files[idx]}')
        try:
            img = np.asarray(Image.open(files[idx]).convert('L'))
        except UnidentifiedImageError:
            print('   [*] Not an Image SKIPPING')
            continue
        
        #threshold = np.mean(img) + 3*np.std(img)

        #img = img / threshold * 255
        #img = np.clip(img, 0, threshold)
        
        tmp_dir = os.path.join(dump_loc, Path(files[idx]).stem + '.npy')
        np.save(tmp_dir, img)
        timer_end = time.time()
        print(f'   [*] npy filed dumped at {tmp_dir}')
        print('   [*] process completed in %.2fs' % (timer_end-timer_start))

if __name__ == '__main__':
    print(f'[*] Starting Preprocssing')
    if args.single:
        print(f'    [] Processing {args.single}')
        test_files = glob(args.single)
    else:
        print(f'    [] Processing files from {args.img_dir}')
        test_files = glob((args.img_dir+'/*').format('float32'))
    preprocess(test_files)
    print(f'[*] Script Succeeded')
    
    