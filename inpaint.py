from glob import glob
from pathlib import Path
import numpy as np
import cv2
import argparse
import os
import time
import sys
import tensorflow as tf
from skimage.measure import compare_ssim as ssim
from PIL import Image, UnidentifiedImageError

parser = argparse.ArgumentParser(description='')
parser.add_argument('--img_dir', dest='img_dir', default=os.path.join(os.path.join(os.getcwd(),'test'), 'denoised'), help='Image files that need to be proessed are stored here')
parser.add_argument('--mask_dir', dest='mask_dir', default=os.path.join(os.getcwd(),'mask_data'), help='Output directory to get masks')
parser.add_argument('--out_dir', dest='out_dir', default=os.path.join(os.path.join(os.getcwd(),'test'), 'final'), help='Inpainted images are to be dumbed here')
parser.add_argument('--debug', dest='debug', help='Image-by-Image mode')
args = parser.parse_args()

def make_kernel(a):
  """Transform a 2D array into a convolution kernel"""
  a = np.asarray(a)
  a = a.reshape(list(a.shape) + [1,1])
  return tf.constant(a, dtype=1)

def heat_conv(input, kernel):
  """A simplified 2D convolution operation for Heat Equation"""
  input = tf.expand_dims(tf.expand_dims(input, 0), -1)

  result = tf.nn.depthwise_conv2d(input=input, filter=kernel,
                                    strides=[1, 1, 1, 1],
                                    padding='SAME')

  return result[0, :, :, 0]

def show_viz(i,original, masked, mask, inpainted):
    """Show Image using matplotlib"""
    plt.figure(i)
    plt.subplot(221), plt.imshow(original, 'gray')
    plt.title('original image')
    plt.subplot(222), plt.imshow(masked, 'gray')
    plt.title('source image')
    plt.subplot(223), plt.imshow(mask, 'gray')
    plt.title('mask image')
    plt.subplot(224), plt.imshow(inpainted, 'gray')
    plt.title('inpaint result')

    plt.tight_layout()
    plt.draw()
    
def show_ssim(original, masked, inpainted):
    """Show SSIM Difference"""
    print("SSIM : ")
    print("  Original vs. Original  : ", ssim(original,original))
    print("  Original vs. Masked    : ", ssim(original,masked))
    print("  Original vs. Inpainted : ", ssim(original,inpainted))
    
def inpaint(masked, mask):
    # Init variable
    N = 2000
    ROOT_DIR = os.getcwd()

    # Create variables for simulation state
    U = tf.Variable(masked)
    print("      [] Created tf variable 'U'")
    G = tf.Variable(masked)
    print("      [] Created tf variable 'G'")
    M = tf.Variable(np.multiply(mask,1))
    print("      [] Created tf variable 'M'")
    
    print("      [] Constructing kernel....")
    K = make_kernel([[0.0, 1.0, 0.0],
                     [1.0, -4., 1.0],
                     [0.0, 1.0, 0.0]])
    print("      [] Created kernel")

    dt = tf.compat.v1.placeholder(tf.float16, shape=())

    """Discretized PDE update rules"""
    """u[i,j] = u[i,j] + dt * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) - dt * lambda_m[i,j]*(u[i,j]-g[i,j])"""

    #Tensorflow while_loop function, iterate the PDE N times.
    index_summation = (tf.constant(1), U, M, G, K)
    def condition(i, U, M, G, K):
        return tf.less(i, 100)

    def body(i,U,M,G,K):
        U_ = U + 0.1 * heat_conv(U,K) - 0.1 * M * (U-G)
        return tf.add(i, 1), U_, M, G, K

    #Tensorflow Session
    with tf.compat.v1.Session():
        # Initialize state to initial conditions
        tf.compat.v1.global_variables_initializer().run()

        #Run PDE using tensorflow while_loop
        t = time.time()
        uf=tf.while_loop(cond=condition, body=body, loop_vars=index_summation)[1]
        U = uf.eval()

    print("      [] Execution Time : {} s".format(time.time()-t))

    return U

def container(files):
    print(f'    [] Overridding PILs DecompressionBombError')
    Image.MAX_IMAGE_PIXELS = None  # Override PIL's DecompressionBombError
    tf.compat.v1.disable_eager_execution()
    
    dump_loc = args.out_dir
    print(f'    [] Found {len(files)} files')
    
    for idx in range(len(files)):
        timer_start = time.time()
        print(f'  [*] Processing file {files[idx]}')
        try:
#             original = cv2.imread(os.path.join(IMG_DIR, 'image{}_ori.png'.format(1)),0)
            masked = cv2.imread(files[idx],0)
            mask = cv2.imread(os.path.join(args.mask_dir, Path(files[idx]).stem + "_mask.png"),0)
        except UnidentifiedImageError:
            print('   [*] Not an Image SKIPPING')
            continue
        
        print(f'    [*] Normalizing image')
        masked = cv2.normalize(masked, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        print(f'    [*] Normalizing mask')
        mask = 1-cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        print(f'    [*] Attempting to inpaint')
        inpainted = inpaint(masked,mask)
        
        
        filename = os.path.join(dump_loc, Path(files[idx]).stem + "_inpainted.png")
        print(f'      [*] Dumping inpainted image to {filename}')
        cv2.imwrite(filename, inpainted*255)
        
def singleMode():
    print(f'    [] Overridding PILs DecompressionBombError')
    Image.MAX_IMAGE_PIXELS = None  # Override PIL's DecompressionBombError
    tf.compat.v1.disable_eager_execution()
    
    timer_start = time.time()
    print(f'  [*] Processing file {args.img_dir}')
    try:
#             original = cv2.imread(os.path.join(IMG_DIR, 'image{}_ori.png'.format(1)),0)
        masked = cv2.imread(args.img_dir,0)
        mask = cv2.imread(os.path.join(args.mask_dir, Path(args.img_dir).stem + "_mask.png"),0)
    except UnidentifiedImageError:
        print('   [*] Not an Image SKIPPING')
        exit()
        
    print(f'    [*] Normalizing image')
    masked = cv2.normalize(masked, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print(f'    [*] Normalizing mask')
    mask = 1-cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)   
    
    print(f'    [*] Attempting to inpaint')
    inpainted = inpaint(masked,mask)
    
    filename = os.path.join(args.out_dir, Path(args.img_dir).stem + "_inpainted.tif")
    print(f'      [*] Dumping inpainted image to {filename}')
    cv2.imwrite(filename, inpainted*255)    
        
if __name__ == '__main__':
    print(f'[*] Starting Inpainting process')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    if(not args.debug):
        print(f'    [] Processing files from {args.img_dir}')
        test_files = glob((args.img_dir+'/*').format('float32'))
        container(test_files)
    else:
        print(f'  [] Running in debug mode')
        singleMode()
    print(f'[*] Script Succeeded')