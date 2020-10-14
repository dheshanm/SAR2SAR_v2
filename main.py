#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Hiding verbose tensorflow logs
import tensorflow as tf
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='GPU flag. 1 = gpu, 0 = cpu')
parser.add_argument('--out_dir', dest='out_dir', default=os.path.join(os.getcwd(), 'output'), help='test examples are saved here')
parser.add_argument('--input_data', dest='data_dir', default=os.path.join(os.getcwd(), 'data'), help='data set for testing')
parser.add_argument('--stride_size', dest='stride_size', type=int, default=64, help='define stride when image dim exceeds 264')
parser.add_argument('--save_npy', dest='save_npy', type=bool, default=False, help='True if you want npy files saved to disk')
parser.add_argument('--single', dest='single', help='To process a single file')
args = parser.parse_args()

checkpoint_dir = os.path.join(os.getcwd(), 'checkpoint')

if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

from model import denoiser

def denoiser_test(denoiser):
    test_data = args.data_dir
    if args.single:
        print("[*] Starting in single mode")
        test_files = glob(args.single)
    else:
        print(
            "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), test_data, args.out_dir))
        test_files = glob((test_data+'/*.npy').format('float32'))
    denoiser.test(test_files, ckpt_dir=checkpoint_dir, save_dir=args.out_dir, dataset_dir=test_data, stride=args.stride_size, save_npy=args.save_npy)

if __name__ == '__main__':
    print("[*] Debug Info")
    if args.use_gpu:
        print("  [] Running in GPU mode")
        gpu_options = tf.compat.v1.GPUOptions(allow_growth = True)
        print("  [] Using dynamic memory alloc.")
        print("  [] Current Stride Size: %d" % args.stride_size)
        print("  [] Save npy: %r" % args.save_npy)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)
    else:
        print("  [] Running in CPU mode")
        print("  [] Current Stride Size: %d" % args.stride_size)
        with tf.compat.v1.Session() as sess:
            model = denoiser(sess)
            denoiser_test(model)
            
    print("[] Script succeeded")
