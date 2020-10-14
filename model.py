from utils import *
from u_net import *
import time
from tqdm import tqdm

class denoiser(object):
    def __init__(self, sess, input_c_dim=1):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.compat.v1.placeholder(tf.float32, [None, None, None, self.input_c_dim],
                                 name='clean_image')
        self.Y = autoencoder((self.Y_))
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        print("[*] Initialize model successfully...")

    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(self.sess, full_path)
            return True
        else:
            return False

    def test(self, test_files, ckpt_dir, save_dir, dataset_dir, stride=64, save_npy=True):
        """Test SAR2SAR"""
        tf.compat.v1.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        print("[*] start testing...")
        print(f' [] Found {len(test_files)} files')
        for idx in range(len(test_files)):
            timer_start = time.time()
            print("  [*] Processing file "+test_files[idx])
            real_image = load_sar_images(test_files[idx]).astype(np.float32) / 255.0
            # scan on image dimensions
            pat_size = 256
            # Pad the image
            im_h = np.size(real_image, 1)
            im_w = np.size(real_image, 2)

            count_image = np.zeros(real_image.shape)
            output_clean_image = np.zeros(real_image.shape)

            if im_h==pat_size:
                x_range = list(np.array([0]))
            else:
                x_range = list(range(0, im_h - pat_size, stride))
                if (x_range[-1] + pat_size) < im_h: x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

            if im_w==pat_size:
                y_range = list(np.array([0]))
            else:
                y_range = list(range(0, im_w - pat_size, stride))
                if (y_range[-1] + pat_size) < im_w: y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))
#             Nested Progress Bars
#             outer_loop = tqdm(x_range)
#             inner_loop = tqdm(y_range)
            loop = tqdm(range(len(x_range)*len(y_range)))
            
            for x in x_range:
#                 inner_loop.refresh()  # force print final state
#                 inner_loop.reset()    # reuse bar
#                 outer_loop.update()   # update mid tqdm
                for y in y_range:
#                     inner_loop.update() #update inner tqdm
                    loop.update()
                    
                    tmp_clean_image = self.sess.run([self.Y], feed_dict={self.Y_: real_image[:, x:x + pat_size,
                                                                                     y:y + pat_size, :]})
                    output_clean_image[:, x:x + pat_size, y:y + pat_size, :] = output_clean_image[:, x:x + pat_size,
                                                                               y:y + pat_size, :] + tmp_clean_image
                    count_image[:, x:x + pat_size, y:y + pat_size, :] = count_image[:, x:x + pat_size, y:y + pat_size,
                                                                        :] + np.ones((1, pat_size, pat_size, 1))
            output_clean_image = output_clean_image/count_image


            noisyimage = denormalize_sar(real_image)
            outputimage = denormalize_sar(output_clean_image)

            imagename = test_files[idx].replace(dataset_dir+"\\", "")
            timer_end = time.time()
            print()
            print("    [*]Denoised image %s in %.2fs" % (imagename, (timer_end-timer_start)))

            save_sar_images(outputimage, noisyimage, imagename, save_dir, save_npy)
