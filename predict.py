"""

"""

import datetime
import logging
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import shutil
import cnn_lstm_otc_ocr
import utils
import helper



os.environ[“CUDA_VISIBLE_DEVICES”] = “0”
FLAGS = utils.FLAGS

logger = logging.getLogger('Traing for OCR using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)



def infer(path, mode='infer'):
    
    imgList = [os.path.join(path, e) for e in os.listdir(path) if e.endswith('.jpg')]
    print(len(imgList))

    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    total_steps = len(imgList) / FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        # ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        ckpt = FLAGS.checkpoint_dir
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        for curr_step in range(total_steps):

            show_img_name = []
            batch_imgs = []
            seq_len_input = []

            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                show_img_name.append(img)
                im = cv2.imread(img, 0).astype(np.float32) / 255.
                # im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                scale = FLAGS.image_height / im.shape[0]
                im = cv2.resize(im, None, fx=scale, fy=scale)
                im = np.expand_dims(im, 2)
                batch_imgs.append(im)

            max_width = max([e.shape[1] for e in batch_imgs])
            inputs_imgs = np.zeros((len(batch_imgs), batch_imgs[0].shape[0], max_width, 1))

            for idx, item in enumerate(batch_imgs):
                inputs_imgs[idx, 0:item.shape[1], :] = item
            seq_len_input = [e.shape[1], for e in batch_imgs]

            imgs_input = inputs_imgs
            seq_len_input = np.asarray(seq_len_input)

            feed = {model.inputs: imgs_input, model.seq_len: seq_len_input}
            dense_decoded_code = sess.run(model.dense_decoded, feed)

            batch_result = []
            for decode_code in dense_decoded_code:
                pred_strings = utils.label2text(decode_code)
                batch_result.append(pred_strings)
            for i in range(len(show_img_name)):
                print(show_img_name[i], batch_result[i])

        with open('./result.txt', 'a') as f:
            for code in decoded_expression:
                f.write(code + '\n')


def main(_):
    if FLAGS.mode == 'train':
        pass
    else:
        infer(FLAGS.infer, FLAGS.mode)

    train(FLAGS.train_dir, FLAGS.mode)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

# python main.py --train_dir ../img --image_height=30 --restore --batch_size=64 --log_dir=./log/aa --mode=train




