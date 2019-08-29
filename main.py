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


def train(train_dir=None, mode='train'):
    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    label_files = [os.pat.join(train_dir, e) for e in os.listdir(train_dir) if e.endswith('.txt') and os.path.exists(os.path.join(train_dir, e.replace('.txt', '.jpg')))]
    train_num = int(len(label_files) * 0.8)
    test_num = len(label_files) - train_num

    print('total num', len(label_files), 'train num', train_num, 'test num', test_num)
    train_imgs = label_files[0:train_num]
    test_imgs = label_files[train_num:]


    print('loading train data')
    train_feeder = utils.DataIterator(data_dir=train_imgs)
    

    print('loading validation data')
    val_feeder = utils.DataIterator(data_dir=test_imgs)
   

    num_batches_per_epoch_train = int(train_num / FLAGS.batch_size)  # example: 100000/100

    num_batches_per_epoch_val = int(test_num / FLAGS.batch_size)  # example: 10000/100
   

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        for cur_epoch in range(FLAGS.num_epochs):
            
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()
            if cur_epoch == 0:
                random.shuffle(train_feeder.train_data)

            # the training part
            for cur_batch in range(num_batches_per_epoch_train):
                if (cur_batch + 1) % 100 == 0:
                    print('batch', cur_batch, ': time', time.time() - batch_time)
                batch_time = time.time()
                
                batch_inputs, result_img_length, batch_labels = \
                    train_feeder.get_batchsize_data(cur_batch)
               
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels,
                        model.seq_len: result_img_length}

                # if summary is needed
                summary_str, batch_cost, step, _ = \
                    sess.run([model.merged_summay, model.cost, model.global_step, model.train_op], feed)
                # calculate the cost
                train_cost += batch_cost * FLAGS.batch_size

                train_writer.add_summary(summary_str, step)

                # save the checkpoint
                if step % FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save checkpoint at step {0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)

                # train_err += the_err * FLAGS.batch_size
                # do validation
                if step % FLAGS.validation_steps == 0:
                    acc_batch_total = 0
                    lastbatch_err = 0
                    lr = 0
                    for val_j in range(num_batches_per_epoch_val):
                        result_img_val, seq_len_input_val, batch_label_val = \
                            val_feeder.get_batchsize_data(val_j)
                        val_feed = {model.inputs: result_img_val,
                                    model.labels: batch_label_val,
                                    model.seq_len: seq_len_input_val}

                        dense_decoded, lastbatch_err, lr = \
                            sess.run([model.dense_decoded, model.cost, model.lrn_rate],
                                     val_feed)

                        # print the decode result
                        val_pre_list = []
                        for decode_code in dense_decoded:
                            pred_strings = utils.label2text(decode_code)
                            val_pre_list.append(pred_strings)
                        ori_labels = val_feeder.get_val_label(val_j)

                        acc = utils.accuracy_calculation(ori_labels, val_pre_list,
                                                         ignore_value=-1, isPrint=True)
                        acc_batch_total += acc

                    accuracy = acc_batch_total / num_batches_per_epoch_val

                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                          "accuracy = {:.3f},avg_train_cost = {:.3f}, " \
                          "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                     lastbatch_err, time.time() - start_time, lr))


def infer(img_path, mode='infer'):
    imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
    imgList = helper.load_img_path(img_path)
    print(imgList[:5])

    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    total_steps = len(imgList) / FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        for curr_step in range(total_steps):

            imgs_input = []
            seq_len_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                im = cv2.imread(img, 0).astype(np.float32) / 255.
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])

                def get_input_lens(seqs):
                    length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)

                    return seqs, length

                inp, seq_len = get_input_lens(np.array([im]))
                imgs_input.append(im)
                seq_len_input.append(seq_len)

            imgs_input = np.asarray(imgs_input)
            seq_len_input = np.asarray(seq_len_input)
            seq_len_input = np.reshape(seq_len_input, [-1])

            feed = {model.inputs: imgs_input}
            dense_decoded_code = sess.run(model.dense_decoded, feed)

            for item in dense_decoded_code:
                expression = ''

                for i in item:
                    if i == -1:
                        expression += ''
                    else:
                        expression += utils.decode_maps[i]

                decoded_expression.append(expression)

        with open('./result.txt', 'a') as f:
            for code in decoded_expression:
                f.write(code + '\n')


def main(_):

    train(FLAGS.train_dir, FLAGS.mode)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()

# python main.py --train_dir ../img --image_height=30 --restore --batch_size=64 --log_dir=./log/aa --mode=train




