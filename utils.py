"""

"""

import os
import numpy as np
import tensorflow as tf
import cv2

# +-* + () + 10 digit + blank + space
# num_classes = 3 + 2 + 10 + 1 + 1

maxPrintLen = 100

tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
# tf.app.flags.DEFINE_integer('image_width', 180, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('cnn_count', 4, 'count of cnn module to extract image features.')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden', 128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 40, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 1000, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 500, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')

tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('train_dir', './imgs/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './imgs/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './imgs/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 0, 'num of gpus')

FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

# charset = '0123456789+-*()'
encode_maps = []
# decode_maps = []

with open('charset.txt', 'r', encding='utf-8') as f:
    content = f.readlines()
for idx, char in enumerate(content):
    encode_maps.append(char.strip('\n').strip(""))

def text2label(label):
    text = []
    for idx in label:
        text.append(encode_maps.index(idx))
    return text

def label2text(labels):
    pre_str = []
    for i in labels:
        if i == -1:
            pre_str.append('')
        else:
            pre_str.append(encode_maps[i])
    return ''.join(pre_str).strip('')


num_classes = len(encode_maps) + 1

# SPACE_INDEX = 0
# SPACE_TOKEN = ''
# encode_maps[SPACE_TOKEN] = SPACE_INDEX
# decode_maps[SPACE_INDEX] = SPACE_TOKEN


class DataIterator:
    def __init__(self, data_dir):
        self.image = []
        self.train_data = data_dir

    def get_batchsize_data(self, iter_step):
        batch_img = []
        batch_label = []
        for i in range(FLAGS.batch_size):
            im = cv2.imread(self.train_data[FLAGS.batch_size*iter_step+_], 0).astype(np.float32) / 255.0
            scale = FLAGS.image_height / im.shape[0]
            im_resized = cv2.resize(im, None, fx=scale, fy=scale)
            im_resized = np.expand_dims(im_resized, 2)
            batch_img.append(im_resized)

            with open(self.train_data[FLAGS.batch_size*iter_step+_][0:-3]+'txt', 'r', encoding='utf-8') as f:
                content = f.readlines()[0]
            label = text2label(content.strip('\n').strip(''))
            batch_label.append(label)
            f.close()
        assert len(batch_img) == len(batch_label), 'train or val data batch len not equal'
        max_width = max([e.shape[1] for e in batch_img])
        result_img = np.zeros((len(batch_img), batch_img[0].shape[0], max_width, 1))
        for idx, item in enumerate(batch_img):
            result_img[idx, 0:item.shape[1]] = item
        maxlen = max([len(e) for e in batch_label])
        result_img_lenght = [e.shape[1] for e in batch_img]
        batch_labels = sparse_tuple_from_label(batch_label)

        return result_img, np.asarray(result_img_lenght, dtype=np.int32), batch_labels

    def get_val_label(self, iter_step):
        batch_label_eval = []
        for _ in range(FLAGS.batch_size):
            with open(self.train_data[FLAGS.batch_size*iter_step+_][0:-3]+'txt', 'r', encoding='utf-8') as f:
                content = f.readlines()[0]
            batch_label_eval.append(content.strip('\n').strip(''))
            f.close()
        return batch_label_eval


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'w') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs
