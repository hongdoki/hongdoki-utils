"""Converts numpy format data to TFRecords file format with Example protos."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import os
import sys
import glob
import tensorflow as tf

FLAGS = None


def _int64_feature(value):
    tf.train.Feature()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert_to(data_set, name):
    """Converts a dataset to tfrecords."""
    images = data_set[0]
    labels = data_set[1]
    num_examples = images.shape[0]

    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3] if len(images.shape) > 4 else 1

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        image_raw = images[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(rows),
            'image/width': _int64_feature(cols),
            'image/depth': _int64_feature(depth),
            'image/class/label': _int64_feature(int(np.argmax(labels[index]))),
            'image/encoded': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def main(unused_argv):
    # Get the data.
    npy_paths = glob.glob(os.path.join(FLAGS.directory, '*X.npy'))
    for path in npy_paths:
        images = np.load(path)
        labels = np.load(path[:-5] + 'y.npy')

        num_classes = labels.shape[1]
        # convert each class separately
        if FLAGS.write_each_class:
            for i in xrange(num_classes):
                a = images[labels[:, i] == 1]
                convert_to((a, np.array([[1 if j == i else 0 for j in range(num_classes)]] * len(a))),
                           path.split('/')[-1][:-6] + ('_cls%d' % i))


        # convert all classes with shuffle

        if FLAGS.shuffle:
            perm = np.random.permutation(len(labels))
            labels = labels[perm]
            images = images[perm]
        convert_to((images, labels), path.split('/')[-1][:-6])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--directory',
        type=str,
        default='tmp',
        help='Directory to download data files and write the converted result'
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        default=True,
        help='shuffle npy data before coverting to tfrecords'
    )
    parser.add_argument(
        '--write_each_class',
        type=bool,
        default=True,
        help='write converted result into several files by each class'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
