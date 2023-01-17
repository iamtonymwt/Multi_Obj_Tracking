import tensorflow.compat.v1 as tf
import sys

def load_tfrecords(tf_records_filenames):
      for record in tf.python_io.tf_record_iterator(tf_records_filenames):
            example = tf.train.Example()
            example.ParseFromString(record)
            name = example.features.feature['image/filename'].bytes_list.value
            height = example.features.feature['image/height'].int64_list.value
            xmin = example.features.feature['image/object/bbox/xmin'].float_list.value
            reID = example.features.feature['image/object/track/label'].int64_list.value
            print("name: ", name)
            print("height: ", height)
            print("xmin: ", xmin)
            print("reID: ", reID)
            break

if __name__ == '__main__':
      load_tfrecords("./dataset/caltech/train.tfrecord")