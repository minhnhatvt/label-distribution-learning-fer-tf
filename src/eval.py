import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Progbar
import argparse
from train import *
from models import *
from data_utils import *
import time
import sys
sys.path.append("cfg_files")

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

def test(model, test_dataset):
    print("Evaluating model..")
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    test_accuracy.reset_states()
    num_classess = config.num_classes
    start = time.time()
    batches_per_epoch = tf.data.experimental.cardinality(test_dataset).numpy()
    pb = Progbar(batches_per_epoch, width=30)
    for x_batch, va_regression_true, emotion_cls_true  in test_dataset:
        y_batch = (emotion_cls_true, va_regression_true)
        preds, total_loss, loss_dict = test_step(model, x_batch, y_batch)
        test_loss(total_loss)  # update metric
        test_accuracy(emotion_cls_true, preds)  # update metric
        pb.add(1)

    end = time.time()
    print("Evaluating time: %d seconds" % ((end - start)))
    print("Evaluate accuracy: {:.4}".format(test_accuracy.result()))

def parse_arg(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_data_path", type=str,
                        default="data/rafdb/raf_test.csv",
                        help="Path to the test_data.csv, should have the following columns:\n'subDirectory_filePath,expression'")
    parser.add_argument("--test_image_dir", type=str,
                    default="data/rafdb/aligned",
                    help="Path to the directory containing training images")
    parser.add_argument("--trained_weights", type=str,
                        default=None,
                        help="load the trained weights of the model in /path/to/model_weights")
    parser.add_argument("--cfg", type=str,
                        default="config_resnet50_raf",
                        help="config file_name")

    global args
    args = parser.parse_args(argv)

def main(test_data_path, test_image_dir, config):
    model = create_model(config)
    print("Model created!")

    if args.trained_weights is not None:
        print("Load weights from: " + args.trained_weights)
        model.load_weights(args.trained_weights + "/Model" )

    test_dataset = get_test_dataset(test_data_path, test_image_dir, config)
    test(model, test_dataset)

if __name__ == '__main__':
    parse_arg()
    config = __import__(args.cfg).config
    print(config.__dict__)

    main(test_data_path= args.test_data_path, test_image_dir=args.test_image_dir, config= config)