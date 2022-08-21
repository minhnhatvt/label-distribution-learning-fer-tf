import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Progbar
import pandas as pd

from models import *
from data_utils import *
import utils

import argparse
import sys
sys.path.append("cfg_files")

def parse_arg(argv=None):
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data", type=str,
    #                     default="",
    #                     help="Path to the train_data.csv, should have the following columns:\n'subDirectory_filePath,expression,valence,arousal,knn'",
    #                     required=True)
    # parser.add_argument("--val_data", type=str,
    #                     default=None,
    #                     help="Path to the validation_data.csv, should have the following columns:\n'subDirectory_filePath,expression'")
    # parser.add_argument("--pretrained_weights", type=str,
    #                     default=None,
    #                     help="load the pretrained weights of the model in /path/to/model_weights")
    parser.add_argument("--cfg", type=str,
                        default="config",
                        help="config file_name")
    parser.add_argument("--resume",
                        action= "store_true",
                        help="Resume training from the last checkpoint")


    args = parser.parse_args(argv)
    return args

def train_step(model, optimizer, x_batch_train, y_batch_train, x_batch_aux, config,
               global_labels, lamb_hat, lamb_optim, dcm_loss=None,
               loss_weights_dict=None, class_weights_dict=None,
               knn_weights=None,
               lamb=None, idx=None, neighbor_idx=None):
    if loss_weights_dict is None:
        loss_weights_dict = {
            'emotion_loss': 1,
            'va_loss': 1,
        }

    sample_weights = utils.create_sample_weights(y_batch_train[0], class_weights_dict)
    # Get neighbor prediction

    B, K, H, W, _ = x_batch_aux.shape
    x_batch_aux = tf.reshape(x_batch_aux, shape=(-1, H, W, 3))
    feat_aux, preds_aux = model(x_batch_aux, training=True)

    preds_aux = tf.reshape(preds_aux, shape=(B, K, -1))  # shape (B,K,C)
    feat_aux = tf.reshape(feat_aux, shape=(B, K, -1))  # shape (B,K,C)

    with tf.GradientTape() as tape:
        feat, preds = model(x_batch_train, training=True)
        attention_weights = model.weighting_net((feat), (feat_aux), training=True)

        # attention_weights = tf.ones((B,K))

        # construct label distribution
        emotion_cls_pred = preds
        emotion_cls_true = y_batch_train[0]
        # emotion_cls_true = tf.one_hot(emotion_cls_true,depth=7)
        # neighbor_dist = construct_target_distribution(tf.zeros(B,tf.uint8), (preds_aux), knn_weights, attention_weights, lamb=0)

        if idx is not None:
            lamb = tf.gather(lamb_hat, idx)
            lamb = tf.sigmoid(lamb)

        emotion_cls_true = utils.construct_target_distribution(emotion_cls_true, (preds_aux), knn_weights,
                                                         (attention_weights), lamb=lamb)

        emotion_loss = utils.CELoss(emotion_cls_true, emotion_cls_pred, sample_weights)

        dcm_loss_value = dcm_loss(feat, y_batch_train[0], feat_aux, tf.gather(global_labels, neighbor_idx),
                                    lamb_hat=lamb_hat,
                                    indices = tf.concat([idx, tf.reshape(neighbor_idx, -1)], axis=0))

        total_loss = emotion_loss + config.gamma * dcm_loss_value

        optimizing_variables = model.trainable_variables + [lamb_hat]
        gradients = tape.gradient(total_loss, optimizing_variables)
        gradients, lamb_hat_grad = gradients[:-1], gradients[-1]

        optimizer.apply_gradients(
            [(grad, var) for (grad, var) in zip(gradients, optimizing_variables) if grad is not None])

        lamb_optim.apply_gradients([(lamb_hat_grad, lamb_hat)])

        dcm_loss.update_centers(dcm_loss.compute_grads())

    return preds, total_loss, {'emotion_loss': emotion_loss,
                               'lamb': tf.reduce_mean(tf.reduce_mean(tf.sigmoid(tf.gather(lamb_hat, idx))))}


def test_step(model, x_batch, y_batch):
    feat, preds = model(x_batch, training=False)

    emotion_cls_pred = preds
    emotion_cls_true = y_batch[0]
    emotion_loss = tf.keras.losses.SparseCategoricalCrossentropy()(emotion_cls_true, emotion_cls_pred)

    total_loss = emotion_loss

    return preds, total_loss, {'emotion_loss': emotion_loss}






def train(model, optimizer, train_dataset, global_labels, config,
          val_dataset=None, epochs=5, load_checkpoint=False,
          loss_weights_dict=None,
          class_weights_dict=None):
    # define metrics for controlling process
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('val_accuracy')

    batches_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()

    ckpt_dir = config.checkpoint_dir
    # init values
    dcm_loss = utils.DiscriminativeLoss(config.num_classes, feature_dim=config.feature_dim)
    lamb_hat = tf.Variable(tf.zeros(shape=(len(global_labels), 1), dtype=tf.float32) + 0.5)
    lamb_optim = tf.keras.optimizers.SGD(10)
    best_val = 0
    iter_count = 0
    val_interval = config.val_interval
    save_interval = config.save_interval



    # setup checkpoint manager
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model, optimizer=optimizer,
                                     dcm_loss=dcm_loss)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="./checkpoints", max_to_keep=1
    )
    if load_checkpoint:
        status = checkpoint.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from sratch....")
    else:
        print("Initializing from scratch...")

    iter_count = checkpoint.step.numpy()

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        print("Epoch {}".format(int(iter_count / batches_per_epoch) + 1))
        print("LR: ", float(optimizer.learning_rate(optimizer.iterations)))

        # Training
        pb_i = Progbar(batches_per_epoch, width=30, interval=0.5,
                       stateful_metrics=['total_loss', 'emotion_loss', 'emotion_acc', 'avg_lamb'])
        for x_batch_train, va_regression_true, emotion_cls_true, x_batch_aux, knn_weights, idx, neighbor_idx in train_dataset:
            checkpoint.step.assign_add(1)
            iter_count += 1
            curr_epoch = int(iter_count / batches_per_epoch)

            lamb = None #we use trainable lambda
            y_batch_train = (emotion_cls_true, va_regression_true)
            preds, total_loss, loss_dict = train_step(model, optimizer, x_batch_train, y_batch_train, x_batch_aux, config,
                                                      global_labels, lamb_hat, lamb_optim, dcm_loss=dcm_loss,
                                                      class_weights_dict=class_weights_dict,
                                                      knn_weights=knn_weights,
                                                      lamb=lamb, idx=idx, neighbor_idx=neighbor_idx,
                                                      )
            train_loss(total_loss)
            emotion_acc = train_accuracy(emotion_cls_true, preds)

            pb_i.add(1, [('total_loss', total_loss.numpy()),
                         ('emotion_accuracy', emotion_acc),
                         ('avg_lamb', loss_dict['lamb'])])

            if iter_count % 1000 == 0 and val_dataset is not None:
                val_loss.reset_states()
                val_accuracy.reset_states()
                for x_batch, va_regression_true, emotion_cls_true in val_dataset:
                    y_batch = (emotion_cls_true, va_regression_true)
                    preds, total_loss, loss_dict = test_step(model, x_batch, y_batch)
                    val_loss(total_loss)  # update metric
                    val_accuracy(emotion_cls_true, preds)  # update metric
                acc = val_accuracy.result()
                print("\n---Iterations: {}, Val Acc: {:.4}".format(iter_count, acc))
                if (acc > best_val):
                    model.save_weights(f"{ckpt_dir}/best_val/Model")
                    print("====Best validation model saved!====")
                    best_val = acc

        save_path = manager.save()
        if (curr_epoch) % save_interval == 0:
            model.save_weights('f{ckpt_dir}/epoch_' + str(curr_epoch) + '/Model')

        print('End of Epoch: {}, Iter: {}, Train Loss: {:.4}, Emotion Acc: {:.4}'.format(curr_epoch, iter_count,
                                                                                         train_loss.result(),
                                                                                         train_accuracy.result()))
        # Validation
        if val_dataset is not None:
            if (curr_epoch) % val_interval == 0:  # validate
                val_loss.reset_states()
                val_accuracy.reset_states()

                for x_batch, va_regression_true, emotion_cls_true in val_dataset:
                    y_batch = (emotion_cls_true, va_regression_true)
                    preds, total_loss, loss_dict = test_step(model, x_batch, y_batch)

                    val_loss(total_loss)  # update metric
                    val_accuracy(emotion_cls_true, preds)  # update metric

                print('Val loss: {:.4},  Val accuracy: {:.4}'.format(val_loss.result(), val_accuracy.result()))
                print('===================================================')

                if (val_accuracy.result() > best_val):
                    model.save_weights(f"{ckpt_dir}/best_val/Model")
                    print("====Best validation model saved!====")
                    best_val = val_accuracy.result()
        print()
    return model


def main(train_data_path, image_dir, config, val_data_path=None, val_image_dir = None):
    assert (val_image_dir is not None) or (val_data_path is None)

    model = create_model(config)
    print("Model created!")

    train_dataset = get_train_dataset(train_data_path, image_dir, config)
    optimizer = utils.get_optimizer(train_dataset, config)
    val_dataset = get_test_dataset(val_data_path, image_dir, config) if val_data_path is not None else None
    print("Dataset loaded!")

    train_data = pd.read_csv(train_data_path)
    global_labels = tf.constant(train_data['expression'], dtype=tf.int32)

    print("Start training...")
    train(model, optimizer, train_dataset, global_labels, config,
          val_dataset=val_dataset,
          epochs=config.epochs,
          load_checkpoint=args.resume)

if __name__ == '__main__':
    args = parse_arg()
    # run_train(args.train_data,
    #           val_data_path=args.val_data)


    config = __import__(args.cfg).config
    print(config.__dict__)

    main(train_data_path= "data/rafdb/raf_train_knn_res50_3085.csv", image_dir="data/rafdb/aligned", config= config,
         val_data_path="data/rafdb/test.csv", val_image_dir="data/rafdb/aligned",
         )


