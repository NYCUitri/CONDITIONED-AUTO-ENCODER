# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import glob
import tensorflow as tf  
import numpy as np
import yaml
from datasets import create_dataset, create_tfrecords
from nets import model
import time


def yaml_load():
    with open("parameter.yaml") as stream:
        param = yaml.safe_load(stream)
    return param


def main():
    param = yaml_load()

    dataDir = param['dataDir']
    #00: Create train data
    '''
    saveDirs = "./tfrecords"
    fDict = create_tfrecords.load_files_info(dataDir)
    create_tfrecords.convert_to_tfrecords(fDict, saveDirs)
    print("-------")
    '''

    #01:load train data 
    tfDir = param['tfDir']
    fileList = glob.glob(os.path.join(tfDir, "*.tfrecord"))
    
    # TODO: Show file list
    #print(fileList)

    data_train, label_batch = create_dataset.parse_tfrecords(fileList, param['batch_size'])

    # TODO: show data and label
    #print(data_train.shape)
    #print(label_batch.shape)
    label_int32 = tf.cast(label_batch, tf.int32)

    mostIndex = tf.argmax(tf.compat.v1.bincount(label_int32))
    mostValue = tf.gather(label_int32, mostIndex)
    label_float32 = tf.cast(tf.equal(label_int32, mostValue), tf.float32)
    label_train = tf.where(tf.greater(label_float32, 0), -1, 1)

    data_train = tf.cast(data_train, tf.float32)
    label_train = tf.cast(label_train, tf.float32)
    #02:build model
    #ypred, logits = model.build_model(data_train, label_train, True, None, param['frameNums'], param['mels'])
    #loss, accuracy = model.calu_loss(data_train, label_train, ypred, 5, 0.75, label_batch, logits)

    ypred = model.build_model(data_train, label_train, True, None, param['frameNums'], param['mels'])
    loss = model.calu_loss(data_train, label_train, ypred, 5, 0.75)

    #03:train model
    train_step = param['trainNum'] // param['batch_size']
    tf.summary.scalar('train/loss', loss)
    merged = tf.compat.v1.summary.merge_all() 
    summary_writer = tf.compat.v1.summary.FileWriter(param['log'], tf.compat.v1.get_default_graph())

    global_step = tf.compat.v1.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(param['learning_rate'], global_step=global_step, decay_steps=train_step, decay_rate=0.94, staircase=True, name='exponential_decay_learning_rate')
    # learning_rate = param['learning_rate']
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.compat.v1.control_dependencies(update_ops):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False).minimize(loss, global_step=global_step)
    saver = tf.compat.v1.train.Saver(max_to_keep=10)

    config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                      allow_soft_placement=True)
    with tf.compat.v1.Session(config=config) as sess:
        ckpt = tf.compat.v1.train.get_checkpoint_state(param['checkpoint'])
        if ckpt and tf.compat.v1.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.compat.v1.global_variables_initializer())
        # flops = tf.profiler.profile(options = tf.profiler.ProfileOptionBuilder.float_operation())
        # print('FLOP before freezing', flops.total_float_ops)
        coord = tf.compat.v1.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        print('Start training...')

        loss_list = []
        for epoch in range(param['epochs']):
            start_time = time.time()
            for step in range(train_step):
                _ = sess.run(train_op)

            train_loss = 0
            n_batch = 0
            train_acc = 0
            #if epoch + 1 == 1 or (epoch + 1) % 10 == 0:
            for step in range(train_step):
                #err, acc, train_summary = sess.run([loss, accuracy, merged])
                err, train_summary = sess.run([loss, merged])
                train_loss += err   
                n_batch += 1
                #train_acc += acc
            summary_writer.add_summary(train_summary, epoch + 1)
            #print("Epoch %d of %d took %fs" % (epoch + 1, param['epochs'], time.time() - start_time))
            #print("   train loss:%f" % (train_loss / n_batch))

            loss_list.append(train_loss / n_batch)
            #print("   train acc:%f" % (train_acc / n_batch))
            
            if epoch + 1 == 1 or (epoch + 1) % 40 == 0:
                model_path = os.path.join(param['checkpoint'], param['model_name'])
                save_path = saver.save(sess, model_path, global_step=global_step)
                print("Model saved in file: ", save_path)

        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(30, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss_list)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train"], loc="upper right")
        plt.savefig("learning_curve.png")

        summary_writer.close()
        coord.request_stop()
        coord.join(threads)
    



if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    main()