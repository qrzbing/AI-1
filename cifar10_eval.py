# -*- coding:utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
# TODO: 分析各函数的含义，添加注释
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

# 测试数据存放的路径
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/cifar10_eval',
    """Directory where to write event logs."""
)
# 测试数据
tf.app.flags.DEFINE_string(
    'eval_data', 'test',
    """Either 'test' or 'train_eval'."""
)
# 检查点存放路径
tf.app.flags.DEFINE_string(
    'checkpoint_dir', '/tmp/cifar10_train',
    """Directory where to read model checkpoints."""
)
# 每次评价的间隔时间
tf.app.flags.DEFINE_integer(
    'eval_interval_secs', 60 * 5,
    """How often to run the eval."""
)
# 应该是评价用的测试样例的数量
tf.app.flags.DEFINE_integer(
    'num_examples', 10000,
    """Number of examples to run."""
)
# FIXME: 这是啥，评价是否只跑一次？
tf.app.flags.DEFINE_boolean(
    'run_once', False,
    """Whether to run eval only once."""
)
# TODO: 测试集文件的位置？


def eval_once(saver, summary_writer, top_k_op, summary_op):
    """Run Eval once. 本函数为跑训练样本一次

    Args: 参数：
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """

    # 接下来一段是从文件中恢复checkpoint file
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(
            FLAGS.checkpoint_dir)  # checkpoint的存储路径
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]  # 得到global step
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners. 开启线程协调器
        coord = tf.train.Coordinator()
        # 接下来一段函数是图片和标签的批处理
        try:
            threads = []  # 一串序列
            # 下面这一段是对每个线程进行操作
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(
                    # 启动线程
                    qr.create_threads(
                        sess,
                        coord=coord,
                        daemon=True,
                        start=True
                    )
                )

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions. 正确预测的计数器
            total_sample_count = num_iter * FLAGS.batch_size  # 总的测试数据
            step = 0  # 轮数
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1. 计算准确率
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        # 关闭线程协调器
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    """Eval CIFAR-10 for a number of steps."""
    # 复现计算图
    with tf.Graph().as_default() as g:
        # Get images and labels for CIFAR-10.
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data=eval_data)
        print(images)
        print(labels)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        # 从我们建立的推理模型（应该是前向过程）计算出logits模型
        logits = cifar10.inference(images)
        print(logits)
        # Calculate predictions. 计算预测值
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        # 返回 label 数量的一个bool型张量

        # 重新加载滑动平均
        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
        summary_op = tf.summary.merge_all()

        # 指定一个文件用来保存图。
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(
                saver,
                summary_writer,
                top_k_op,  # calculate predictions
                summary_op
            )
            # FIXME: 这一块是啥时候在哪判断为 False 的？
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):  # 已经存在，删除路径
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)  # 建立路径
    evaluate()  # 测试函数


if __name__ == '__main__':
    tf.app.run()
