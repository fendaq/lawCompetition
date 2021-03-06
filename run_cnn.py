#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf

from prepareData import read_vocab, batch_iter, get_data_with_vocab, build_vocab, read_catagory, read_word2vec
import cnn_model

base_dir = './good'
train_dir = os.path.join(base_dir, 'data_train.json')
test_dir = os.path.join(base_dir, 'data_test.json')
valid_dir = os.path.join(base_dir, 'data_valid.json')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
target_name = 'relevant_articles'
if target_name == 'accusation':
    cat_dir = os.path.join(base_dir, 'accu.txt')
else:
    cat_dir = os.path.join(base_dir, 'law.txt')
# 任务类型有三种，accusation,term_of_imprisonment,relevant_articles

save_dir = './model/'+target_name
save_path = os.path.join(save_dir, 'best_valid')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.x: x_batch,
        model.y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 64)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run(
            [model.loss, model.precision], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(x_train, y_train, x_val, y_val):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'd:/tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.precision)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 10000  # 如果超过3000轮未提升，提前结束训练

    flag = False
    for _ in range(config.num_epochs):
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, 0.8)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run(
                    [model.loss, model.precision], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train,
                                 loss_val, acc_val, time_dif, improved_str))

            session.run(model.optimizer, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(x_test, y_test):
    print("Loading test data...")
    start_time = time.time()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test = np.argmax(y_test, 1)
    y_pred = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred[start_id:end_id] = session.run(
            model.y_pred, feed_dict=feed_dict)

    # 评估

    # 混淆矩阵
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    y_res = []
    for _, pred in enumerate(y_pred):
        y_res.append(int(pred))
    return y_res


if __name__ == '__main__':
    print('Configuring CNN model...')
    config = cnn_model.TCNNConfig()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, valid_dir, test_dir,
                    vocab_dir, config.vocab_size, 10)
    categories, cat_to_id = read_catagory(cat_dir)

    if config.vocab_init:
        words, word_to_id = read_word2vec(vocab_dir)
    else:
        words, word_to_id = read_vocab(vocab_dir)

    config.vocab_size = len(words)
    x_train_, y_train_ = get_data_with_vocab(
        train_dir, word_to_id, cat_to_id, config, target_case=target_name)
    x_val_, y_val_ = get_data_with_vocab(
        valid_dir, word_to_id, cat_to_id, config, target_case=target_name)
    x_test_, y_test_ = get_data_with_vocab(
        test_dir, word_to_id, cat_to_id, config, target_case=target_name)
    model = cnn_model.CharLevelCNN(config)
    train(x_train_, y_train_, x_val_, y_val_)
    y_pred = test(x_test_, y_test_)
    # np.savetxt(target_name+'.txt',y_pred)
