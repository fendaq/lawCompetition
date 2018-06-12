import os
from .cnn_model import TCNNConfig, CharLevelCNN
import tensorflow as tf
import numpy as np
import keras


class Predictor():
    def __init__(self):
        self.config = TCNNConfig()
        self.batch_size = self.config.batch_size
        self.model = CharLevelCNN(self.config)
        self.model_dir = os.path.dirname(__file__)+'/model/'
        self.accusation_model = self.model_dir+'accusation/best_valid'
        self.relevant_model = self.model_dir+'relevant_articles/best_valid'
        self.year_model = self.model_dir+'year/best_valid'
        self.vocab_dir = self.model_dir+'vocab.txt'
        self.seq_length = self.config.seq_length

    def predict(self, content):
        accu_result = np.zeros(shape=len(content), dtype=np.int32)
        relevant_result = np.zeros(shape=len(content), dtype=np.int32)
        year_result = np.zeros(shape=len(content), dtype=np.int32)
        words, words_to_id = self.read_vocab(self.vocab_dir)
        data_len=len(content)
        batch_size=self.batch_size
        data_id = []
        for i in range(len(content)):
            data_id.append([words_to_id[x]
                            for x in content[i] if x in words_to_id])                            
        x_pad = keras.preprocessing.sequence.pad_sequences(
        data_id, int(self.seq_length), dtype='float32')

        num_batch = int((data_len - 1) / self.batch_size) + 1

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess=session, save_path=self.relevant_model)
            for i in range(num_batch):  # 逐批次处理
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                feed_dict = {
                self.model.x: x_pad[start_id:end_id],
                self.model.keep_prob: 1.0
                }
                relevant_result[start_id:end_id] = session.run(self.model.y_pred, feed_dict=feed_dict)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess=session, save_path=self.accusation_model)
            for i in range(num_batch):  # 逐批次处理
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                feed_dict = {
                self.model.x: x_pad[start_id:end_id],
                self.model.keep_prob: 1.0
                }
            accu_result[start_id:end_id] = session.run(self.model.y_pred, feed_dict=feed_dict)

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess=session, save_path=self.year_model)
            for i in range(num_batch):  # 逐批次处理
                start_id = i * batch_size
                end_id = min((i + 1) * batch_size, data_len)
                feed_dict = {
                self.model.x: x_pad[start_id:end_id],
                self.model.keep_prob: 1.0
                }
            year_result[start_id:end_id] = session.run(self.model.y_pred, feed_dict=feed_dict)


        result = []
        for i in range(len(content)):
            if year_result[i]==302:
                year_result[i]=-1
            if year_result[i]==301:
                year_result[i]==-2
            temp = {"accusation": [int(accu_result[i])], "articles":
                    [int(relevant_result[i])], "imprisonment": int(year_result[i])}
            result.append(temp)
        return result

    def read_vocab(self, vocab_dir):
        with open(vocab_dir, mode='r', encoding='utf8') as fp:
            words = [_.strip() for _ in fp.readlines()]
            word_to_id = dict(zip(words, range(len(words))))
        return words, word_to_id
