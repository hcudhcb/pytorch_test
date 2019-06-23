#!/usr/bin/python
# -*- coding:utf-8 -*-
# Author:Yuanyuan Zhang

import tensorflow as tf
import keras.backend as K
import sys
import os

import numpy as np

from gensim.models import Word2Vec
from keras.layers.core import Dense

from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import Nadam

class LSTM(object):
    def __init__(self,vector_path,label_path,model_path,learningrate):
        self.vector_path = vector_path
        self.label_path = label_path
        self.model_path = model_path
        self.learningrate = learningrate
    def get_x_vector(self):
        wordlist = []
        with open(self.vector_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                word = line.split()
                wordlist.append(word)
        w2v_model = Word2Vec.load(self.model_path)
        # vecs = np.array([np.array([np.array(w2v_model[word]) for word in sentence]) for sentence in wordlist])
        vecs = [[[0 for i in range(300)]] * 75 for sentence in wordlist]
        for i, sentence in enumerate(wordlist):
            for j, word in enumerate(sentence):
                vecs[i][j] = w2v_model[word]
        vecs = np.array(vecs, dtype='float16')
        return vecs
    def get_y_vector(self):
        return np.loadtxt(self.label_path)
    def shuffle(self):
        vector = self.get_x_vector()
        label = self.get_y_vector()
        index = [i for i in range(len(vector))]
        np.random.shuffle(index)
        return vector[index], label[index]
    def build_model(self):
        """
        build the model by using sequential
        return a model with 2 layers
        """
        model = Sequential()

        model.add(LSTM(400, input_shape=(75, 300), activation='relu', return_sequences=False))

        model.add(Dense(1, init='lecun_uniform', activation='sigmoid'))

        nadam = Nadam(lr=self.learningrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        model.compile(loss='binary_crossentropy', optimizer=nadam, metrics=['accuracy'])
        print(model.summary())
        return model

    def running(self):
        if True:
            model = self.build_model()
            x,y = self.shuffle()
            history = model.fit(x, y, batch_size=128, epochs=100, verbose=1,
                                validation_split=0.2)
            with open('log.txt', 'w') as f:
                f.write(str(history.history))
            model.save_weights('model_weights.h5')
            model.save('model.h5')
if __name__ == '__main__':
    if len(sys.argv)<5:
        print("please input x_path,y_path,word2vec_model_path,learningrate")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True  # allocate as you need
    session = tf.Session(config=config)
    K.set_session(session)
    LSTM(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]).running()