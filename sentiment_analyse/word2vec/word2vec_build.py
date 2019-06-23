# !/usr/bin/python
# -*- coding:utf-8 -*-
# Author:Yuanyuan Zhang

import os
import sys
from gensim.models import Word2Vec


class word2vector(object):
    def __init__(self, sentence_path, w2v_path):
        self.sentence_path = sentence_path
        self.w2v_path = w2v_path

    def extract_wordlist(self):
        wordlist = []
        with open(os.path.join(self.sentence_path), 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in lines:
                word = line.split()
                wordlist.append(word)
        return wordlist

    def word2vec_model_build(self, input):
        # input = self.e
        model = Word2Vec(size=300, window=2, min_count=1)
        model.build_vocab(input)
        model.train(input, total_examples=model.corpus_count, epochs=100)
        modelpath = os.path.join(self.w2v_path, 'word2vec.h5')
        model.save(modelpath)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("please input processed_sentence dir")
        sys.exit(0)
    else:
        input_dir = sys.argv[1]
    # input_dir = os.path.dirname(os.getcwd())
    w2v_model_dir = os.path.join(input_dir, "dataset/woc2vec_model")
    w2v_model_path = os.path.join(w2v_model_dir, 'w2v_model.h5')
    input_dir = os.path.join(input_dir, "dataset/processed_sentence")
    input_path = os.path.join(input_dir, 'result.txt')
    if not os.path.exists(w2v_model_dir):
        os.mkdir(w2v_model_dir)

    wordlist = word2vector(input_path, w2v_model_dir).extract_wordlist()
    word2vector(input_dir, w2v_model_dir).word2vec_model_build(wordlist)



