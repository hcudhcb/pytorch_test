#!/usr/bin/python
# -*- coding:utf-8 -*-
# Author:Yuanyuan Zhang
import sys
import csv

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from data_process import data_process as data_processing


class extract_sentence(object):
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
    def extract_sentence(self):
        stopWords = set(stopwords.words('english'))
        stopWords.add(".")
        with open(self.input_path,'r',encoding="utf-8",errors="ignore") as read_file:
            lines = csv.reader(read_file)
            for line in lines:
                words = " "
                words_return = ""
                paragraph = str(line[-1])
                test_dataprocess = data_processing()
                process_line = test_dataprocess.processAll(paragraph, "")
                words_current = word_tokenize(process_line)
                for i, word in enumerate(words_current):
                    if word in stopWords:
                        words_current.remove(word)
                    else:
                        words_current[i] = ' ' + word
                words_return = words_return + words.join(words_current)
                f = open(os.path.join(self.output_path,'result.txt'), 'a')
                for line in words_return:
                    f.writelines(line)
                f.write('\n')
                f.close()
if __name__ == '__main__':
    if len(sys.argv)<=1:
        print("please input input_dir")
        sys.exit(0)
    else:
        dir = str(sys.argv[1])
    #dir = os.path.dirname(os.getcwd())
    #dir = "/bio/kihara-fast-scratch/wang3702/zhangyuanyuan/dataset"
    input_path = os.path.join(dir,"dataset.csv")
    output_dir = os.path.join(dir,"processed_sentence")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    extract_sentence(input_path,output_dir).extract_sentence()






