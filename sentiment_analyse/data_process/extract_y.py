#!/usr/bin/python
# -*- coding:utf-8 -*-
# Author:Yuanyuan Zhang
import csv
import os
import sys
class extract_y(object):
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
    def extract_label(self):
        with open(self.input_path, 'r', encoding="utf-8", errors="ignore") as read_file:
            lines = csv.reader(read_file)
            for line in lines:
                label = str(line[0])
                if label=='4':
                    label = '1'
                f = open(self.output_path,'a')
                f.write(label)
                f.write('\n')
                f.close()
if __name__ == '__main__':
    if(len(sys.argv)<3):
        print("please input input_csv_path,output_path")
        exit(0)
    else:
        if not os.path.exists(sys.argv[2]):
            os.mkdir(sys.argv)
        extract_y(sys.argv[1],sys.argv[2]).extract_label()
