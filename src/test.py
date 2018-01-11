#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from settings import config
from bayesian import Bayesian
from rnn import Rnn
import xml.etree.ElementTree
import sys

def write_predict_result(data,predict):
    output_file = config.get("output-path","./predict.xml")
    for comment,i in zip(root,range(len(root))):
        comment.set("polarity",str(predict[i]))
    sys.stdout = open(output_file,"w")
    xml.etree.ElementTree.dump(root)

if __name__ == "__main__":
    print("============================ Loading data ============================")
    input_path = config.get("input-path","data/task2_input_cn.xml")
    root = xml.etree.ElementTree.parse(input_path).getroot()
    data =[[comment.attrib,comment.text] for comment in root]

    language = config.get("language",'cn')
    if language == "en":
        model = Bayesian(config)
    else:
        model = Rnn(config)
    model.load()
    print("============================ Predicting...  ============================")
    predict = model.predict(data)
    # print(predict)
    write_predict_result(root,predict)
