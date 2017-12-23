#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from settings import config
from bayesian import Bayesian
# from rnn import Rnn
import pickle

def print_result(evaluate_set,predict,accuracy,f1,precision,recall):
    print("accuracy: ",accuracy)
    print("f1:",f1)
    print("precision:",precision)
    print("recall:",recall)
    if config.get("print-predict?",False):
        for i in range(10):
            print(evaluate_set[i],"predict:",predict[i])

def write_wrong_predict(data,predict):
    wrong_predict = []
    for i in range(len(data)):
        if predict[i] != data[i][2]:
            wrong_predict.append(data[i])
    with open("wrong-result","w") as f:
        for item in wrong_predict:
            f.write(str(item)+"\n")

if __name__ == "__main__":
    print("============================ Loading data ============================")
    if config["language"] == 'cn':
        data_path = config["data-path-cn"]
    else:
        data_path = config["data-path-en"]
    train_set = pickle.load(open(data_path+"train.pkl","rb"))
    test_set = pickle.load(open(data_path+"test.pkl","rb"))
    evaluate_set = pickle.load(open(data_path+"evaluate.pkl","rb"))
    print("Data loaded from",data_path)

    algorithm = config.get("model",'bayesian')
    if algorithm == 'bayesian':
        model = Bayesian(config)
    elif algorithm == 'rnn':
        model = Rnn(config)

    if algorithm == 'bayesian' or config.get("train") == True:
        print("============================ Training...  ============================")
        model.train(train_set,test_set)


    if algorithm == 'bayesian' or config.get("train") == False:
        print("============================ Predicting...  ============================")
        predict, accuracy, f1, precision ,recall = model.predict(evaluate_set)
        print_result(evaluate_set,predict,accuracy,f1,precision,recall)
        write_wrong_predict(evaluate_set,predict)
