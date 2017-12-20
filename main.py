#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from settings import config
from bayesian import Bayesian
import pickle

def print_result(evaluate_set,predict,accuracy,f1):
    print("accuracy: ",accuracy)
    print("f1:",f1)
    if config.get("print-predict?",False):
        for i in range(10):
            print(evaluate_set[i],"predict:",predict[i])

if __name__ == "__main__":
    if config["language"] == 'cn':
        data_path = config["data-path-cn"]
    else:
        data_path = config["data-path-en"]
    train_set = pickle.load(open(data_path+"train.pkl","rb"))
    test_set = pickle.load(open(data_path+"test.pkl","rb"))
    evaluate_set = pickle.load(open(data_path+"evaluate.pkl","rb"))

    algorithm = config.get("model",'bayesian')
    if algorithm == 'bayesian':
        model = Bayesian(config)

    model.train(train_set,test_set)
    predict, accuracy, f1, [p,r] = model.predict(evaluate_set)
    print_result(evaluate_set,predict,accuracy,f1)
