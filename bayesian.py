import math
import re

def count_function(l):
    return lambda x: len(list(filter(lambda item: (item == x),l)))

class Bayesian():
    def __init__(self,config):
        self.gram_number = config.get("bayesian-gram-number",3)
        self.smooth_weight = config.get("bayesian-smooth-weight",1)
        self.adaboost_number = config.get("bayesian-adaboost-number",0)
        self.language = config.get("language","en")
        self.pos = {}
        self.neg = {}
        self.pos_count = 0
        self.neg_count = 0

    def add_comment(self,comment,label):
        if label == 1:
            m = self.pos
            self.pos_count += 1
        else:
            m = self.neg
            self.neg_count += 1

        for i in range(len(comment)):
            txt = "".join(comment[i:i+self.gram_number])
            m[txt] = m.get(txt,0) + 1

    def train(self,data,_):
        length = len(data)
        for _,comment,label in data:
            self.add_comment(comment,label)

        for i in range(self.adaboost_number):
            for i in range(length):
                predict = self.predict_label(data[i][1])
                if predict != data[i][2]:
                    self.add_comment(comment,label)

        self.smooth(self.pos)
        self.smooth(self.neg)

    def smooth(self,m):
        for key in m:
            m[key] += self.smooth_weight

    def predict_label(self,comment):
        b = math.log(self.pos_count/self.neg_count)
        for i in range(len(comment)):
            txt = "".join(comment[i:i+self.gram_number])
            p_pos = math.log(self.pos.get(txt,self.smooth_weight)/self.pos_count)
            p_neg = math.log(self.neg.get(txt,self.smooth_weight)/self.neg_count)
            b = b + (p_pos - p_neg)
        if b > 0:
            return 1
        else:
            return -1


    def predict(self,data):
        predict = [self.predict_label(comment) for _,comment,_ in data]
        return predict
