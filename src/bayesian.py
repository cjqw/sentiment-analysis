import math
import re
import pickle
from parser import split

def count_function(l):
    return lambda x: len(list(filter(lambda item: (item == x),l)))

class Bayesian():
    def __init__(self,config):
        self.gram_number = config.get("bayesian-gram-number",3)
        self.smooth_weight = config.get("bayesian-smooth-weight",1)
        self.file_path = config.get("model-path","model/")
        self.language = config.get("language","en")
        self.pos = {}
        self.neg = {}
        self.pos_count = 0
        self.neg_count = 0

    def save(self):
        model_path = self.file_path+self.language+"-bayesian.model"
        model = [self.gram_number,self.smooth_weight,self.pos,self.neg,
                 self.pos_count,self.neg_count]
        print(model_path)
        pickle.dump(model,open(model_path,"wb"))

    def load(self):
        model_path = self.file_path+self.language+"-bayesian.model"
        model = pickle.load(open(model_path,"rb"))
        self.gram_number,self.smooth_weight,self.pos,self.neg = model[:4]
        self.pos_count,self.neg_count = model[4:]

    def add_comment(self,comment,label):
        if label == 1:
            m = self.pos
            self.pos_count += 1
        else:
            m = self.neg
            self.neg_count += 1
        content = split(comment,self.language)
        for i in range(len(content)):
            txt = "".join(content[i:i+self.gram_number])
            m[txt] = m.get(txt,0) + 1

    def train(self,data,_):
        length = len(data)
        for _,comment,label in data:
            self.add_comment(comment,label)
        self.smooth(self.pos)
        self.smooth(self.neg)

    def smooth(self,m):
        for key in m:
            m[key] += self.smooth_weight

    def predict_label(self,review):
        b = math.log(self.pos_count/self.neg_count)
        comment = split(review,self.language)
        for i in range(len(comment)):
            txt = "".join(comment[i:i+self.gram_number])
            p_pos = math.log(self.pos.get(txt,self.smooth_weight)/self.pos_count)
            p_neg = math.log(self.neg.get(txt,self.smooth_weight)/self.neg_count)
            b = b + (p_pos - p_neg)
        if b > self.threshold:
            return 1
        else:
            return -1

    def predict(self,data):
        if self.language == "en":
            self.threshold = -5
        else:
            self.threshold = -2
        predict = [self.predict_label(item[1]) for item in data]
        return predict
