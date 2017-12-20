import math
import re

def count_function(l):
    # count the number of xs in l
    return lambda x: len(list(filter(lambda item: (item == x),l)))

class Bayesian():
    def __init__(self,config):
        self.gram_number = config.get("bayesian-gram-number",3)
        self.smooth_weight = config.get("bayesian-smooth-weight",1)
        self.language = config.get("language","en")
        self.pos = {}
        self.neg = {}
        self.pos_count = 0
        self.neg_count = 0

    def add_comment(self,m,comment):
        if self.language == "en":
            comment = re.split(r'[\.,;!?\'\s]+',comment)
        for i in range(len(comment)):
            txt = comment[i:i+self.gram_number]
            if self.language == "en":
                txt = "".join(txt)
            if m.get(txt):
                m[txt] += 1
            else:
                m[txt] = 1

    def train(self,data,_):
        for _,comment,label in data:
            if label == 1:
                self.add_comment(self.pos,comment)
                self.pos_count += 1
            else:
                self.add_comment(self.neg,comment)
                self.neg_count += 1

    def smooth(self,m):
        for key in m:
            m[key] += self.smooth_weight

    def predict_label(self,comment):
        if self.language == "en":
            comment = re.split(r'[\.,;!?\'\s]+',comment)

        p_pos = math.log(self.pos_count/(self.pos_count+self.neg_count))
        p_neg = math.log(self.neg_count/(self.pos_count+self.neg_count))
        for i in range(len(comment)):
            txt = comment[i:i+self.gram_number]
            if self.language == "en":
                txt = "".join(txt)
            p_pos += math.log(self.pos.get(txt,self.smooth_weight)/self.pos_count)
            p_neg += math.log(self.neg.get(txt,self.smooth_weight)/self.neg_count)
        if p_pos > p_neg:
            return 1
        else:
            return -1
        self.smooth(self.pos)
        self.smooth(self.neg)

    def predict(self,data):
        predict = [self.predict_label(comment) for _,comment,_ in data]
        result = [(predict[i]/2 + 1.5  + data[i][2]) for i in range(len(data))]
        count = count_function(result)
        precision = count(3)/(count(3)+count(1))
        recall = count(3)/(count(3)+count(2))
        accuracy = (count(3)+count(0))/len(data)
        f1 = 2*precision*recall/(precision+recall)
        return predict, accuracy, f1 , [precision, recall]
