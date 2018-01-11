from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,Bidirectional,Dropout
from operator import itemgetter
from parser import split
import pickle

class Rnn():
    def __init__(self,config):
        self.head_word = config.get("rnn-head-word",100)
        self.tail_word = config.get("rnn-tail-word",1)
        self.oov_word = config.get("rnn-oov-word",2)
        self.start_word = config.get("rnn-start-word",1)
        self.start_from = config.get("rnn-start-from",2)
        self.max_len = config.get("rnn-max-len",80)
        self.max_feature = config.get("rnn-max-feature",30000)
        self.batch_size = config.get("rnn-batch-size",32)
        self.epoch = config.get("rnn-epoch",10)
        self.language = config.get("language","cn")
        self.model_name = config.get("rnn-model-name","model.h5")
        self.load_from_file = config.get("rnn-load-from-file",False)
        self.file_path = config.get("model-path","model/")
        self.indices = {}

    def save(self):
        model_path = self.file_path+self.language+"-rnn.model"
        model = [self.max_len,self.max_feature,self.batch_size,
                 self.model_name,self.indices]
        self.model.save_weights(self.file_path+self.model_name)
        pickle.dump(model,open(model_path,"wb"))

    def load(self):
        model_path = self.file_path+self.language+"-rnn.model"
        model = pickle.load(open(model_path,"rb"))
        self.max_len,self.max_feature,self.batch_size = model[:3]
        self.model_name,self.indices = model[3:]
        self.build_model()
        self.model.load_weights(self.file_path+self.model_name)

    def build_indices(self,data):
        word_count = {}
        for _,comment,_ in data:
            for word in split(comment,self.language):
                word_count[word] = word_count.get(word,0) + 1
        words = [(word_count[key],key) for key in word_count]
        words = sorted(words)
        length = len(words)
        for i in range(length):
            if i < self.head_word or length - i < self.tail_word:
                self.indices[words[i][1]] = self.oov_word
            else:
                self.indices[words[i][1]] = i + self.start_from - self.head_word

    def convert_to_labels(self,comment):
        return [self.indices.get(word,0) for word in split(comment,self.language)]

    def get_x_y(self,data):
        x = list(map(lambda x: x[1], data))
        try:
            y = list(map(lambda x: int((x[2] + 1)/2), data))
        except:
            y = None
        x = [self.convert_to_labels(comment) for comment in x]
        return x, y

    def build_model(self):
        model = Sequential()
        model.add(Embedding(self.max_feature,128))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model

    def train(self,train_set,test_set):
        self.build_indices(train_set)

        x_train, y_train = self.get_x_y(train_set)
        x_test, y_test = self.get_x_y(test_set)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        x_train = sequence.pad_sequences(x_train, maxlen=self.max_len)
        x_test = sequence.pad_sequences(x_test, maxlen=self.max_len)
        print('x_train shape:', x_train.shape)
        print('x_test shape:', x_test.shape)

        self.build_model()

        if self.load_from_file: return
        self.model.fit(x_train,y_train,
                  batch_size = self.batch_size,
                  epochs = self.epoch,
                  validation_data = (x_test,y_test))
        self.model.save_weights(self.file_path+self.model_name)

    def predict(self,evaluate_set):
        self.model.load_weights(self.file_path+self.model_name)

        x,_ = self.get_x_y(evaluate_set)
        x = sequence.pad_sequences(x, maxlen=self.max_len)
        result = self.model.predict(x, batch_size=self.batch_size,verbose=1)
        s = [-1 for _ in range(len(result))]
        for i in range(len(result)):
            if result[i][0] > 0.5:
                s[i] = 1
        return s
