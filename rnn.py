from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from operator import itemgetter

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
        self.model_name = config.get("rnn-model-name","model.h5")
        self.load_from_file = config.get("rnn-load-from-file",False)
        self.indices = {}

    def build_indices(self,data):
        word_count = {}
        for _,comment,_ in data:
            for word in comment:
                word_count[word] = word_count.get(word,0) + 1
        words = [(word_count[key],key) for key in word_count]
        words = sorted(words)
        length = len(words)
        for i in range(length):
            if i < self.head_word or length - i < self.tail_word:
                self.indices[words[i][1]] = self.oov_word
            else:
                self.indices[words[i][1]] = i + self.start_from - self.head_word

    def convert_to_label(self,word):
        return self.indices.get(word,0)

    def get_x_y(self,data):
        x = list(map(lambda x: x[1], data))
        y = list(map(lambda x: int((x[2] + 1)/2), data))
        x = [list(map(self.convert_to_label,comment))
             for comment in x]
        return x, y

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

        model = Sequential()
        model.add(Embedding(self.max_feature,128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        self.model = model

        if self.load_from_file: return
        self.model.fit(x_train,y_train,
                  batch_size = self.batch_size,
                  epochs = self.epoch,
                  validation_data = (x_test,y_test))
        self.model.save_weights("model/"+self.model_name)

    def predict(self,evaluate_set):
        self.model.load_weights("model/"+self.model_name)

        x,_ = self.get_x_y(evaluate_set)
        x = sequence.pad_sequences(x, maxlen=self.max_len)
        result = self.model.predict(x, batch_size=self.batch_size,verbose=1)
        s = [-1 for _ in range(len(result))]
        for i in range(len(result)):
            if result[i][0] > 0.5:
                s[i] = 1
        return s
