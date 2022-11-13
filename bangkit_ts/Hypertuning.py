import  pandas as pd
import nltk
import re
import time
import numpy as np

from numpy.random import seed
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd

class Hypertuning:
    def __init__(self, filename, text_column, target_column):
        self.df = pd.read_excel(filename)
        self.df_text = self.df[text_column]
        self.embedding_dim = 50
        self.random_state = 113
        self.y = self.df[target_column].values
        self.X = None
        self.vocab_size = 0
        self.maxlen = 0


    def tokenizing(self):
        tokenizer = Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.df_text)
        tokenizer.index_word
        tokenizer.texts_to_sequences(self.df_text)
        
        self.vocab_size = len(tokenizer.index_word) +1
        self.maxlen = 20

        X = tokenizer.texts_to_sequences(self.df_text)
        X = pad_sequences(X, maxlen = self.maxlen, padding='post')

        return X
        

    def embedding(self):
        vectorizer = CountVectorizer(min_df=0, lowercase=False)
        vectorizer.fit(self.df_text)
        vectorizer.vocabulary_vektor=vectorizer.transform(self.df_text).toarray()
    
    
    def fold(self):
        kf = KFold(n_splits = 10, shuffle = True, random_state=self.random_state)
        
        X = self.tokenizing()
        
        for train_index, test_index in kf.split(X, self.y) :
            X_train, X_test = X[train_index], X[test_index] 
            y_train, y_test = self.y[train_index], self.y[test_index]
        
        return X_train, X_test, y_train, y_test
    
    def get_model(self, batch_size=64, units=128, dropout=0.2):
        self.tokenizing()
        self.fold()
        
        model_lstm = Sequential()
        model_lstm.add(Embedding(self.vocab_size, self.embedding_dim, input_length = self.maxlen))
        model_lstm.add(BatchNormalization())
        model_lstm.add(Dropout(dropout))
        model_lstm.add(LSTM(units=128))
        model_lstm.add(Dense(32, activation='relu'))
        model_lstm.add(Dense(1, activation='sigmoid'))
        model_lstm.compile(optimizer = 'adam', 
                            loss = 'binary_crossentropy',
                            metrics=['accuracy'])
        return model_lstm

    def run(self):
        X_train, X_test, y_train, y_test = self.fold()

        monitor = EarlyStopping(monitor='val_loss', mode='min', patience=5)
        
        model_lstm = KerasClassifier(build_fn=self.get_model,
                                    callbacks=[monitor],
                                    verbose=1)
        
        params_lstm =   {
                            'epochs' : [1, 2],
                            #'batch_size': [32, 64, 128],
                            #'units': [32, 64, 128],
                            #'dropout' : [ 0.2 ]
                        }

        ht_lstm = GridSearchCV(model_lstm, param_grid = params_lstm,
                            scoring = 'accuracy',
                            cv = KFold(n_splits=2))
        
        hyt_lstm = ht_lstm.fit(X_train, y_train, validation_data=(X_test, y_test))

        print("\nThe best parameter is {}".format(hyt_lstm.best_params_))
        print("\nThe best_accuracy is {}".format(hyt_lstm.best_score_))