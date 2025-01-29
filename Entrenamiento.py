#!/usr/bin/env python3
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np

class Entrenamiento:
    
    def __init__(self, X_train_test, y_train, max_words=5000, max_sequence_length=100):
        # Para modelos tradicionales (SVM, Logística, etc)
        self.vectorizer = TfidfVectorizer()
        self.X_train_tfidf = self.vectorizer.fit_transform(X_train_test)

        # Variables para la RNN Y CNN
        self.X_train_text = X_train_test # Para la RNN Y CNN
        self.max_words = max_words
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words = self.max_words)
        self.tokenizer.fit_on_texts(X_train_text)
        self.X_train_seq = self.tokenizer.texts_to_sequences(X_train_test)
        self.X_train_seq = pad_sequences(self.X_train_seq, maxlen = self.max_sequence_length)

        self.y_train = y_train

    def entrenar_svm(self, kernel: str, probability: bool):
        """Entrenar SVM usando los datos transformados por TF-IDF"""
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError("[!] Kernel no válido")
        
        svm_model = SVC(kernel = kernel, probability = probability)
        svm_model.fit(self.X_train_tfidf, self.y_train)
        
        return svm_model

    def entrenar_regresion_logistica(self):
        """Entrenar Regresión Logística usando los datos transformados por TF-IDF"""
        logistic_model = LogisticRegression(max_iter = 1000)
        logistic_model.fit(self.X_train_tfdif, self.y_train)
        
        return logistic_model

    def entrenar_bayes(self):
        """Entrenar Naive Bayes usando los datos transformados por TF-IDF"""
        nb_model = MultinomialNB()
        nb_model.fit(self.X_train_tfidf, self.y_train)
        
        return nb_model

    def entrenar_decision_tree(self, max_depth = None):
        """Entrenar Árbol de Decisión usando los datos transformados por TD-IDF"""
        dt_model = DecisionTreeClassifier(max_depth = max_depth)
        dt_model.fit(self.X_train_tfidf, self.y_train)

        return dt_model

    def entrenar_rnn(self, num_classes, epochs = 10, batch_size = 32, units = 64):
        """Entrenar RNN usando las secuencias de texto"""
        model = Sequential()
        model.add(Embedding(input_dim = self.max_words, output_dim = 64, input_length = self.X_train_seq.shape[1]))
        model.add(SimpleRNN(units = units, activation="tanh"))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        model.fit(self.X_train_seq, self.y_train, epochs=epochs, batch_size=batch_size)

        return model

    def entrenar_cnn(self, num_classes, epochs = 10, batch_size = 32, filters = 128, kernel_size = 3, pool_size = 2):
        """Entrenar CNN usando las secuencias de texto"""
        model = Sequential()
        model.add(Embedding(input_dim = self.max_words, output_dim = 128, input_length = self.X_train_seq.shape[1]))
        model.add(Conv1D(filters = filters, kernel_size = kernel_size, activation = "relu"))
        model.add(MaxPooling1D(pool_size = pool_size))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dense(num_classes, activation="softmax"))

        model.compila(optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
        model.fit(self.X_train_seq, self.y_train, epochs=epochs, batch_size=batch_size)

        return model
