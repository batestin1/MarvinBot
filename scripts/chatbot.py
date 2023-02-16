from read_book import BookReader
import tensorflow as tf
import numpy as np
import random
import string
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ChatBot:

    def __init__(self, book_path, var_path):
        with open(var_path) as f:
            parameters = json.load(f)
        self.book = book_path
        self.book_reader = BookReader(self.book)
        self.text = self.book_reader.read_book()
        self.text = self.text.lower()
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))
        self.chars = sorted(list(set(self.text)))
        self.char_to_num = dict((c, i) for i, c in enumerate(self.chars))
        self.num_to_char = dict((i, c) for i, c in enumerate(self.chars))
        self.seq_length = 100
        self.step = 1
        self.sentences = []
        self.next_chars = []

        for i in range(0, len(self.text) - self.seq_length, self.step):
            self.sentences.append(self.text[i:i+self.seq_length])
            self.next_chars.append(self.text[i+self.seq_length])
        self.x = np.zeros((len(self.sentences), self.seq_length, len(self.chars)), dtype=bool)
        self.y = np.zeros((len(self.sentences), len(self.chars)), dtype=bool)

        for i, sentence in enumerate(self.sentences):
            for t, char in enumerate(sentence):
                self.x[i, t, self.char_to_num[char]] = 1
            self.y[i, self.char_to_num[self.next_chars[i]]] = 1
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(128, input_shape=(self.seq_length, len(self.chars))),
            tf.keras.layers.Dense(len(self.chars), activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def train(self, batch_size, epochs):
        self.model.fit(self.x, self.y, batch_size=batch_size, epochs=epochs)

    def generate_text(self, seed_text, num_chars):
        generated_text = seed_text

        for i in range(num_chars):
            x_pred = np.zeros((1, self.seq_length, len(self.chars)))
            for t, char in enumerate(seed_text):
                if char in self.char_to_num:
                    x_pred[0, t, self.char_to_num[char]] = 1.
            preds = self.model.predict(x_pred)[0]
            next_index = np.argmax(preds)
            next_char = self.num_to_char[next_index]
            generated_text += next_char
            seed_text = seed_text[1:] + next_char
        return generated_text

    def get_response(self, prompt):
        seed_text = prompt.lower()[-self.seq_length:]
        response = self.generate_text(seed_text, 100)
        return response
