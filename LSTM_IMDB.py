import tensorflow as tf
import os
import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
import encode_text, decode_integers, predict


VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(trainData, trainLabels), (testData, testLabels) = imdb.load_data(num_words=VOCAB_SIZE)
print(trainData[0])

# Change the length of the sequences to an equal MAXLEN length:
trainData=sequence.pad_sequences(trainData, MAXLEN)
testData = sequence.pad_sequences(testData, MAXLEN)

# Building the MODEL:
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(trainData, trainLabels, epochs=10, validation_split = 0.2)

results = model.evaluate(testData, testLabels)
print(results)

# Encode the words ti integer: (their index in imdb dataset)
word_index = imdb.get_word_index()


text = "that movie was just amazing, so amazing"
encoded = encode_text(text, word_index, MAXLEN)
print(encoded)

# Decoe The words by their index:
# while were at it lets make a decode function

reverse_word_index = {value: key for (key, value) in word_index.items()}

print(decode_integers(encoded, reverse_word_index))

# now time to make a prediction

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review, model)

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review, model)
