"""
from Tutorial
https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-sentiment-analysis
"""
from keras.datasets import imdb
from keras.preprocessing import sequence, text
import tensorflow as tf
import numpy as np

VOCAB_SIZE = 88584
MAXLEN = 250

# Load IMDb dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# Pad sequences if necessary
train_data = sequence.pad_sequences(train_data, maxlen=MAXLEN)
test_data = sequence.pad_sequences(test_data, maxlen=MAXLEN)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

# Compile and train the model
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print("Final Model Evaluation result: ", results)

# Making Predictions
word_index = imdb.get_word_index()

def encode_text(input_text):
    tokens = text.text_to_word_sequence(input_text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], maxlen=MAXLEN)

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, MAXLEN))
    pred[0] = encoded_text
    result = model.predict(pred)
    return result[0]

# Example for a positive review
positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great "
positive_review_percentage = predict(positive_review)
print("Accuracy Values of Positive Review: ", positive_review_percentage )

# Example for a negative review
negative_review = "That movie sucked. I hated it and wouldn't watch it again. It was one of the worst things I've ever watched. "
negative_review_percentage = predict(negative_review)
print("Accuracy Values of Negative Review: ", negative_review_percentage)

negative_review_variation = "That movie wasn't too bad. I didn't quite like it and won't recommend it to others."
negative_variation_accuracy = predict(negative_review_variation)
print("Accuracy Values of Negative Review with Variation: ", negative_variation_accuracy)

if negative_variation_accuracy >= 0.5:
    print("Positive Review!")
else:
    print("Negative Review!")
