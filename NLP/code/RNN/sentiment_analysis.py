"""
from Tutorial
https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-sentiment-analysis
"""
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv

VOCAB_SIZE = 98584
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
history = model.fit(train_data, train_labels, epochs=5, validation_split=0.2)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print("Final Model Evaluation result: ", results)

# Making Predictions
word_index = imdb.get_word_index()

def encode_text(input_text):
    tokens = text_to_word_sequence(input_text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], maxlen=MAXLEN)

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, MAXLEN))
    pred[0] = encoded_text
    result = model.predict(pred)
    return result[0][0]

# List of reviews
reviews = [
    "That movie was so awesome! I really loved it and would watch it again because it was amazingly great.",
    "That movie sucked. I hated it and wouldn't watch it again. It was one of the worst things I've ever watched.",
    "The movie was terrible, I wouldn't recommend it.",
    "I just saw this movie tonight and it was one of the best I have ever seen. Simply amazing. It made GRAVITY look like a kiddie flick.",
    "It was an okay movie, nothing special but not bad either.",
]

# Evaluate each review and store results
review_accuracies = []
sentiments = []
colors = []

for review in reviews:
    review_accuracy = predict(review)
    review_accuracies.append(review_accuracy)
    
    if review_accuracy > 0.5:
        sentiment = "Positive"
        color = 'green'
    elif review_accuracy < 0.5:
        sentiment = "Negative"
        color = 'red'
    else:
        sentiment = "Neutral"
        color = 'blue'

    sentiments.append(sentiment)
    colors.append(color)

# Plot histogram
plt.figure(figsize=(12, 8))
bar_positions = range(len(reviews))
plt.bar(bar_positions, review_accuracies, color=colors)
plt.xlabel('Review Index')
plt.ylabel('Accuracy')
plt.title('Sentiment Analysis of Reviews')

# Set x-tick labels to review indices
plt.xticks(bar_positions, [f"Review {i+1}" for i in bar_positions])

# Annotate each bar with the sentiment
for i, review_accuracy in enumerate(review_accuracies):
    plt.text(i, review_accuracy + 0.02, f"{sentiments[i]} ({review_accuracy * 100:.2f}%)", ha='center', va='bottom')

plt.ylim(0, 1)
plt.tight_layout()

# Position reviews just below the plot
for i, review in enumerate(reviews):
    percentage = review_accuracies[i] * 100
    plt.figtext(0.1, 0.02 + 0.05 * (len(reviews) - i), f"Review {i+1} ({percentage:.2f}%): {review}", fontsize=10, ha='left')

plt.subplots_adjust(bottom=0.35)
plt.show()

# Open a CSV file to write the results
with open('review_predictions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Review", "Accuracy", "Sentiment"])

    # Evaluate each review
    for review in reviews:
        review_accuracy = predict(review)
        sentiment = "Positive" if review_accuracy >= 0.5 else "Negative"
        writer.writerow([review, review_accuracy[0], sentiment])
        print(f"Review: {review}\nAccuracy: {review_accuracy}\nSentiment: {sentiment}\n----------")