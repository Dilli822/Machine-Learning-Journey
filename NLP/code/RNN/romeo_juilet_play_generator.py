from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = { u:i for i, u in enumerate(vocab) }
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])
    
text_as_int = text_to_int(text)

print("Text: ", text[:13])
print("Encodede: ", text_to_int(text[:13]))

import matplotlib.pyplot as plt

# Visualize the text
plt.figure(figsize=(10, 6))
plt.scatter(range(len(text)), [ord(char) for char in text], color='blue', s=1)
plt.title('Visualization of Text')
plt.xlabel('Index')
plt.ylabel('Character (ASCII Value)')
plt.grid(True)
plt.show()

# Visualize the encoded text
plt.figure(figsize=(10, 6))
plt.scatter(range(len(text_as_int)), text_as_int, color='green', s=1)
plt.title('Visualization of Encoded Text')
plt.xlabel('Index')
plt.ylabel('Encoded Value')
plt.grid(True)
plt.show()

# Visualize the length of the text
plt.figure(figsize=(10, 6))
plt.hist([len(line) for line in text.split('\n')], bins=50, color='orange')
plt.title('Distribution of Line Lengths')
plt.xlabel('Line Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
