

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = { u:i for i, u in enumerate(vocab) }
idx2char = np.array( vocab )

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

def int_to_txt(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

seq_length = 100
exmples_per_epoch = len(text) // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_txt(x))
    print("\nOUTPUT")
    print(int_to_txt(y))

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

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
