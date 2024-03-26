from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)
train_data[0]
len(train_data[0]) # every review either should have unique length
print("train data --> ", train_data[0])
print("train data length ---> ", len(train_data[1]))
print("train data type --> ", type(train_data[0]))
