import tensorflow as tf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow_datasets as tfds
import random

dataset, info = tfds.load("mnist", with_info = True)
print("dataset is --> ", dataset)
print("Info is -->", info)

# spliting into test and train
dataset_train = dataset['train']
dataset_train = dataset['test']

for i in dataset_train.take(1):
    print(i)
    
# pandas can load the data
filreade = pd.read_csv("annual-enterprise-survey-2023-financial-year-provisional.csv")
print(filreade)

print(len(filreade))

row, column = filreade.shape 
print("row -->", row , "column --> ", column)



weight = tf.Variable(initial_value=0.5, dtype=tf.float32)
bias = tf.Variable(initial_value=0, dtype=tf.float32)

x = tf.constant(1, dtype=tf.float32)
y = weight * x + bias 

inputs = None
for input in range(0,5):
    inputs = random.randint(1,1)
    print(inputs)
    
weights = None
for weight in range(0,5):
    # weights = random.randint(0,1)
    weights = random.randint(1,1)
    print(weights)
    
print(y)

weightsw = tf.Variable([inputs], dtype=tf.float32)
inputsi = tf.Variable([weights], dtype=tf.float32)
print(weights, inputs)

output = weightsw * inputs + bias
print(output.numpy())

# .take() 
# .random.randint(0,4)
# .random.uniform(0,3)
# .shape
