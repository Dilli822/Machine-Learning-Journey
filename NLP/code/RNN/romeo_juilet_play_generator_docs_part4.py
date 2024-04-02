
# Note: This is the result of following tutorial strictly:
# link: https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-create-a-play-generator

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

# spliting input target into two hello to hell and ello 
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

# building models here 
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model 

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

"""
From Tutorial:
6:14
(64, 100, 65) # (btach_suze, sequence_length, vocab_size)
And that's going to be the probability of every one of those characters occuring right. That's what that does the last one for us. So obviously, our last dimension
is going to be 65. For the vocabulary size. 
This is a sequence length, and that's a batch, I just want to make sure this was really clear before we keep going. Otherwise, this can get very confusing very quickly.

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch) # ask our model for a prediction on our first batch of training data
    print(example_batch_predictions.shape, "# (btach_suze, sequence_length, vocab_size)" ) # print out the output shape


So what I want to do now is actually look at the length of the example batch predictions, and just print them out and look at what they actually are. So example batch
predictions is what happens when I use my model on some random input example, actually will the first one from my data set with when it's not trained. 
So I can actually use my model before it's trained wit random weights and random biases and parameters by simply using model. 

And then I can put little brackets like this and just pass in some example that I want to get a prediction for. So that's what I am going to do, I'm going to give it 
the first batch, and it can even it shows me thet shape of this btach 64 100, I'm going to pass that to the model. And it's gonna give us a prediction for that. And in fact,
it's actually gonna to give us prediction, for every single element in the batch, right, every single training example in the batch is going to give us a prediction for so
let's look at what those predictions are. 

So this is what we get, we get a length 64 tensor, right. And then inside of here, we get a list inside of a list or an array inside of an array with all these different
predictions. So we'll stop there for this, like explaining this aspect here. But you can see we're getting 64 predictions, because there's 64 elements in the batch. Now
let's look at one prediction.

64
tf.Tensor(
[[[ 1.56090898e-03 -6.47407025e-03 -6.31173979e-03 ... -2.04308867e-03
   -1.21397583e-03 -3.53259244e-03]
  [-5.14911371e-04 -4.85860836e-03 -8.70018080e-03 ... -4.17831121e-03
   -1.06203486e-03 -2.53723608e-03]
  [-6.06308971e-03 -7.21563003e-04 -1.11019646e-03 ... -3.88383260e-03
    5.99927036e-03 -2.79084733e-03]
  ...
  [-4.39675758e-03  7.66527955e-04 -3.00406548e-03 ...  1.38708428e-02
   -9.82107222e-03  7.81740155e-03]
  [-8.25484004e-03  4.21268749e-04  5.16259344e-04 ...  1.09678870e-02
   -1.21590029e-02  5.52886026e-03]
  [-9.06381570e-03 -1.40891457e-03  1.14349963e-03 ...  1.56909246e-02
   -1.09380390e-02  7.84927048e-03]]

So let's look at the very first prediction for say the first element in the batch right. So let's do that here. And we see now that we get a length 100 tensor. And that this 
is what it looks like there's still another layer inside. And in fact, we can see there's another nested layer here, right, another nested array inside of this array.
So the reason for this is because at every single time step, which means the length of the sequene, right?

Because remember, a recurrent neural network is going to feed one at a time, every word in the sequence. In this case, our sequences are like the 100. At every time step, we're
actually saving that output as a as a prediction, right, and we're passing that back. So we can see that for one batch one training, sorry, not one batch, one training we get 100
outputs. 
And these outputs are in some shape, we'll talk about what those are in a second. So that's something to remember that for every single training example, we get whatever the length
of that training example was outputs, because that's the way that this model works. And then finally we look at the prediction at just the very first time step. So this is 100
different time steps.

"""

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch) # ask our model for a prediction on our first batch of training data
    print(example_batch_predictions.shape, "# (btach_suze, sequence_length, vocab_size)" ) # print out the output shape



# we can see that the predictions is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)

# lets example one prediction 
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# and of course its 65 values representing the probability of each character occuring next

"""
So let's look at the first time step and see what the prediction is. And we can see that now we get a tensor of length 65. And this is telling us the
probability of every single character occuring next at the first time step, So that's what I want to walk through is showing you what's actually outputted
from the model the current way that it works. 

And that's why we need to actually make our own loss function to be able to determine how good are models performing , when it outputs something ridiculous that
look like this, because there is no just built-in loss function in tensorflow that can look at a three dimensional nested array of probabilities over you know
the vocabulary size and tell us how different the two things are. 

So we need to make our own loss function. So if we want to make our own loss function.

time_pred = pred[0]
print(len(time_pred))
print(time_pred)

65
tf.Tensor(
[-4.5214333e-03  1.4388280e-03  2.5028815e-03  1.0752274e-03
  2.9356158e-03 -6.4068967e-03 -1.8893488e-03 -2.7068045e-03
  1.5851962e-03 -2.0843773e-04 -2.5679297e-03 -4.8691719e-03
 -2.0612846e-03  7.3782937e-04 -8.0915583e-05 -2.2612279e-03
 -1.9452027e-03  5.2201482e-03 -5.6045284e-03 -1.3544702e-03
 -4.3902006e-03  2.6183301e-03 -1.6308895e-03 -2.9814637e-03
  1.0324984e-03 -9.9441106e-04 -3.6520907e-03  2.0815050e-03
 -2.0050444e-05  1.2187161e-03 -2.3339340e-03  6.3501147e-04
  1.3825553e-04  4.6385732e-03  2.7937253e-03  4.6120719e-03
 -9.9980249e-04 -1.3165772e-03 -1.9080117e-03  2.8080116e-03
 -3.1818768e-03 -3.3954252e-03 -2.6108916e-03  1.0523781e-03
  1.7173991e-03 -4.9942913e-03 -3.9403909e-04 -1.7715455e-03
 -2.3613132e-03  9.4550673e-04  3.9594416e-03  9.7980350e-04
 -2.2321143e-03  1.9360876e-03 -7.3161339e-03 -3.5568082e-05
  2.0343908e-03  5.4669864e-03 -1.1165510e-03  1.5985980e-03
  1.5585008e-04 -1.3686297e-03 -1.8977404e-03  3.9296914e-03
  5.9781377e-03], shape=(65,), dtype=float32)

"""

time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probability of each character occuring next

# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabilities)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1)) [0]
predicted_chars = int_to_txt(sampled_indices)

predicted_chars # this is what the model predicted for traning sequence 1
print(predicted_chars) # gives any random predicted values
"""
-----
the vocabulary size and tell us how different the two things are. 

So we need to make our own loss function. So if we want to make our own loss function to determing the predicted character, from this array, so what we'll go there now, what 
we can do is get the categorical with this. We can sample the categorical distribution. sample_indices = tf.random.categorical(pred, num_samples=1)
And that will tell us the predicted character. So what I mean is, let's just look at this, and then we'll explain this. So since our model works on random weights and biases.
right now we havenot trained yet. 

This is actually all of the predicted characters that it had. So at every time step, and the first time step, a predicted age, then a predicted - then h then g u and so on forth
you get the point, right? So what we're doing to get this value is we're going to sample the prediction. So at this, this is just the first time step actually, we're sample the
prediction. 

Actually, no, sorry we are sampling every time step, my bad there, we're gonna say sampled indices = np.reshapes is reshaping this just changing the shape of it, we are gonna say
predicted characters equals int to text sampled indices. So it's , I really, it's hard to explain all this, if you guys donot have a statistics kind of background a little bit to
talk about why we're sampling, and not just taking the argument max value of like this array, because you would think that what we'll do is just take the one that has the highest
probaility out of here from batch 65 output, and that will be the index of the next predicted  character.

There's some issuses with doing that for the last function, just because if we do that, then what that means is, we're going to kind of get stuck in an infinite loop almost where
we just keep accepting the biggest character. So what we'll do is pick a character based on this probability distribution. Kind of yeah, again it's hard it's called sampling 
distrbution, you can look that up if you don't know what that means.

But sampling is just like trying to pick a character based on a probability distribution, it doesnot guarantee that the character with the highest probability is going to be picked
it just uses those probabilities to pick it. I hope that makes sense. I know that was like a really not a good definition, but that's best I can do. So here we reshaped the array and 
convert all the integers to numbers to see the actual 

sampled_indices = np.reshape(sampled_indices, (1, -1)) [0]
predicted_chars = int_to_txt(sampled_indices)

predicted characters by showing you this. And you know, the character here is what was predicted at time step zero to be the next character, and so on. Okay, so now we can create
a loss function that actually handles this for us.

"""
# so we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were.

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)


"""
And you know, the character here is what was predicted at time step zero to be the next character, and so on. Okay, so now we can create
a loss function that actually handles this for us.

# so we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were.

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)
    
so this is the loss function that we have keras has like a built in one that we can utlize which is what we're doing. But what this is going to do is take all the labels and
all of the probability and we'll compute a loss on those.

So how different or how similar those two things are. Remember, the goal of our algorithm and the neural network is to reduce the loss are. Remeber the goal our algorithm and
the neural network is to reduce the loss.

"""