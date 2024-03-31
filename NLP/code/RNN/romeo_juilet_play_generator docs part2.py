
# Note: This is the result of following tutorial strictly:
# link: https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-create-a-play-generator

"""
>>> RNN Play Generator <<<<<
This guide based on the following:https://www.tensorflow.org/tutorials/text/text_generation
-------------------------------------------------------------------------------------------------------
"""

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# READ CONTENT OF FILE
# LET'S LOOK AT THE CONTENTS OF THE FILE.
# READ, THEN DECODE FOR PY2 COMPAT 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of the characters in it
print('Length of text: {} characters '.format(len(text)))
# Take a look at the first 250 characters in text
print(text[:250])

vocab = sorted(set(text))
# Creating a mapping from unique characters to indices 
char2idx = { u:i for i, u in enumerate(vocab) }
idx2char = np.array( vocab )

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

print("Text: ", text[:13])
print("Encodede: ", text_to_int(text[:13]))

""" 
From Tutorial 6:35

Now, I figured while we were at it, we might as well write a function that goes the other way. So into
text reason I'm trying to convert this to a NumPy Array first is just because we're going to be passing 
in different objects potentially in here. 

So if it's not already a NumPy array, it needs to be a NumPy Array, which is kind of what this is doing
otherwise, we're just going to pass on that we don't need to convert it to a NumPy array, if it already 
has one, we can just join all of the characters from this list into here. So that's essentially what this
is doing for us it just joining into text.

And then we can see if we go into text text is int:13. That translates that back to us for citizen. I mean,
you can look more into this function if you want, but it's not that complicated. 

"""

def int_to_txt(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

print("int to integer --> ", int_to_txt(text_as_int[:13]))

"""
Creating Training Example:

Remember our task is to feed the model as sequence and have it return to us the next character. This means 
we need to split our text data from above into many shorter sequences that we can pass to the model as 
training examples.

The training examples we will prepare will use a seq_length sequences as input and a seq_length sequence as 
the output where the sequence is the original sequence shifted one letter to the right. For example
input: Hell | output: ello
Our first step will be to create a stream of characters from our text data.

"""

"""
From tutorial:

Okay so now that we have all this text  encoded ad integers, what we need to do is create some training 
examples, it's not really feasible to just pass the entire, you know 1.1 million characters to our model
at once for training, we need to split that up into something that's meaningful. 

So what we're actually going to be doing is creating training examples where we have B first where the training
input, right,so the input value is going to be some sequence of some length, we'll pick the sequence length, in this 
case, we're actually going to pick 100 and then the output or the expected output, So I guess, like the label
for that training example is going to be the exact same sequence shifted right by 1 character.

input: Hello | Output: ello

So essentially, I put a good example here, our input will be something like hell, right. Now our output will
be ello. so what it's going to do is predict this last character, essentially, and these are what our training
examples are going to look like.

So the entire begining sequence, and then the output sequence should be that beginning sequence minus the first
letter, but track on what the last letter should be. So that this way, we can look at some input sequence and then
predict that output sequence that you know, plus a character right. 

seq_length = 100 # length of sequence for a training example
exmples_per_epoch = len(text)//(seq_length)

Okay, so that's how that works . Um, so now we're gonna do is define a sequence length of 100, we're gonna say the 
amount of example per epoch is going to be the length of the text divided by the sequence length plus one.

The reason we're doing this is because for every training example, we need to create a sequence input that's 100 
characters long. And we need to create a sequence output that's 100 characters long, which means that we need to 
have 101 characters that we use for every training example, right? 

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

Hopefully, that would make sense. So what this next line here is going to do is convert our entire string data set
into characters. And it's actually going to allow us to have a stream of characters, which means that it's going to 
essentially contain, you know 1.1 million characters inside of this tf dataset object from tensor slices. 
That's what that's doing.

Next,so let's run this and make sure this works alright.

"""

seq_length = 100 # length of sequence for a training example
exmples_per_epoch = len(text)//(seq_length)

# create training examples / targets 
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
"""
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
Alright, what we're gonna do is, say sequences is equal to char dataset.batch sequece length of each batch. So in 
this case, we want to one and then drop remainder means let's say that we have, you know 105 characters in our text.
Well, since sequences of length 101, we'll just drop the last four characters of our text because we can't put those
into batch.

So that's what this is doing for us is going to take our entire character dataset here that we've created, and batch 
it into length of 101, and then just drop the remainder. So that's what we're going to do here, sequences does.

"""
sequences = char_dataset.batch(seq_length +1, drop_remainder = True)

"""

Now split input target. What this is going to essentially is just create those training examples that we needed. So 
taking this these sequences of 101 length, and converting them into the input and target text, and I'll show you how 
they work in a second, we can do this convert the sequences to that by just mapping them to this function 
dataset = sequence.map(split_input_target) 

so that's what this function does. So if we say sequences.amp and we put this function here,that means every single 
sequence will have this operation appiled to it. And that will be stored inside this dataset obj. Or I guess you'd 
say object. 

But we also just say that's it's going to be you know the variable, right. So if we want to look at an example of how 
this works, we can kind of see so it just says example. The input will be first citizen. Before we proceed any further
hear me speak, all speak look the output:
EXAMPLE 

INPUT
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You

OUTPUT
irst Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You 


 EXAMPLE 

INPUT
are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you 

OUTPUT
re all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you k

"""

# Now we need to use these sequences of length 101 and split them into input and output
def split_input_target(chunk): # for the example: hello
    input_text = chunk[:-1] # hell
    target_text = chunk[1:] # ello
    return input_text, target_text

dataset = sequences.map(split_input_target) # we use map to apply the above function to every entry

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_txt(x))
    print("\nOUTPUT")
    print(int_to_txt(y))


"""
We need to create a training batches batch size equals to 64. The vocabulary size is the length of the 
vocabulary, which if you remember all the way back up to the top of the code was the set are the sorted set
of the text, which essentially told us how many unique characters are in there, the embedding dimension is 
256, then the RNN units is 1024.

And the buffer size is 10,000. What we're going to do now is create a dataset that shuffles we're going to 
switch around all these sequences so they don't get shown in the proper order, which we actually don't want.

And we are going to batch them by the batch size. So if we havenot kind of gone over what batching, and all 
this does before. I mean you can read these comments as a straight from the tensorflow docs, what we want to 
do is feed our models 64 batches of data at a time. so what we are going to do is shuffle all the data. batch
it into that size and then again drop the remainder if there's not enough batches, which is what we'll do.

We're going to define the embedding dimension, which is essentially how big we want every single vector to 
represent our words are in the embedding layer, and then the RNN units. I won't really discuss what that is 
right now. but that's essentially how many it's hard to really, I'm just gonna omit describing that for right
now because I don't want to butcher an explanation. It's not that important. anyways.okay.


"""

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab) # vocab is the number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinte sequences
# so it doesnot attempt to shuffle the entire sequence in memory, instead,
# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder = True)

