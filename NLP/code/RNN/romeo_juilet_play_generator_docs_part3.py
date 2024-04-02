

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

# Building the Model
"""
Now it is time build the model. we will use an embedding layer a LSTM and one dense layer that 
contains a node for each unique character in our training data. The dense layer will give us a probability
distribution over all nodes.
"""

"""
function build model return to us a model. The reason for this is because, right now, we're going
to pass the modelbtaches of size 64, for training, right.But we're going to do later is save this 
model. And then we're going to batch pass it batches of one pieces of you know trainig whatever data so that
you can actually make a prediction on just one piece of data.

Because for right now, what it's going to do is takes a batch size of 64, it's going to take 64 training examples,
and returned to a 64 outputs. That's what this model is going to be built to do the way we build it now to start.

But later on, we're going to rebuild the model using the same parameters that we've saved and trained for the 
model. But change it just be a batch size of one. So that way, we can get one prediction for one input sequence, right.
So that's why I'm creating this build model function. 

Now in here, it's going to have the vocabulary sizes first argument, the embedding dimension, which remember was 256
as a second argument, but also these are the parameters up here, right? 
And then we're going to find the batch size, as you know, batch size none. What this none means is we donot know
how long the sequences are going to be in each batch. 

All we know is that we're going to have 64 entries in each batch. and then of those 64 entries. so training examples,
right, we donot know how long each one will be , although in our case, we're going to use ones that are length 100.
But when we actually use the model to make preditions, we don't know how long the sequence is going to be that we input,
so we leave this none. 

Next, we'll make an LSTM layer which is long short term memory RNN units, which is 1024, which again, I don't really wanna
explain, but you can look up if you want return sequences means return the intermediate stage at every step.
The reason we're doing this, is because we want to look at what the model seeing at the intermediate steps and not just the 
final stage. So if you leave this 
tf.keras.layers.LSTM(rnn_units,
                             return_sequence=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
                             
False, and you don't se this to true, what happens is this LSTM just returns one output, that tells us what the model kind of
found at the very last time. But we actually went the output at every single time step for this specific model. 

And that's why we're setting this true stateful, I'm not going to talk about that one right now. That's something 
you can look up if you want and then recurrent initializer is just what these values are going to start at in the LSTM.

We are just picking this because this is what TensorFlow is kind of said is a good default to pick, I won't go into more depth
about that again, things that you can llok up more if you want. Finally, we have a dense layer, which is going to contain the 
amount of vocabulary size notes. 

The reason we're doing this is because we want the final layer to have the amount of nodes in it equal to the amount of 
charaters in the vocabulary. This way, every single one of those nodes can represent a probability distribution, the that character
comes next. So, all of those nodes value some sum together should give us the value of one. 

And that's going to allow us to look at that last layer as a prediction layer where it's telling us the probability that these
characters comes next, we've discussed how that's worked previously with oter neural networks. So let's run this now.


"""
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
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (64, None, 256)           16640     
                                                                 
 lstm (LSTM)                 (64, None, 1024)          5246976   
                                                                 
 dense (Dense)               (64, None, 65)            66625     
                                                                 
=================================================================
Total params: 5330241 (20.33 MB)
Trainable params: 5330241 (20.33 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
So if we look at the model summar, we can see we have our initial embedding layer, we have our
LSTM. And then we have our dense layer at the end. 

Now notice 63 is the batch size right ? that's the initial shape, none is the length of sequence,
which we don't know and then this is going to be just the output dimension, or sorry this is the amount 
of values in the vector, right we're gonna start with 256, we'll just do 1024 units in the LSTM. And then 
65 of nodes because that is the length of the vocabulary. 

All right, so combined, that's how many trainable parameters we get. You can see each of them for each layer.
And now it's time to move on to the next section.

"""

# Creating a Loss Function 
"""
Now we are actually going to create our own loss function for this problem. This is because our model will output 
a (64, sequence_length, 65) shaped tensor that represents the probability distribution of each timestep for every
sequence in the batch.
However, before we do that let's have a look at a sample input and the output from our untrained model. This is so 
we can understand what the model is actaully giving us. 

"""

"""
From Tutorial

Okay we are moving on to the next step of the tutorial which is creating a loss function to compile our model with. 
Now I'll talk about why we need to do this in a second. But I first want to exlore the output shape of our model.
So remember, the input to our model is something that is of length 64,because we're going to have batches of 64 
training examples, right?

So every time we feed our model, we're going to give it 64 training examples. Now with those training examples are
our size consists of lenth 100. That's what I want you to remember, we're passing 64 entries that are all of length
100 into the model as its training data, right? 

But sometimes, and when we make predictions with the model, later on, we'll be passing it, just one entry that is of 
some variable length, right. And that's why we've created this build model function, So we can build this model using
the parameters that we've saved later on, once we train the model and it can expect a different input shape, right, 
actually testing with it. 

Now, what I want to do is explore the output of this model, though, at the current point in time. So
we've created a model that accepts a batch of 64 training examples that are length 100. So let's just look at what the
output is from the final layer, give this a second to run, we get (64, 100, 65)  
we get (64, 100, 65):
And that represents the batch_size, the sequence length, and the vocabulary size. Now the reason for this is we have to 
remember that when we create a dense layer as our last layer that has 65 nodes, every prediction is going to contain 65 
numbers. 
up to 6:18

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
