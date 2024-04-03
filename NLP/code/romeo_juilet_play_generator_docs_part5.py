
# Note: This is the result of following tutorial strictly:
# link: https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-create-a-play-generator

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

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

# so we need to create a loss function that can compare that output to the expected output and give us some numeric value representing how close the two were.

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

"""
New tutorial starts from here
https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-training-the-model

"""

# Compiling the model
# At this point we can think of our prolem as a classification problem where predicts the probability of each unique letter coming next.
# Okay we are going to compile the model with adam optimizer and the loss function as loss, which we defined above as loss function 
model.compile(optimizer='adam', loss=loss)


"""

Creating CheckPoints

Now we are going to setup and configure our model to save checkpoint as it trains. This will allow us to load our model from a checkpoint 
and continue training it.

And now we're going to set up some checkpointts. I'm not going to talk about how these work, you can kind of just read through this if you
want. And then we're going to train the model. Remember to start your GPU hardware accelerator under runtime, change runtime type GPU, 
because if you do not then this is going to be very slow. But once you do that you can train the model, I've already trained it.

But if we go through this training, we can see it's gonna say train for 172 steps, it's gonna take about, you know, 30 seconds per epoch, 
probably maybe little bit less than that. And the more epochs you run this for, the better it will get, this is a different, we're not likely
going to overfit here.


"""

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_perfix = os.path.join(checkpoint_dir, "ckpt_(epoch)")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_perfix,
    save_weights_only = True
)

# Training : Finally we will start training the model
# if this is taking a while go to Runtime > Change Runtime Type and Choose GPU under hardware accelerator

history = model.fit(data, epochs=1, callbacks=[checkpoint_callback])

"""
So we can run this for like, say 100 epochs if we wanted to. For our case, let's actually start by just trianing this on, let's say two epochs,
just to see how it does, And then we'll train it on like 10, 20, 40, 50 and compare the results. But you'll notice the more epochs, the better
it's going to get. But just like for our case, we'll start with two and then we'll work our way up. 

So while that trains will actually explain the next aspect of this without running the code. So essentially, what we need to do, after we've 
trained the model, we've initialized the weights and biases, we need to rebuild it using an new batch size of one. 

So remember, the initial batch size was 64, which means that we'd have to pass it 64 inputs or sequences for to work properly. But now what 
I've done is I'm going to rebuild the model and change it to a batch size of one so that we can just pass it some sequence of whatever length
we want, and it will work.

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
So if we run this, we've rebuilt the model with batch size one, that's the only thing we've changed.

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))
And now what I can do is load the weights by saying model.load weights, tf.train.latest... and then build the model using the tensor shape one.
I know it sounds strange. This is how we do this rebuild the model. One, none is just saying expect the input one and then none means we don't
know what the next dimension length will be.

checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_perfix = os.path.join(checkpoint_dir, "ckpt_(epoch)")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_perfix,
    save_weights_only = True
)


But here, checkpoint directory is just we(ve defined where on our computer, we're gonna save these TensorFlow checkpoints. This is just saying 
this is the was the prefix we're gonna save the checkpoint with. So we're gonna do the checkpoint directory, and then checkpoint epoch where 
epoch will stand for obviously, whatever epoch we're on. 

So we'll save checkpoint here, we'll save a checkpoint at epoch one, a checkpoint two, to get the latest checkpoint, we do this. And then if 
we wanted to load any intermediate checkpoint, say, like checkpoint 10, which is what I've defined here, we can use this block of code down here.

And I've hardwired the checkpoint that I'm loading by saying TF.train.load_checkpoint,whereas this one just gets the most recent, so we'll get 
the most recent, which should be checkpoint two for me. And then what we're going to do is generate the text.

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# Once the model is finished training we can find the latest checkpoint that stores the models weights the following line
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# We can load any checkpoint we want by specifying the exact file to load 
checkpoint_num = 10

"""

# Loading the Model
# We'll rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one piece of text to the model and have it make a prediction.

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# Once the model is finished training we can find the latest checkpoint that stores the models weights the following line
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# We can load any checkpoint we want by specifying the exact file to load 
checkpoint_num = 10

"""

So this function, I'll dig into it in a second, but I just want to run and show you how this works. Because I feel like we've done a lot of work for 
not very many results right now. And I'm just gonna type in the string Romeo. And just show you that when I do this, we give it a second.
And it will actually geneate an output sequence like this. So we have 
predicted sequence 

romeows are,
She would never of Yough in once is confents,'d whereford,
I on not of my palisured, but is his made on that
And what consued flie the wide
not us so scere the comes traitor was thy crown:
This most oning pecpure,
Oram thou art us shall deint deat again:
cur device of the freening from whose befuraces in his
Kind shall pastingly.
Nor, if thou wart pengless of fugly.

CAMILLO:
Oven, Poou lame,
Let's this trunk, the sidy? let not, yeart iftees,
Over break himself,
Thund are to-morrow, the kelsemant.

First Senator:
Are as my father, scarion, procectis With starr dowZUTIO:
Evertisa, to-thank is then word, the more strunge
To do this signion, sight.

FORTERILA:
To the loot of fines are ploceed is asf strine?

AUTOLYCUS:
Then it ware, I mide I druak? Good in her,
They are thy volice and
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

>>> Above output is mostly like pseudo english. Most of it are like kind of proper words. But again, this is because we trained it on just 2 epochs,
So i'll talk about how we build this in a second. But if you wanted a better output for this part, then you would train this on on more epochs.

So now let's talk about how I actually generated that output. So we rebuilt the model to accept a batch size of one, which means that I can pass
it a sequence of any length, And in fact what I start doing (in the above I just typed romeo) is passing the sequence that I've typed in here, 
which was Romeo, then what that does is we run this function generate text, I just stole this from tensorflow website like I am stealing almost
all of this code.

And then we say the number of characters to generate is 800, 

# Generating the text
def generate_text(model, start_string):
    # Evaluate step (Generating text using the learned model)
    
    # Number of characters to generate
    num_generate = 800

--------------------------------------------------------------------------------------
input_eval = [char2idx[s] for s in start_string]

The input evaluation, which is now what we need to pre-proecss this text again so that 
this works properly, we could use my little function, or we can just write this line of code here, which does with the function that
I wrote does for us. So char2idx for s in start string is what we typed in, in that case, Romeo, 

then what we're going to do is expand the dimensions.  >>>  input_eval = tf.expand_dims(input_eval, 0)
so essentially turn just a list like this that has all these numbers, no 987 into a double list like this or just
>>>>>>>>>>>>     input_eval = [char2idx[s] for s in start_string]
                 [9,8,7]
a nested list bcoz that's what it's expecting as the input one batch one entry.

Then what we d is we're going to say the string that we want to store because we want to print this out at
the end right? We'll put in this text generated list, temperature equals one point out what this will allow us to is 
>>>>>>>>>> text_generated = []
>>>>>>>>>> temperature = 1.0

if we change the temp value to be higher, I mean, you can read the comment here, right low means temperature results in more predictable text,
higher temperature results in more surprising text so this is just a parameter to mess with if you want, you dont necessarily need it.

>>>>>> model.reset_states()
And I would like I've just left mine at one for now, we're gonna start by resetting the state of the mode. This is because when we rebuild the model,
it's gonna have stored the last state that it remembered when it was training. 

>>>>> for i in range(num_generate):

So we need to clear that before we pass new input text to it, we say for i in range num_generate, which means how many characters we want to generate,
which is 800. 

Here, what we're going to do is, say  >> predictions = model(input_eval), that's going to start as the start string that's going to start as the start string
that's encoded right. 
And then what we're going to do is say predictions equals TF.squeeze  >>>> predictions = tf.squeeze(predictions, 0),what this does is take our predictions,
which is going to be in a nested list that give us >>> [[[ ]]] list and just removes that exterior dimension.

So we just have the predictions that we want, we donot have that extra dimension that we need to index again. And then we're gonna say using a categorical
distribution to predict the character returned by the model.
>>>>    predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

That's what it writes here. We'll divide by the temperature, if it's one, that's not going to do anything. And we'll say predicted ID equals we'll sample whatever
the output was from the model, which is what this is doing.

>>>>>> input_eval = tf.expand_dims([predicted_id], 0)
>>>>>> text_generated.append(idx2char[predicted_id])

And then we're going to take that output, so the predicted ID, and we're going to add that to the input evaluation. And then we're going to say as text generate.append,
and we're going to convert the text that are integers now , back into a string, and return all of this.

Now, I know this semmes like a lot, again this is just given to us, by TensorFlow to you know create this aspect, you can read through the comments yourself, if you want 
to understand it more, but I think that was a decent, decent explanation of what this is doing. So yeah, that is how we can generate, you know, sequences using a RNN.
        
        
"""
# Generating the text
def generate_text(model, start_string):
    # Evaluate step (Generating text using the learned model)
    
    # Number of characters to generate
    num_generate = 800
    
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    [9,8,7]
    input_eval = tf.expand_dims(input_eval, 0)
    
    # empty string to store our results
    text_generated = []
    
    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0
    
    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        # we pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))
        
inp = "romeo"
print(generate_text(model, inp))
