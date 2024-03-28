"""
Continue from 6:15 Tutorial
https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-sentiment-analysis

"""
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


"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, None, 32)          2834688   
                                                                 
 lstm (LSTM)                 (None, 32)                8320      
                                                                 
 dense (Dense)               (None, 1)                 33        
                                                                 
=================================================================
Total params: 2843041 (10.85 MB)
Trainable params: 2843041 (10.85 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
Model Summary  None

From Tutorial:
We can look at the fact that the embedding layer actually has the most amount of parameters, because essentially
it's trying to figure out, you know, all these different numbers, how we can convert that into a tensor
of 32 dimensions, which is not the easy to do. And this is going to be the major aspect 2834688  that's being trained.
And then we have our LSTM layer, we can see the parameters there. And our final dense layer, which is getting 33 
parameters, that's because the output from every single one of these dimensions 32 + bias node, right that we need.

So that's what we'll get there. You can see modelled summary, we get the sequential model. Okay so training, alright 
so now it's time to compile and train the model, you can see I've already trained mine.

----- ONLY FOR NOTEBOOK OR GOOGLE COLAB -------
What I'm gonna say here is if you want to speed up your training, because this will actually take a second, and we'll talk
about why we pick these things in a minute is go to runtime, change runtime type and add a hardware accelerator of GPU.
What this will allow you to do is utilize a GPU while you're training, which should speed up your trainin about 10 to 20 
times.

So I probably should have mentioned that beforehand. But you can do that. And please do for these examples. So model
compile.

Alright, so we're compiling our model, we're picking the loss function as a binary cross entropy. The resons we're picking 
this is because this is going to essentially tell us how far away. We are from the correct probability right, because we have 
two different things we could be predicting. So you know either zero or one, so positive or negative. 
So this will give us a correct loss for that kind of problem that we've talked about before. 

The optimizer we're gonna use RMS Prop again, I am not going to discuss all the different optimizers, you can look them
up if you care that much about what they do. 

And we're going to us metrics as ACC, one thing I will say is the optimizer is not crazy important. For this one, you can use 
adm if you want to do and it would still work fine. My usual go to is just use the adam optmizer unless you think there's a
better one to use. But anyways, that's something to mention. 

Okay, so finally we will fit the model, we've looked at the syntax a lot before. So model fit, will give the training data,
the training labels, the epochs, and we'll do a validation split of 20%, such as 0.2 stands for, which means that what we're going
to be doing is using 20% of the training data to actually evaluate and validate the models we go through. 

And we can see that after training, which I've already done and you guys are welcome to obviously do on your own 
computer, we kind of stalled out an evaluation accuracy of about 88%, whereas the models actually gets overfit to 
about 97.98%.
SO what this is telling us essentially, is that we don't have enough training data and that after we've even done
just one epoch, we've pretty nuch stuck on the same validation accuracy, and that there's something that needs to change 
them all to make it better.

But for now, that's fine. We'll leave it the way it is. 

"""

# Training Magic Happens here. Compile and train the model
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)


"""
OKay, so now we can look at the results. I've already did the results here, just to again,
speed up some time. But we'll do the evaluation on our test data and test labels to get a more accurate kind of result here.
And that tells us where we have an accuracy of about 85.5 % which you know, isn't great but it's decent 
considering that we didnot really write that much code to get to the point that we're at right now.

"""

results = model.evaluate(test_data, test_labels)
print("Final Model Evaluation result: ", results)



import matplotlib.pyplot as plt

# Plot accuracy for each epoch
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(history.history['acc']) + 1),
    history.history['acc'], label='Training Accuracy', marker='o')
plt.plot(range(1, len(history.history['val_acc']) + 1), 
    history.history['val_acc'], 
    label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy for Each Epoch')
plt.legend()

# Plot loss for each epoch
plt.subplot(1, 2, 2)
plt.plot(range(1, len(history.history['loss']) + 1), 
    history.history['loss'], label='Training Loss', marker='o')
plt.plot(range(1, len(history.history['val_loss']) + 1), 
    history.history['val_loss'], label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Epoch')
plt.legend()

plt.tight_layout()
plt.show()


# Plot training and validation accuracy
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()
