# Note: This is the result of following tutorial strictly:
# link: https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-create-a-play-generator

"""
>>> RNN Play Generator <<<<<
Now time for one of the coolest examples we've seen so far. We are going to use a RNN to generate a play. We will simply
show the RNN an example of something we want it to recreate and it will learn how to write a version of it on its own. 
We'll do this using a character predictive model that will take as input a variable length sequence and predict the next
character. We can use the model many times in a row with the output from the last predictions as the input for
the next call to generate a sequence.

This guide based on the following:https://www.tensorflow.org/tutorials/text/text_generation
-------------------------------------------------------------------------------------------------------
From Tutorial:
So now we're on to our last and final example, which is going to be creating a RNN Play generator this is going to be 
the first kind of neural network we've done, that's actually going to be creating something for us. But essentially
that's capable of predicting the next charcater in sequence.

So we're going to give it some sequence as an input. And what it's going to do is just simply predict the most likely 
next character. Now there's quite a bit that's going to go into this, but the way we're going to use this to predict a
play is we're going to train the model on a bunch of sequences of texts from the play Romeo and 
Juilet. And then we're going to have it so that will ask the model, we'll give it some starting prompt some string to 
start with. 

And that'll be the first thing we pass to it, it'll predict to us what the most likely next character for that sequence
is. And we'll take the output from the model and feed it as the input again to the model and keep predicting sequence of 
characters.

So keep predicting the next character from the previous output as many times as we want to generate an entire play. So 
we're gonna have this neural network that's capable of predicting one letter at a time, actually end up generating 
an entire play for us by running it multiple times on the previous output from the last iteration.

Now that's kind of the problem. That's what we're trying to solve. So let's go ahead and get into it and talk about
about what's involved in doing this. So the first thing we're going to do, obviously, is our imports. So from keras
pre-processing import sequence, import keras we need TensorFlow NumPy and on. 

So we'll load that in. And now what we're going to do is download the file, so the dataset for Romeo and Juilet,
which we can get by using this line here. 

"""

from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np


"""
DataSet:
For this example, we only need one piece of training data. In fact we can write our own poem or play and pass that 
to the network for training if we'd like. However, to make things easy we'll use an extract from a shakesphere play.


From tutorial:

For this example we only need one piece of training data. In fact we can write our own poem or play
Now that's kind of the problem. That's what we're trying to solve. So let's go ahead and get into it and talk about
about what's involved in doing this. So the first thing we're going to do, obviously, is our imports. So from keras
pre-processing import sequence, import keras we need TensorFlow NumPy and on. 

So we'll load that in. And now what we're going to do is download the file, so the dataset for Romeo and Juilet,
which we can get by using this line here. So keras has this utils thing, which will allow us to get a file, save
it as whatever we want. 

In this case, we're gonna save it as Shakespare.txt and we are goint to get that dataset from this link.
Now I believe this is just some like shared drive that we have access to from kera's, so we'll load that in here. 
And then this will simply give us the path on this machine, because remember, this is Google Collaboratory. And to 
this text file, if you want, you can "actually load in your own text data."

So we don't necessarily need to use the shakespeare play, we can use anything we want. In fact, an example that'll show
later is using the Movie Script. But the way you do that is run this block of code here. And you'll see that pops up this thing
for choose files just choose a file from your directory. 
"""

path_to_file = tf.keras.utils.get_file('shakespare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Loading our own data: To load our own data we'll need to upload a file from the dialog below. Then we'll need to follow
# this steps from the above but load in this new file instead.
## uncomment if you are using google colab
# from google.colab import files 
# path_to_file = list(files.upload()[0])

"""
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of the characters in it
print("Length of text: { } characters ".format(len(text)))

Now after we do that, we want to do is actually open this file. So remeber that was just saving the path to it. So we'll open
that file in RB mode, which is read bytes mode, I believe. And then we're going to say.read and you're going to read that in as an 
entire string, we're going to decode that into UTF-8 FORMAT.
print(text[:250])

And then we're just printing the length of the text or the amount of characters in the text. so if we do that, we can see
we have the length of the text is 1.1 million characters approax. And then we can have have a look at the first 250 characters 
by doing this. So we can see that this kind of what the plate looks like we have whoever speaking colon,: then some line,
whoever speaking :, some line, and there's all these brake lines. 

So backslash ends, which are telling us, you know, go to the next line, right. So it's going to be important, beceause we're
going to hope our neural network will be able to predict things like brake lines and spaces and even this kind of format as we 
teach it more and get further in.

"""

# READ CONTENT OF FILE
# LET'S LOOK AT THE CONTENTS OF THE FILE.
# READ, THEN DECODE FOR PY2 COMPAT 
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of the characters in it
print('Length of text: {} characters '.format(len(text)))
# Take a look at the first 250 characters in text
print(text[:250])


"""
So backslash ends, which are telling us, you know, go to the next line, right. So it's going to be important, beceause we're
going to hope our neural network will be able to predict things like brake lines and spaces and even this kind of format as we 
teach it more and get further in.

But now it's time to talk about encoding. so obviously, all of this text is in text form, it's not pre-processed for us, which means
we need to pre-process it and enode it as integers before we can move forward. Now fortunately, for us, this problem is actually
a little bit easier than the problem we discussed earlier with encoding words, because what we're  going to do is simply encode 
each character in the text with an integer.

Now, you can imagine why this makes this easier because there really is a a finite set of characters. Whereas ther's kind of indefinite,
or, you know, I guess infinite amount of words that could be created. So we are not really going to run into the problem where, 
you know, two words are encoded with such differ two characters are encoded with such different integers.

That makes it difficult for the model to understand because I mean we can lookat what the value of vocav is here. we are only 
going to have so many characters in the text. And for characters, just doesnot matter as much because you know, and 
R is not like super meaningful compared to an A. So we can kind of encode in a simple format, which is what we're going to do. 

vocab = sorted(set(text))
char2idx = { u:i for i, u in enumerate(vocab) }

So essentially, we need to figure out how many unique characters are in our vocabulary. So to do that, we're going to say vocab 
equals sorted set text. This will sort all of the unique characters in the text. And then what we're going to do is 
create a mapping from unique characters to in instance.

So essentially, we're gonna say UI for IU in a new enumerator vocabulary, what this will do is give us essentially 0,
whatever the string is, one, whatever the string is to whatever the strinf is for every single letter or character in our vocabulary
which will allos us to create this mapping. 

vocab = sorted(set(text))
idx2char = np.array( vocab )

And then what we'll do is just turn this initial vocabulary into a list or into an array. So we can just use the index at which a letter
appears as the reverse mapping. So going from index to letter,rather than lettered index, which is what this one's doing here.

def text_to_int(text):
    return np.array([char2idx[c] for c in text])
    
Next, I've just written a function that takes some text and converts that to an int, or the int_to representation for just to make 
a little bit easier for us as we get later on in the tutorial. So we 're just going to say np.array of in this case, and we're 
just going to convert every single character in our text into its integer representation by just referencing and putting that in a list
here, and then obviously, converting that to NumPy aRRAY. 

text_as_int = text_to_int(text)

SO Then , if we wanted to have a look at how this works, we can say text_as_int equals to int_text. so remember text is the entire loaded 
file that we had above here. So we're just going to convert that to integer representation entirely using this function.
And now we can look at how this works down here. So we can see that the text first citizen which is the first 13 letters, is 
Encodede:  [18 47 56 57 58  1 15 47 58 47 64 43 52]
and obviously each character has its own encoding. And you can go through and kind of figure out what they are based on the ones that
are repeated, RIGHT.

SO THAT IS HOW THAT WORKS.

""" 

vocab = sorted(set(text))
# Creating a mapping from unique characters to indices 
char2idx = { u:i for i, u in enumerate(vocab) }
idx2char = np.array( vocab )

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

print("Text: ", text[:13])
print("Encodede: ", text_to_int(text[:13]))
