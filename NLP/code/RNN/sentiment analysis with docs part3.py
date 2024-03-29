"""
Continue from Tutorial
https://www.freecodecamp.org/learn/machine-learning-with-python/tensorflow/natural-language-processing-with-rnns-making-predictions

"""

from keras.datasets import imdb
from keras.preprocessing import sequence, text
import tensorflow as tf
import numpy as np

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

# Training Magic Happens here. Compile and train the model
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["acc"])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print("Final Model Evaluation result: ", results)

"""

Making Predictions
Now lets use our network to make predictions on our own reviews. Since our reviews are encoded well
need to convert any review that we write into that form so the network can understand it. To do that well
load the encodings fromt the dataset and use them to encode our own data.

from tutorial:
since we have trained our model, and we want to actually use it to make a prediction on some kind of 
moview review. So since our data was pre-processed, when we gave it to the model, that means we actually
need to process anything we want to make a prediction on in the exact same way, we need to use the same 
lookup table, we need to encode it, you know, precisely the same. 

Otherwise, when we give it to the model, it's going to think that the words are different, and it's not 
going to make an accurate prediction. So what I've done here is I've made a function that will encode any 
text into what do you call it, the proper pre-processed kind of integers, right, just like our training 
data was pre-processed that's what this function is going to do for us is pre-processed some line of text.

so what I've done is actually gotten the lookup table. So essentially, the mappings from IMDB read that
properly, from the data set that we loaded earlier. So let me go see if I can find where I defined IMDb.

You can see up here. So kera's dataset import IMDb, just like we loaded it in, we can also actually get all
the word indezed in that map, we can actually print this out if we want to look at which it is after.
But anyways, we have that mapping, which means that all we need do is Kera's pre-processing text, text to 
word sequene.

What this happens is give given some text convert all that text into what we call tokens, which are just the 
individual words themselves. And then what we're going to do is just use a kind of for loop inside of here 
that says word index at word if word in word index,else zero for words and tokens.

Now what this means is essentially, if the word that's in these tokens now is in our mapping, so in that
vocabulary of 88,000 words, then what we'll do is replace its location in the list with that specific word,
or with that specific integer that represents it.

return sequence.pad_sequences([tokens], maxlen=MAXLEN)

otherwise, we'll put zero just to stand for you know, we don't now what this character is .And then what 
we'll do is return sequeunce.pad_sequences. And we'll pad this token sequence. And just return actually
the first index here. The reason we're doing that is because this pad sequences works on a list of 
sequences, so multiple sequences.

So we need to put this inside a list, which means that this is going to return to us a list of lists. So 
we just obviously want thet first entry because we only want you know that one sequence that we padded.

So that's how this works. Sorry, that's bit of a mouthful explain, but you guys can run through and print
this stuff out if you want to see how all of it works specifically. But yeah, so we can run this cell and 
have a look at what this actually does for us on some sample text. so that movie was just amazing so amazing,
Encoded :  [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    0   0   0   0   0   0   0   0   0  12  17  13  40 477  35 477]]
    
we can see we get the output that we were kinf of expecting so integer encoded words here, and then bunch of
zeros just for all the padding. Now while we're at it, I decide why not? Why we don't we make a decode function
so that if we have any movie review like this, that's in the integer form, we can decode that into the text value.

"""
# Making Predictions
word_index = imdb.get_word_index()

def encode_text(input_text):
    tokens = text.text_to_word_sequence(input_text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], maxlen=MAXLEN)

input_text = "that movie was just amazing, so amazing"
encoded = encode_text(input_text)
print("Encoded : ", encoded)

"""
we can see we get the output that we were kinf of expecting so integer encoded words here, and then bunch of
zeros just for all the padding. Now while we're at it, I decide why not? Why we don't we make a decode function
so that if we have any movie review like this, that's in the integer form, we can decode that into the text value.

def decode_integer(integers):

So the way we're going to do that is start by reversing the word index that we just created. Now the reason for that
is the word index we looked at, which is this right? Goes from word to integer. But we actually now want to go from 
integer to word so that we can actually translate a sentence right.

So what I've done is made this decode integers function, we've set the padding key as zero, which means that if we see
zero, that's really just means you know,nothing's there we are going to create a text string, which we're going to add
to that I'm gonna say. 

For num integers, integers is our input, which will be a list that looks something like this, or an array, whatever
you want to call it, we're gonna say if number doesnot equal pad, so essentially, if the number is not zero, right
it's not padding, then what we'll do is add the lookup of reverse word index num.

So whatever that number is, into this new string plus a space and then just return text colon and negative one, which 
means return everything except the last space that we would have added. And then if I print the decode integers, we can
see that this encoded thing that we had before, which looks like this gets encodede by the string, that movie was just 
amaxing so amazing. So may sorry, not encodede decoded, because this was the encodede form.
so that' how that works.

"""

reverse_word_index = { value: key for ( key, value ) in word_index.items() }

def decode_integer(integers):
    PAD = 0 
    text = ""
    for num in integers[0]:
        if num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

print("decode string from integers --> ", decode_integer(encoded))
            
"""
Making Prediction:
Okay, so now it's time to actually make a prediction. So I've written a function here that will make a prediction
on some piece of text as the movie review for us. 
And I'll just walk us through quickly how this works, then I'll show us the actual output from our model, you know
making predictions like this. So what we say is we'll take some parameter text, which will be our moview review, 
and we're going to encode that text using the ENCODE text function we've created above.

So just this one right here that essentially takes our sequence of you know, words, we'd get the pre-processing,
so turn that into a sequence, remove all the spaces what not, you know, get the words. Then we turn those into 
the integers we have that we return that.

So here we have our proper pre-process techs, then what we do we create a blank NumPy Array that is just a bunch
of zeros, that's in the form one 250, or in that shape.
Now the reason I'm putting that in that shape is because the shape that our model expects is something 250, which 
means some number of entries. And then 250 integers representing each word, right?

Cuz that's the length of moview review, is what we've told the model is like 250. So that's length of the review, 
then what we do is what predict zero, so that's what's up here, equals the encoded text.
So we just essentially insert our one entry into this, array we've created, then what we do is say model.predict
on that array and just return and print the result[0].

Now it's pretty much all there is to it. I mean, that's how it works. The reason we're doing result[0] is bcoz 
again, models optimized to predict on multiple things, which means like, I would have to do you know, list of 
encoded text, which is kind of what I've done by just doing this prediction lines here. which means it's going 
to return to me on array of arrays.

So if I want the first prediction, I need to index[0], because that will give me the prediction for our first
and only entry.
Alright, so I hope that makes sense. Now we have a positive review of written and a negative review.
And we're just going to compare the analysis on both of them. 

That movie was so awesome! I really loved it and would watch it again because it was amazingly great

That movie sucked. I hated it and wouldn't watch it again. It was one of the worst things I've ever watched.

>>> First One gets Predicted at 72% positive whereas the other is 23%.

>>>>> Make small changes in the text to see how model adapt, give output. <<<<<<
>>>>> Play with text, small changes can give difference in the model o/p <<<<<<<
>>>>  increase epochs to see the result <<<<<

"""

# prediction - magic happens here
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1, 250))  # Corrected the shape of the numpy array
    result = model.predict(pred)
    print(result[0])
    return result[0]

# Example for a positive review
positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great "
positive_review_percentage = predict(positive_review)
print("Accuracy Values of Positive Review: ", positive_review_percentage )

# Example for a negative review
negative_review = "That movie sucked. I hated it and wouldn't watch it again. It was one of the worst things I've ever watched. "
negative_review_percentage = predict(negative_review)
print("Accuracy Values of Negative Review: ", negative_review_percentage)

negative_review_variation = "That movie wasn't too bad. I didn't quite like it and won't recommend it to others."
negative_variation_accuracy = predict(negative_review_variation)
print("Accuracy Values of Negative Review with Variation: ", negative_variation_accuracy)

if negative_variation_accuracy >= 0.5:
    print("Positive Review!")
else:
    print("Negative Review!")
