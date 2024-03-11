# Deep Computer Vision

"""
Image Classification and Object detection/recognition using deep computer vision
with something we call a convolutional neural network.

The goal of our convoloutional neural networks will be to classify and detect images or specificobjects from within the image. 
We will be using image data as our features and a label for those images as our label or output. 

We already know how neural networks work so we can skip through the basics and move right into 
explaining the following concepts.
1. Image Data
2. Convolutional Layer
3. Pooling Layer
4. CNN Architectures

The major differences we are about to see in these types of neural networks are the layers that make them up.


1. Image Data 
   so far we have dealt with pretty straight forward data that has 1 or 2 dimensinos. Now we are about to deal with 
   image datais usually made up of 3 dimensions . These dimensions are as follows:
   
   - image height
   - image width
   - color channels
   The only item in the list above you may not understand is color channels. The number of color channels represents the depth of 
   an image and coorelates to the colors used in it. For example, an image with three channels is likely made up of rgb(red, green, blue)
   pixels. So Far, each pixel we have three numeric values  in the range of 0-255 that defines its color.For an image of color depth 1 we would
   likely have a greyscale image with one value defining each pixel, again in the range of 0-233.
   

 Convolutional Neural Network
 Note: I will use the term convnet and convolutional neural network interchangably.
 Each convolutional neural network is made up of one or many convolutional layers. These layers are different than the dense layers we have
 seen before. Their goal is to find the patterns from within images that can be used to classify the image or parts of it. But this may sound
 familiar to what our densly connected neural network in the previous section was doing, well that's because it is.
 
 The fundamental difference between a dense layer and convolutional layer is that dense layers detect patterns globally while convolutional 
 layers detects patterns locally. When we have a densely connected layer each node in that layyers sees all the data from the previous layer.
 This means that this layer is looking at ALL of the information and is only capable of analyzing the data in a global capacity. Our convolutional
 layer however will not be densly connected, this means it can detect local patterns using part of the input data to that layer.
 
 Let's have a look at how a densly connected layer would look at an image vs how a convolutional layer would.
 
 For an example in our cat image cat has left side head and image now our classification model
 memorized that part and Ah thats cat it has certain global pattern that model is memorizing it .
 But it would not recoginze if the same cat image is flipped horizontal our model gets confused.
 It learnt the pattern in the specific red dotted rectangle part only.
 
 But Convolutional network scan each line and find the features in the image and based on that features 
 it pass the features to the dense layers, now dense classifier based on that features,determing the combination of 
 these presences of features that make up specific classes or make up specific objects.
 
 SO the dense layers sees the pattern in the whole dense layer whereas convolutional will look in local areas notice the 
 features in local areas not just global.
 
"""