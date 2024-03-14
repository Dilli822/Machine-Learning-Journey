
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


"""
MY Docs Link:
https://docs.google.com/document/d/1g3X082nmluU-jMGmZotAANUv6vPIDBz5IgZbT3gOIWQ/edit

* Sample Size in CNN
 This isnot really the best term to describe this but each convolutional layer is
 going to examine n x m blocks of pixels in each image.Type we will consider 3x3 or 5x5 
 blocks. In the example above we use a 3 x 3 "sample size". This size be the same as
 the size of our filter.
 
 Our layers work by sliding these filters of n x m pixels over every possible position
 in our image and populating a new feature map/response map indicating whether the
 filter is present at each location.
 
 Response Map -> Quantifying the response of the filter's pattern at different locations.
 
 Dense neural networks:
 Dense neural networks analyze input on a global scale and recognize patterns in specific areas.
 Dense neural network do not work well for image classification or object detection.
 It also analyze input globally and extract features from specific areas.
 
 Convolutional neural networks:
 Convolutional neural networks scan through the entire input a little at a time and learn local patterns.
 It returns a feature map that quantifies the presence of filters at a specific locations.
 and that filters advantages of it, we slide it in entire image and if this feature or filter
 is present in anywhere image then we will know the pattern in the image.

"""


"""

DRAWING BOARD - HOW CONVOLUTIONAL NETWORKS WORKS?
https://miro.com/app/board/uXjVNhgCCxA=/?share_link_id=982196115426





""""
Diagrams:
https://miro.com/app/board/uXjVNhgCCxA=/?share_link_id=173124021394

Step 1. For the green part, we use First Filters will try to find the
dot product between green and first filter box. Numeric values 
obtained after multiplying each of their element's wise numeric 
values inside the boxes.
A. B --> output role of the dot product is a numeric value 
that tells how similar these two originals and the first filter 
box, or the sample and the filter box are.

Step 2. After first green part we move to another grid area greeny 3 x 3 
and compare each pixel with the filters oh we got some similarities 
in the second row second column ie. row2col2
but no filter in the first row second column in the original so we 
set the absence pixel field to 0.

Assign randomly.
numeric values 1 are perfect match and values nearer to 1 are similar then closer to 0.
These features inform us the presence of features in this original map.

we will have tons of different layers as we are constantly expanding, as we go to the depth of the neural network, need lot of computation which leads more time consumption.

---- Rough Idea on How CNN works? -------
STEP 3:
we have generated output feature map from the original image.
Now the next convolutional layer will find the feature except
the output feature maps that means next convolutional layer
will process or find the combinations of lines and edges and
maybe find what a curve is. we slowly work our way up from very,
very small amount of pixels, to finding more and more, almost, 
small amount of pixels, to finding more and more, almost abstract 
different features that exist in the image. And this really allows 
us to do some amazing things with the convolutional neural network.
when we have a ton of differnt layers stacking up on  eachother 
we can pick out all the small little edges which are pretty easy 
to find. And with all these combinations of layers working together,
we can even find things like eyes, feets, or heads

when we have a ton of differnt layers stacking up on 
eachother we can pick out all the small little edges
which are pretty easy to find. And with all these
combinations of layers working together, we can even 
find things like eyes, feets, or heads.

We can find very complicated structures, because we slowly
our work way up starting very easy problem, which are likely 
finding lines, and then finding the combination of lines, edges,
shapes and the very abstract things. 
That's how convolutional neural networks .

"""

"""
------ Padding -------
- Padding makes best sense ways making space, sometimes we want to
make sure that the output feature map from our original image here 
is the same shape or N X N Size and the shape.
In the diagram what we are doing is ewe have original image size is
5 x 5 and the feature image is 3 x 3 so for that if we want to make 
this 5 x 5 as an output what we need to do is simply add the padding
our original image.

We add extra col and row or border around the original image,
adding so must make the pixel at the center of the image as shown
in the diagram.
Observe the red colored has pixel exactly at the center of the 
box.
but the green part has not and they cannot be center with the padding
allows us to do is generate an output map that is the same size as
our original input, and allows us to look at features that are 
maybe right on the edges of images that we might not have been 
able to see. Although this is not important for large but understanding
is and applying for small is fine. X are added to mark cross.

--- STRIDE ----- 
Stride is something that explain how many times we move the sample box
every time that we are about to move it.

Stride of One - Note we have added padding now we move to another pixel
or 1 pixel that is called stride of one.
But we can also take n stride or n times moving or we can two stride
obviously for larger stride then it will be 2 or 4 stride.

"""



"""
https://miro.com/app/board/uXjVNhgCCxA=/?share_link_id=748462891002

 ----- Pooling Operation --------
we have tons of layers and lots of computation for all this filter
and there must be someway to make these a little bit simpler
a little bit easier to use. Well, yes, that's true. And there is a 
way to do that And that's called pooling. So there's three
types of pooling

Max -mostly used that tell us about the maximum presence of a 
feature that kind of local area, we really only care if the 
feature exists.

Min - if 0 does not exists.

Average - not mostly used, tells average presence of features 
in local area.


A Pooling operation is just taking specific values from a sample of the
output feature map. So once we generate this output feature map,
what we do to reduce its dimensionality and just make it a little bit
easier to work with, is when we sample typically 2 x 2 areas of this
output feature map, and just take either the min max average value of 
all the values inside of here and map these, we are goona go back this
way to a new feature map that's twice the one times the size essentially.

What are the three main properties of each convolutional layer?
Input size, the number of filters, and the sample size of the filters.
"""