"""


COST OR LOSS FUNCTION
 THREE TYPES OF COST FUNCTION
1. MEAN ABSOLUTE
2. MEAN SQUARED
3. Hinge loss

Gradient Descent
Since our networks parameters are weights and
bias by just changing the bias and weights
we can make either the network better or worst

and the task of finding out that is done by 
LOSS Function
it determine how good or worse is our network

Based on that we can determine move the network
to change the worst scenario.

Link: https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html/
https://chat.openai.com/share/0d9c4dee-d141-44f6-9eec-69c8825131fd

FROM THE GRADIENT DESCENT FIGURE
Higher Dimension --> Higher Space , spread of dataset
Decrease the loss value to get the better result 

Stars path are loss function.
Global Minimum - least possible loss from our neural network

Red Circles then we move from it to the downwards global minimum and the process is called
gradient descent. it tells us what DIRECTION we need to move our function to determine or
get the global minimum.

Back propogation - goes backwards to the network updates the weights and bias according to the 
gradient descent and guide to the direction

NEURAL NETWORKS
INPUT - HIDDEN - OUTPUT LAYER
WEIGHTS ARE CONNECT TO EACH LAYER
BIAS (VALUE = 1) ARE ONLY CONNECTED TO THE HIDDEN AND OUTPUT LAYER, MOVE UP ,MOVE DOWN
THIS BIASES CAN BE THOUGHT OF Y-INTERCEPT THAT DEPENDS ON ACTIVATION FUNCTION


Activation Functions Roles
tanh = between -1 to 1
sigmoid = between 0 and 1
relu = 0 to positive infinity

WE ADD BIASES TO EACH WEIGHTED AND INPUT 
WE COMPARATIVELY LOOK THE OUTPUT 


data collection
defining the problem
inputs and weights and bias
calcuate for each neurons
apply the activation functions
loss functions 
backpropagation
descent gardient
training
validating
testing

BETTER THE PREDICTION LOWER THE LOSS FUNCTION
AS WE TRAIN OUR NETWORKS IT MAY GET BETTER OR WROST BASED ON
HOW WE CHOOSE WEIGHTS AND BIAS AND ACTIVATION FUNCTIONS,
OPTMIZATION AND GRADIENT DESCENT VALUES.

"""