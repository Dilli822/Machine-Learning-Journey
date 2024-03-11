# Input layer
# Hidden Layer
# Output Layer
# Bias
# --- it is not input but a constant numeric value
# --- it gets connected to each hidden layers and output layer neurons
# --- it is trainable 
# --- whenever bias is connected to another layer then its weight is 1
# ---- bias never gets added or connected with each other

# ----------- Activation Functions -----------
# Rectified Linear unit.(Relu)
# it makes any numeric values less than 0 or -ve values as 0 and for +ve it is positive values
# Tanh(Hyberbolic Tanget)
# more positive closer to +ve and -ve closer to -ve just a normal graphs but ranges from -y to +x

# Sigmoid Function
# this function is to remove non linearity and is shaped-curve and any positive numbers closer 
# to 1 and any negative numbers closer 0


# --------- HOW TO USE THEM IN MODEL? ---------


# N1 = Activation function is applied to the output of that particular connected neurons 
# that means for n1 we apply activation function before sending its n1 output to another neuron m1
# WE CAN USE ANY ACTIVATION FUNCTION BASE ON THE REQUIREMENTS 

# FOR O TO 1 RANGE WE USE SIGMOID FUNCTION, SQUISHES VALUES BETWEEN O AND 1


# WHY TO USE ACTIVATION FUNCTION IN NEURAL INTERMEDIATE NETWORK LAYER?
# - TO INTROUDCE THE COMPLEXITY IN INPUT, WEIGHTTS AND BIAS
# - activation function is a higher dimensional function that spreads the 
#    linear or clustered points in a single dataset, spreading the dataset
#     will give us pattern, characteristics and some features of the data
#     moves in n dimensional data moves to 1D, 2D Or 3D data dimension
#     for example n dimension of shapes like SQUARE, CUBE

#  square provides less details then 
#  VOLUME/CUBE shaped 
# OUR AIM IS TO MOVE FRO AND BACK IN HIGHER AND LOWER DIMENSION TO 
# EXTRACT THE INFORMATION FROM THE DATA LIKE IN SQUARE WE CAN GET EITHER LENGTH
# OR BREADTH BUT IN CUBE WE CAN GET MORE DETAILS LENGTH, HEIGHT AND BREADTH
# in here for us is a matrix, scalar or event tensor of n dimension

# THAT EVENTUALLY LEADS TO BETTER PREDICTIONS 


#--------- LOSS FUNCTION --------
# 1. IT CALCULATE HOW FAR AWAY OUTPUT WAS FROM EXPECTED OUTPUT?
# suppose expected output was 1 but we get 0.2 and then how far it is from
# the expected and obtained output
# THAT MEANS WE CAN COMPARE HOW BAD OR GOOD IS NETWORK OUTPUT?
# BIASNESS CAN BE EVALUATED FROM LOSS FUNCTION
# HIGH LOSS MEANS VERY BAD NETWORK
# LOW LOSS MEANS NOT VERY BAD NETWORK
# AND IF HIGH LOSS CHANGE THE WEIGHT AND BIAS DRASCTICALLY AND GUIDE IT TO DIFFERENT OUTPUT OR PATH
# AND IF GOOD THEN THATS OKAY
# SINCE CHANGING OR TWISTING SMALL CHANGES IN THE NETWORK NUMERIC INPUT VALUES 
# CAN BRING VAST DIFFERENCE IN THE OUTOUT AND LOSS VALUE 
