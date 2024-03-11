from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

print("LINEAR REGRESSION VS CLASSIFICATION")
print("Classification classify the items under different class based on their similarlity")

# using keras a module instead of tensorflow to grab our datasets and read them into pandas dataframes
CSV_COLUMN_NAME = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    
)

test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

# row 0 is a header row
train = pd.read_csv(train_path, names=CSV_COLUMN_NAME, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAME, header=0)

train.head()
print("train data -->")
print(train.head())

print(test.columns)

# now we pop the species for training and testing purpose
train_y = train.pop('Species')
test_y = test.pop('Species')

# The label column has now been removed from the features
print(train.head())

# printing the shape
print(train.shape)   # (120, 4) in 4 columns 120 entries

# input function 
def input_fn(features, labels, training=True, batch_size=256):
    # convert the input to the datasets
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # shuffle and repeat if in the training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)
    
# feature columns
# we just check/loop in with the keys and append with the myfeaturecolumns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
print(my_feature_columns)

# Building the Model
# Model can be build using two ways:
# 1. LinearClassifier (similar to Linear Regression)
# 2. DMClassifier (Deep Neural Network)
# 3. We can create a custom model but must follow mathematics and rules

# Builing the model with 2 hidden layers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    # TWO HIDDEN LAYERS OF 30 AND 10 NODES RESPECTIVELY
    hidden_units = [30, 10],
    # the model must choose between 3 classes
    n_classes = 3
)

# training the model
# x = lamba: print("hello") lambda function is anonymous function that accepts n number of arguments
# but return only one output and here lamba is accepting the print builit in function or input_fn which is cool thing
classifier.train(
    input_fn = lambda: input_fn(train, train_y, training=True),
    steps = 5000
)


# train the model
train_eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(train, train_y, training=False))

# Evaluate the  test model on the training data
test_eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(test, test_y, training=False))


accuracy = train_eval_result['accuracy']
inaccuracy = 1 - accuracy

print("\nTraining set accuracy: {:.2f}%".format(accuracy * 100))
print("Training set inaccuracy: {:.2f}%".format(inaccuracy * 100))


# ----------------------------------------------------------------------------------------
# functions for predictions
# flower classification predictions
def input_fnc_flower(features, batch_size = 256):
    # convert the inputs to a dataset without labels
    # we conver the features param into tensor dataset dict with batch size
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features_flower = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted. ")
for flower_feature in features_flower:
    valid = True
    while valid:
        val = input(flower_feature + ": ")
        if not val.isdigit(): valid = False
    
    # wait untill gets validations 
    # if validations found then add that to the dict predict[SepalLength], predict[SepalWidth]
    predict[flower_feature] = [float(val)]
    
predictions = classifier.predict(input_fn = lambda: input_fnc_flower(predict))
print("Predictions variable is ---> ", predictions)

defect_percentages = []
correct_percentages = []
predicted_classes = []  # List to store predicted class names

for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    predicted_species = SPECIES[class_id]
    predicted_classes.append(predicted_species)  # Store predicted class name

    print("Class Id is ---> ", class_id)
    print("Probability ---> ", pred_dict)

    print('Predictions is "{}" ({:.1f}%)'.format(
        predicted_species, 100 * probability
    ))
    correct_percentages.append(probability)

    # Calculate defect (inaccuracy) percentage
    defect = 1 - probability
    print('Predictions Inaccuracy is "{}" ({:.1f}%)'.format(
        predicted_species, 100 * defect
    ))
    defect_percentages.append(defect)

labels = ['Defective Predictions', 'Correct Predictions']
sizes = [sum(defect_percentages), sum(correct_percentages)]
colors = ['red', 'green']

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, startangle=140)
plt.title('Defective vs Correct Predictions of the flower classification')

# Adding text for predicted class name
plt.text(0, -1.2, "Predicted flower is: " + ', '.join(predicted_classes), fontsize=12, horizontalalignment='center')

plt.axis('equal') 
plt.show()
# some expected output or result sample testing data
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalWidth': [1.7, 4.2, 5.4],
    'PetalLength': [6.5, 1.5, 2.1],
}



# ------------ prediction input function values output ------------- 
# Training set accuracy: 95.83%
# Training set inaccuracy: 4.17%
# Please type numeric values as prompted. 
# SepalLength: 2.4
# SepalWidth: 2.6
# PetalLength: 6.5
# PetalWidth: 6.3
# Predictions is "Virginica" (99.2%)


# Class Id is --->  2
# Probability --->  {
#     'logits': array([-7.8942537,  0.5859709,  5.6713963], dtype=float32), 
#      since 9.93 is high probability it must be Virgincia
#     'probabilities': array([1.2759513e-06, 6.1482126e-03, 9.9385047e-01], dtype=float32), 
#     'class_ids': array([2]), # that means it is Virgincia 
#     'classes': array([b'2'], dtype=object), 
#     'all_class_ids': array([0, 1, 2], dtype=int32), 
#     'all_classes': array([b'0', b'1', b'2'], dtype=object)
# }
# Predictions is "Virginica" (99.4%)
# output interpretations 
# Logits: Raw output scores before activation function (e.g., softmax) is applied.
# Probabilities: Transformed logits into probability distribution after applying the activation function.

