import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
import pandas as pd

# Load data
dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # training dataset
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # testing data
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# Define feature columns
feature_columns = []
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Define input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Define and train the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

# Print accuracy
print(result['accuracy'])
# we get output accuracy at 0.7462121 
print(result)
# GIVES OUTPUT IN THE OBJECT FORMAT REMEMBER WE HAVE DONE PARSE IN OBJECTS
# {
#     'accuracy': 0.74242425,
#     'accuracy_baseline': 0.625,
#     'auc': 0.8375268,
#     'auc_precision_recall': 0.7862294,
#     'average_loss': 0.47628114,
#     'label/mean': 0.375,
#     'loss': 0.4689349,
#     'precision': 0.65346533,
#     'prediction/mean': 0.37340146,
#     'recall': 0.6666667,
#     'global_step': 200
# }


# SINCE TENSORFLOW MODEL IS GOOD AT GIVING OUTPUT OR PREDICTION FOR LARGER OR MORE THAN SINGLE PIECE OF DATA
# MAKING PREDICTION 
specific_point_result = list(linear_est.predict(eval_input_fn))
print("list of all specific point result ")
print("every point prediction even though tensorflow is not made for this ", specific_point_result)

# I want to see the prediction of one dict
print("one dict prediction ", specific_point_result[0])

# {
#     'logits': array([-2.2739615], dtype=float32),
#     'logistic': array([0.09330254], dtype=float32),
#     'probabilities': array([0.90669745, 0.09330254], dtype=float32),
#     'class_ids': array([0]),
#     'classes': array([b'0'], dtype=object),
#     'all_class_ids': array([0, 1], dtype=int32),
#     'all_classes': array([b'0', b'1'], dtype=object)
# }
# not survival values 0 means not survived 
print("probabilities of not surviving indexes both 0 and 1 -->", specific_point_result[0]['probabilities'])
print("probabilities of surviving object indexed 1 --->", specific_point_result[0]['probabilities'][1])
print("probabilities of surviving object indexed 0 --->", specific_point_result[0]['probabilities'][0])


# we can also have a look for a specific person stats based on our model
each_person_result = list(linear_est.predict(eval_input_fn))
print("person 1 ", dfeval.loc[0])
print("--------------------------------")
# person 1  sex                          male
# age                          35.0
# n_siblings_spouses              0
# parch                           0
# fare                         8.05
# class                       Third
# deck                      unknown
# embark_town           Southampton
# alone                           y
# Name: 0, dtype: object
print("result of the probablities ---> ", each_person_result[0]['probabilities'][1])
# result of the probablities --->  0.09225855
print("person 2", dfeval.loc[2])
print("result of the person2 stats", each_person_result[2]['probabilities'][1])

print("---------------------")
print("lets find the either the person survived or not stats ", dfeval.loc[3])
print("lets find the either the person survived or not ", y_eval.loc[3])
print("result of the person3 stats  percent accuracy", each_person_result[3]['probabilities'][1])

# The provided code is a TensorFlow script that trains a linear classifier model on the 
# Titanic dataset and evaluates its performance. It also makes predictions on individual 
# data points and examines the results.



# Extract probabilities of survival for each individual from the prediction results
# survival_probabilities = [result['probabilities'][1] for result in each_person_result]
# print("survival probalities of each person -->", survival_probabilities)

# # Plot the survival probabilities
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(survival_probabilities)), survival_probabilities, color='skyblue')
# plt.xlabel('Individuals')
# plt.ylabel('Survival Probability')
# plt.title('Survival Probability Predictions for Each Individual')
# plt.xticks(range(len(survival_probabilities)), [f'Person {i+1}' for i in range(len(survival_probabilities))])
# plt.ylim(0, 1)  # Set y-axis limits to 0 and 1 for probabilities
# # plt.grid(axis='y', linestyle='--', alpha=0)
# plt.grid(axis='y', linestyle='--', alpha=0.8)
# plt.show()

# Extract the survival probability for the specific person from the prediction results
# person_index = 1 # Index of the person you want to visualize (adjust as needed)
# person_survival_probability = each_person_result[person_index]['probabilities'][1]
# print(person_survival_probability)

# # Plot the survival probability
# plt.bar(['Survived', 'Not Survived'], [person_survival_probability, 1 - person_survival_probability], color=['green', 'red'])
# plt.xlabel('Outcome')
# plt.ylabel('Probability')
# plt.title(f'Predicted Survival Probability for Person {person_index+1}')
# plt.ylim(0, 1)  # Set y-axis limits to 0 and 1 for probabilities
# plt.show()


# In matplotlib, alpha is a parameter that controls the transparency of graphical elements, such as lines, markers, and shapes. It takes a value between 0 and 1, where:

# alpha = 0 corresponds to fully transparent (invisible) elements.
# alpha = 1 corresponds to fully opaque (fully visible) elements.
# Values between 0 and 1 represent varying levels of transparency, with higher values indicating less transparency (more opacity).

