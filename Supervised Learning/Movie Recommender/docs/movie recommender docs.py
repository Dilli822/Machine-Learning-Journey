

# Movie Recommendatino system using tensorflow
# Tutorial: https://www.tensorflow.org/recommenders/examples/basic_retrieval


""" 
Recommending Moview: Retrieval . 
It has two stages:
1. Retrieveal Stage - responsible for selecting an initial set of hundreds of candidates from all possible candidates. It's main objective is to filter out
the all candidates that the user is not interested in it, because of computational efficiency dealing with the millions of candidates.

2. The Ranking Stages: it takes the output of the retrieval model and fine-tunes them to select the best possible handful of recommendations. it narrow down
the set of items the user may be interested in to shortlist of likely candidates.

Putting it all together:

The retrieval stage quickly narrows down the huge pool of movies to a smaller, more manageable list based on your past behavior.
The ranking stage then takes that shorter list and tailors it even further to your tastes, considering factors like popularity, genre, and more detailed preferences.
Finally, you see a personalized list of movie recommendations that are most likely to appeal to you, making it easier to find something you'll enjoy watching.

This tutorial we are going to focus on the first stage i.e. Retrievel stage 

Retrieval Models are composed of two sub-models:
1. A Query Model: It computes the query representing (normally a fixed-dimensionality embedding vector) using query features.

The query model helps to quickly find a bunch of movies that match what you're looking for, making it easier for you to pick something to watch without spending ages 
searching through everything.

2. Candidate Model: It computes the candidate representation (an equally-sized vector) using the candidate features.

Finally we multiply the outputs of the two models and give a query-candidate affinity score, with higher scores that gives better match between the candidate and query.

What we will be doing
1. get our data and split it into a training and test set
2. implement a retrieval model
3. fit and evaluate it.
4. Export it for efficient serving by building an approximate nearest neighbours(ANN) index.

"""

# libraries or packages installation
# pip install -q tensorflow-recommenders
# pip install -q --upgrade tensorflow-datasets
# pip install -q scann


# Imports 
import os 
import pprint
import tempfile 

from typing import Dict, Text

import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_recommenders as tfrs 

# Preparing the dataset - Movielens dataset from Tensorflow Datasets 
# we are loading the tf.data.Dataset object containig the ratings data and loading movielens/10k_movies object containing only the movies data

"""
Dataset:
classic dataset from the GroupLens, contains a set of ratings given to movies by a set of users. 
We are treating the data in two ways:

Implicit feedback: Movie watched and rated by users and which the movies which they didnot,
Explicit feedback: How much the users liked the movies they did watch., we can tell roughly how much liked the movie looking at the rating that a user
have given.

We are a positive example for every movie a user watched and negative example on those movie which are not watched. SO what we want to do is model that
predicts a set of movies from the catalogue that the user is likely to watch here implicit data is used here and treat movielens dataset as an implicit 
system.

"""


# ratings data
ratings = tfds.load("movielens/100k-ratings", split="train")
# features of all the available movies
movies = tfds.load("movielens/100k-movies", split="train")

# the ratings dataset returns a dict of movie id, user id, and the assigned rating and timestamp and movie information and user information:

for x in ratings.take(1).as_numpy_iterator():
    pprint.pprint(x)
print(" ")
# let's focus on the ratings of data from the dataset and only keep the user_id and movie_title
ratings = ratings.map(lambda x : {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"]
})
movies = movies.map(lambda x:x ["movie_title"])
print("only movie extracted frmo dataset: ", movies)

# To fit and evaluate the model, we need to split it into a training and evaluation set. In an industrial recommender system, this would most likely 
# be done by time: the data up to time 
#  would be used to predict interactions after 
# In this simple example, however, let's use a random split, putting 80% of the ratings in the train set, and 20% in the test set.

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
print("shuffled: ", movies)
print(" ")
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
print(test)
print(" ")
print(train)
print(" ")
# Set the random seed
# tf.random.set_seed(2)
# # Generate random numbers with shape1 same consistent
# random_numbers = tf.random.uniform(shape=(1,))
# print("Random numbers:", random_numbers)
# Set the random seed to 2
# Set the random seed to 42
# tf.random.set_seed(42)
# # Generate random numbers
# random_numbers_42 = tf.random.uniform(shape=(3,))
# print("Random numbers with seed 42:", random_numbers_42)
# In summary, the fundamental difference is that each seed value initializes the random number generator in a different state, leading to a different sequence of random numbers being generated for each seed.

"""
Let's also figure out unique user ids and movie titles present in the data. 

This is important because we need to be able to map the raw values of our categorical features to embedding vectors in our models. To do that, we need a 
vocabulary that maps a raw feature value to an integer in a continguous range; this allows us to look up the corresponding embeddings in our embedding tables:

"""
print("")
movie_titles = movies.batch(1_000)
print(movie_titles)
print("")

user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
print(user_ids)

print("")
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

print("")
unique_movie_titles[:10]
# u = unique_movie_titles[:10]
# print(u)

"""
Implementing a Model:
Our model is a two-tower retrieval model and so we can build each tower separately and then combine them 
in the final model.

The Query Tower:
Let's start with query tower and first step is to decide on the dimensionality of query and candidate representations.

higher the values more accurate but slower to fit and more prone to overfitting opps!

"""
embedding_dimension = 32 

"""
secondly we define the model itself, here we're going to use keras preprocessing layers to first convert "user ids to integers ",
and then convert those to user embeddings via Embedding Layer. Note that we use the list of unique user ids we computed earlier 
as a vocabulary.
"""

"""
tf.keras.Sequential([ ... ])

This line defines a sequential model in Keras, a popular deep learning library within TensorFlow. A sequential model stacks layers sequentially, meaning the output of one layer becomes the input for the next.

tf.keras.layers.StringLookup(
# Sample data
unique_user_ids = ['user1', 'user2', 'user3', 'user4']
user_data = ['user1', 'user2', 'user3', 'user4', 'user1', 'user2', 'user3']

# Create a StringLookup layer
string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=unique_user_ids, mask_token=None)

# Convert string data to integer indices using the StringLookup layer
encoded_data = string_lookup_layer(user_data)

# Print the encoded data
print("Encoded data:", encoded_data)

vocabulary=unique_user_ids, mask_token=None),
vocabulary = unique_user_ids: This argument specifies the vocabulary for the layer. It should be a list or tensor containing all the unique user IDs you expect to encounter in your data. The StringLookup layer will create a mapping between each unique user ID string and a unique integer index.

mask_token = None: This argument is optional and specifies a special token to be used for out-of-vocabulary (OOV) values. OOV values are user IDs that are not present in the provided vocabulary. By default (as set here with None), any OOV value will be assigned a special index (usually 0).

"""

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary = unique_user_ids, mask_token = None),
    # we add an additional embedding to account for unknown tokens
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

"""
Certainly! Let's break it down in simple terms:

Consider you have a list of unique user IDs like this: ['user1', 'user2', 'user3', 'user4'].

Now, imagine you want to convert these user IDs into numbers for use in a machine learning model. This process is called encoding.

Here's what vocabulary = unique_user_ids, mask_token = None does:

vocabulary:

It's like a dictionary that maps each user ID to a unique number. For example:
'user1' might be mapped to 1.
'user2' might be mapped to 2.
And so on.
This vocabulary tells the system which user ID corresponds to which number. It's like giving each user ID a unique ID number.
mask_token:

This is an optional parameter. It's like saying, "What should we do if we encounter a user ID that's not in our list?"
When mask_token is set to None, it means we'll ignore any user IDs that aren't in our list. We won't give them a number.
If we set mask_token to something like <MASK>, it means if we see a user ID that's not in our list, we'll assign it a special number, like 0 or 999, to indicate that it's a special unknown user.
So, in simpler terms, vocabulary = unique_user_ids, mask_token = None is like creating a system that assigns each user ID a unique number, and if we don't recognize a user ID, we'll just ignore it.

Example:

python
Copy code
import tensorflow as tf

unique_user_ids = ['user1', 'user2', 'user3', 'user4']

# Create a StringLookup layer
string_lookup_layer = tf.keras.layers.StringLookup(
    vocabulary=unique_user_ids, mask_token=None)

# Convert user IDs to numbers
encoded_data = string_lookup_layer(['user1', 'user5', 'user2', 'user3'])

print("Encoded data:", encoded_data)
Encoded data: tf.Tensor([1 0 2 3], shape=(4,), dtype=int64) o/p


So, in simpler terms, tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension) 
creates a layer that converts each unique user ID into a dense vector representation of fixed size.
This representation is learned during training and aims to capture relationships between different 
user IDs in a way that's suitable for the machine learning model.


unique_user_ids = ['user1', 'user2', 'user3', 'user4']
embedding_dimension = 10  # Assuming each user ID will be represented by a 10-dimensional vector

# Create an Embedding layer
embedding_layer = tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)

# Example usage (not trainable in this example)
encoded_user_ids = [1, 2, 3, 0]  # Encoded user IDs
embeddings = embedding_layer(encoded_user_ids)  # Convert user IDs to embeddings

print("Embeddings:")
print(embeddings)


Embeddings:
tf.Tensor(
[[ 0.02671548  0.04362463  0.03416564 -0.02057222 -0.03970771 -0.0499075
  -0.04458055 -0.02788404  0.04806618  0.01322806]
 [-0.01557792 -0.04815435  0.03623804 -0.01904916 -0.01336207  0.04720708
   0.0263673  -0.0163031   0.02903443  0.02782239]
 [-0.01906216 -0.0350431  -0.03866072 -0.00621962 -0.03680024 -0.01547108
  -0.00740239 -0.01153095 -0.03284078  0.02792472]
 [ 0.04582418 -0.02878322 -0.04419418 -0.01704079  0.04123692 -0.03249884
  -0.01296197  0.02443951 -0.03590362  0.01206182]], shape=(4, 10), dtype=float32)

"""

# Now for Candidate Tower

movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

import matplotlib.pyplot as plt
import seaborn as sns

# Count the frequency of each movie title
title_counts = {}
for title in unique_movie_titles:
    title_counts[title] = title_counts.get(title, 0) + 1

# Sort the movie titles based on their frequency
sorted_titles = sorted(title_counts.items(), key=lambda x: x[1], reverse=True)

# Extract titles and counts for plotting
titles, counts = zip(*sorted_titles)

# Plot the distribution of movie titles
plt.figure(figsize=(10, 6))
plt.barh(range(len(titles)), counts)
plt.yticks(range(len(titles)), titles)
plt.xlabel('Frequency')
plt.ylabel('Movie Titles')
plt.title('Distribution of Movie Titles')
plt.show()

# Create a DataFrame with movie titles and their counts
data = {'Movie Title': titles, 'Frequency': counts}
df = pd.DataFrame(data)

# Plot the distribution of movie titles using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Movie Title', data=df)
plt.xlabel('Frequency')
plt.ylabel('Movie Titles')
plt.title('Distribution of Movie Titles')
plt.show()