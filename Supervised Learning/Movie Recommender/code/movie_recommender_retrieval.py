

# Movie Recommendatino system using tensorflow
# Tutorial: https://www.tensorflow.org/recommenders/examples/basic_retrieval
# libraries or packages installation
# Imports 
import os 
import pprint
import tempfile 

from typing import Dict, Text

import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds 
import tensorflow_recommenders as tfrs 

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

embedding_dimension = 32 

user_model = tf.keras.Sequential([
    tf.keras.layers.StringLookup(
        vocabulary = unique_user_ids, mask_token = None),
    # we add an additional embedding to account for unknown tokens
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

# Now for Candidate Tower
movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])
metrics = tfrs.metrics.FactorizedTopK(
    candidates = movies.batch(128).map(movie_model)
)
print("metrics-->", metrics)


task = tfrs.tasks.Retrieval(
    metrics=metrics
)
print("task--->", task)

class MovielensModel(tfrs.Model):
    
    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.Layer = task 
        
    
    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # we pick out the user features and pass them into the user model
        user_embeddings = self.user_model(features['user_id'])
        #　and pick out the movie features and pass them into the movie model, 
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features['movie_title'])
        print(user_embeddings)
        print(positive_movie_embeddings)
        
        # task computes the loss and the metrics 
        return self.task(user_embeddings, positive_movie_embeddings)

print("full model build ", MovielensModel.compute_loss)

# Fitting and Evaluating 
# lets fit and evaluate the models, using optmizer adagard

model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# then shuffle, batch and cache the training and evaluation data
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=10)

model.evaluate(cached_test, return_dict = True)

# Making Predictions
# Now that we have a model, we would lik to able to make predictions, we can use the tfrs.layers.factorized_top_k.BruteForce layer to do this

# Create a model that takes in raw query features, and 
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)

# recommends movies out of the entire movies dataset
index.index_from_dataset(
    tf.data.Dataset.zip (( movies.batch(100), movies.batch(100).map(model.movie_model) ))
)

# user_no = 42

# get recommendations
_, titles = index(tf.constant([ "1" ] ))
print(f"Recommendations for user 1: {titles[0, :3]}")