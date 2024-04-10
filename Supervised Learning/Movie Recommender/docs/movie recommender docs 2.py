

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
print("user model is --> ", user_model)

# Now for Candidate Tower
movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])
print("movie model is ---> ", movie_model)

"""
Metrics:
In our training data, we have positive (user, movie) pairs. We need to compare the affinity score that the model calculates
for this (user,movie) pairs to the scores of all the other possible candidates this is all we are doin because we need to 
figure out how well our model is performing.

For above purpose we can use the tfrs.metrics.FactorizedTopk metric, this accept one argument, the argument is the dataset of
candidates that are used as implicit negatives for evaluation.

Let's convert the movies dataset into embeddings via our model:

"""

metrics = tfrs.metrics.FactorizedTopK(
    candidates = movies.batch(128).map(movie_model)
)
print("metrics-->", metrics)


"""

Loss:
Loss function used to train our mode, tfrs has several loss layers and taks to make this easy. In this instance, we'll make use
of the retrieval task object; a convenience wrapper that bundles together the loss function and metric computation.

What is taks?
here tasks is a keras layer that takes the query and candidates embeddings as arguments and also returns the computed loss, and we
will use that to implement the model's training loop.
"""

task = tfrs.tasks.Retrieval(
    metrics=metrics
)
print("task--->", task)

"""

Full Model:
We can now put it all together into a model. TFRS exposes a base model class  (tfrs.models.model) which streamlines building models.

we setup the components in the init_method and implement compute_loss method,
>>> def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
take raw features and returning a loss value.

>>> return self.task(user_embeddings, positive_movie_embeddings)

Tfrs.model base class allows to compute both training and test losses using the same method.
we can achieve same functionality by inheriting from tf.keras.Model and overriding the train_step and test_step
functions as under the hood it is just a plain keras model.

"""

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
    
# Instantiate MovielensModel

# Print the model
print("full model build ", MovielensModel.compute_loss)

# Fitting and Evaluating 
# lets fit and evaluate the models, using optmizer adagard

model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# then shuffle, batch and cache the training and evaluation data
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

model.fit(cached_train, epochs=3)

"""
As the model trains, the loss is falling and set of top-k retrieval metrics is updated. top-k metrics
tells us whether the true positive is in the top-k retrieved items from the entire candidate set. 

For eaxmple, a top-5 categorical accuracy metric of 0.2 would tell us that, on average the true positive
is in the top 5 retrieved items 20% of the time.

Note: in this example. we evaluate the metrics during the training as well as evaluation. because this can 
quite slow with a large candidate sets, it may be prudent to turn metric calculation off in training and only
run it in evaluation.

Finally, we can evaluate our model on the test set;

"""
model.evaluate(cached_test, return_dict = True)

"""

Note: Test set performance is much worse than training performace.due to two factors
1. overfitting - model will memorize the data if it has mnay parameters, but it can be mediated by Model regularization and
use of User and movie features that help the model generalize better to unseen data.

2. Re-recommending some of users already watched movies. these can crowd out test movies out 
of top k recommendations.

WE CAN REMOVE 2ND PROBLEM BY EXCLUDING PREVIOUSLY SEEN MOVIES FROM TEST RECOMMENDATIOS, MOSTLY COMMON IN RECOMMENDER SYSTEMS LITREATURE.
but we donot fllow it in these tutorials. 
we train the model so that it can learn from past experiences and generalization rules.

"""

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


# This layer will perform approximate lookups: this makes retrieval slightly less accurate, but orders of magnitude faster on large candidate sets.
# Model serving- For Faster Execution we are using below code it may give less accuracy
# scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
# scann_index.index_from_dataset(
#   tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
# )

# # Get recommendations.
# _, titles = scann_index(tf.constant(["42"]))
# print(f"Recommendations for user 42: {titles[0, :3]}")


"""

Item-to-item recommendation
In this model, we created a user-movie model. However, for some applications (for example, product detail pages) it's common to perform item-to-item (for example, movie-to-movie or product-to-product) recommendations.

Training models like this would follow the same pattern as shown in this tutorial, but with different training data. Here, we had a user and a movie tower, and used (user, movie) pairs to train them.
In an item-to-item model, we would have two item towers (for the query and candidate item), and train the model using (query item, candidate item) pairs. These could be constructed from clicks on product detail pages.

"""