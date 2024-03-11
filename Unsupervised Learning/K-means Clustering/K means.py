
# UNSUPERVISED LEARNING
# K-means clustering is one of the simplest and popular
# unsupervised machine learning algorithms. Typically, 
# unsupervised algorithms make inferences from datasets 
# using only input vectors without referring to known, or labelled, outcomes.

# Clustering ---> clustering technique is used when we have no labeled data like we did in titanic dataset
# and iris flower classification
# it group similar or related data but having highly different variation in the data points

# --------------------- BASIC ALGORITHM FOR K-MEANS -------------------

# Step 1: randomly pick k points to place K centroids
# Step 2: assign all of the data points to the centroids by distance. The closest
# centroid to a point is the one it is assigned to.
# Step 3: Average all of the points belonging to find the middle of
# those clusters(center of mass). place the corresponding centroids into that position
# Step 4: Reassign every point once to the closes centroid
# Step 5: Repeat steps 3-4 unitl no point changes which centroid it belongs to.



# ROugh steps
# 1. draw 2d data points x and y axis
# 2. choose finite k(cluster) it can k = 3 or k = 4 k = 2 must be more than 2 or 3
# 3. then place centeroid randomly in the data points
# 4. now calculate the euclidean distance between each individual point and the centroid 
# 5. label the closet data points with the name of closes centroid for an example if data point is near a centroid A 
#     then label that point as A similarly for all data points
# 6. Find the center of mass for all same labelled data points for eg lets say center of mass for A labelled data points
#     and again place the centroid at the center of the mass at the center
# 7. Then again remove all the labels and repeat process step3 
# 8. UNTIL EVENTUALLY ALL THE DATA POINTS OR THE NONE OF THE INDIVIDUAL DATA POINTS CENTROID ARE CHANGING
# 9. Then we get FINALLY A CLUSTER NEW DATA POINTS

import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, max_iterations=1000):
        self.k = k
        print("Received K is -->",  self.k)
        self.max_iterations = max_iterations
        print("max iteration K is -->",  self.max_iterations)

    def fit(self, data):
        # Step 1: Randomly initialize centroids
        centroids = data[np.random.choice(data.shape[0], self.k, replace=False)]
        print("centroids in fit is ", centroids)
        
        for _ in range(self.max_iterations):
            # Step 2: Assign each data point to the nearest centroid
            labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
            
            # Step 3: Update centroids to the mean of data points assigned to them
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.k)])
            print("Updated new centroids is ", new_centroids)
            
            # Check for convergence
            if np.all(centroids == new_centroids):
                print("Centroids and new centroids matched no more matching can be made ")
                break
            
            centroids = new_centroids
        
        self.labels_ = labels
        self.centroids_ = centroids

    def predict(self, data):
        return np.argmin(np.linalg.norm(data[:, np.newaxis] - self.centroids_, axis=2), axis=1)


# Generate random data
np.random.seed(0)
data = np.random.rand(100, 2)

# Normalize the data
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# Initialize KMeans model
kmeans = KMeans(k=5)

# Fit the model to the data
kmeans.fit(data)

# Predict the clusters for the data
predicted_labels = kmeans.predict(data)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=predicted_labels, cmap='viridis')
plt.scatter(kmeans.centroids_[:, 0], kmeans.centroids_[:, 1], marker='x', color='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid()
plt.show()


#https://chat.openai.com/share/9eed9153-8929-41ca-a34a-5df33b7e26c6