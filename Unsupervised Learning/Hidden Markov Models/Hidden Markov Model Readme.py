


# ---------- HIDDEN MARKOV MODELS --------------------
# It is a finite set of states, each of which is associated with a (generally mutli-dimensional) probability distribution [].
# Transitions among the states are governed by a set of probabilities called transition probabilities. 
# 
# A hidden markov model works with probabilities to predict future events or states, we will learn how to create 
# a hidden markov model that can predict the weather

# --------- Data ---------

# bunch of states - cold day, hot day 
# type od data to work with a hidden markov model d
# we are only interested in probability distributions unlike 
# other model we are using 100% of dataset entries


# ---------------- Components of Markov Model ------------
# 1. States: a finite number of or a set of states, like "warm", "cold", "hot", "low", "red", 
# and SO ON THESE STATES ARE HIDDEN 

# 2. Observation: Each state has a particular observation associated with it based on a probability distribution. 
# An example if it is hot day then Dilli has 80% chance of being happy and 20% chance of being sad it is observation.

# 3. Transitions: Each state will have a probability defining the likelyhood of transitioning to a different state.
# an example is th following: a cold day has a 50% change of being followed by a hot day and a 50% chance of being followed by another cold day


# Bins: bins refers to the number of intervals or bins into which the data is divided in the histogram. 
# It determines the granularity of the histogram. For example, if bins=10, the data range will be divided 
# into 10 equally spaced intervals, and the histogram will display the frequency of values falling within each interval.

# Alpha: alpha controls the transparency of the histogram bars. It accepts values between 0 and 1, where 0 means 
# fully transparent (invisible) and 1 means fully opaque (solid). Setting alpha to a value less than 1 allows you to 
# see through the histogram bars, which can be useful for visualizing overlapping data or creating layered plots.

# Density: density is a boolean parameter that determines whether the histogram should be normalized to form a 
# probability density histogram. When density=True, the height of each bin is normalized such that the total area
# under the histogram equals 1, representing the probability density function. This can be useful for comparing 
# histograms of datasets with different numbers of samples or different ranges, as it accounts for differences in data density.

