

# since we are dealing with the tensorflow probability
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import tensorflow.compat.v2 as tf
import matplotlib.pyplot as plt
# Weather Prediction Model
# 1. Cold Days are encodede by a 0 and hot days are encodede by 1
# 2. The first day in out sequence has an 80% chance of being cold
# 3. A cold day has a 30% chance of being followed by a hot day 
# 4. A hot day has a 20% chance of being followed by a cold day
# 5. On each day th temperature is normally distributed with the mean and 
#     the standard deviation 15 and 10 on a hot day.


tfd = tfp.distributions


# starting 0.2 % of cold and 0.8% of hot
present_day_hot_percentage = 0.8
remaining_present_day_hot_percentage = 0.2

# chances of  0.3 % of cold and 0.7% of hot
transition_distribution_percentage = 0.7
remain_transition_distribution_percentage = 0.3

initial_distribution = tfd.Categorical(probs =[ remaining_present_day_hot_percentage, present_day_hot_percentage])
transition_distribution = tfd.Categorical(probs=[
    [transition_distribution_percentage , remain_transition_distribution_percentage],
    [remaining_present_day_hot_percentage,present_day_hot_percentage]
])
# note we are using 0. or 10. just making sure our data types doesnot mismatch
# .Normal terms refers to the statistics concept
observation_distribution = tfd.Normal(loc = [0. , 35.], scale = [5., 10.] ) # refers to point 3 and 4 above and points 5 above

# loc argument represents the mean and the scale is the standard deviation

# ---------- MODEL HERE. MAGIC HAPPENS HERE -------------
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7     # num_steps = how many days or the cycle we want to predict
)


# calculate the probabilities from the model
mean = model.mean()
print("Mean or preedicted weather temperature for 7 days --> ")
print( mean.numpy())


plt.plot(mean, marker="o", color="red")
plt.grid(True)
plt.title("7 Days of Weather Prediction ForeCast ")
plt.legend(["Less than to 20 indicated cold ", "more than or nearer to 25 indicates hot day"])
plt.show()

