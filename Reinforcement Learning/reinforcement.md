
#### Reinforcement Learning 
#### It is a machine learning technique or algorithm where machine gets or rewards based on the action in a environment. Machine are trained and rewarded with negative and positive points that aid it from action moves closer to the optimal solution or goal. Q learning algorithm uses q-table with updation, maintainance and the equation is:

###### Q[state, action] = Q [state, action] + α * (reward + γ * max(Q[newState, :]) - Q[state, action])

###### where α stands for the Learning Rate 
######        Essentially rate ensures that we don't update our Q table too much on every observation.
######       γ stands for the Discount Factor -->  tries to kind of defined the balance between ###### finding really good rewards in our current state and finding the rewards in the future state.so the higher this value is the more we 're going to look towards the future. The lower it is, the more we're going to focus completely on our current reward. 
