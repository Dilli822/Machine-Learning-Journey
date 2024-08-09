import tensorflow as tf 

# variable in tensor using Variable
vari = tf.Variable([1, 2, 4])
print(vari)

# constant in 1-D
oneD = tf.constant([2, 3, 4, 5])
print(oneD)

oneDi = tf.constant([0,0,0,0])
# constant 2D
twoD = tf.constant([
    [1,2,3,4], [2,3,4,5]
    ])
print(twoD)

# constant n- dimenal tensor
threeD = tf.constant( [[[
    
    [1,2,3,4,5]
]]])
print(threeD.shape)
print(threeD.ndim)

# add operation 
sum = oneD + oneDi
print("final sum is", sum)




