
# tutorial link:  https://www.tensorflow.org/tutorials/generative/dcgan
# https://www.youtube.com/watch?v=aircAruvnKk&t=94s
# https://www.youtube.com/watch?v=IHZwWFHWa-w&t=108s
# https://www.youtube.com/watch?v=IHZwWFHWa-w&t=108s
"""
Wrapping up 100DaysOfCode / 100DaysOfLearning 🎯

> Deep Learning is not magic but it is very good at finding patterns.
> Natural Brain Neurons vs Artifical Neural Network
> Soma adds dendrite activity together and passes it to axon.
> More dendrite activity 
https://www.youtube.com/watch?v=Y5M7KH4A4n4&t=34s
https://www.youtube.com/watch?v=bejQ-W9BGJg
"""

# >>>>>>>>>>>>>> Deep Convolutional Generative Adversarial Network <<<<<<<<<<<<<<<<<<<<<<<,
"""
>>>> This tensorflow tutorial teaches us how to generate images of handwritten digits using a Deep Convolutional
     Generative adversarial network. we are using the keras sequential api with a tf.gardienttype training loop.
     
     
# What ae GANs?
-> they are the one of the most interesting ideas in computer science today.
-> Two Models are trained simultaneously by an adversarial process.
->   A generator ("The artist") learns to create images that look real
->   A discriminator("the art critic") learns to tell images apart from fakse.
---> during the training the generator progressively becomes the better at creating images that look real, while the descriminator becomes better at telling them apart.
----> Then the process reaches equilibirum when the discriminator can no longer distinguish real images from fakes.

to learn about GAN'S http://introtodeeplearning.com/

install following packages or lib for gifs
pip install imageio
pip install git+htts://github.com/tensorflow/docs

"""
# setup 
import tensorflow as tf 
import glob 
import imageio
import matplotlib.pyplot as plt 
import numpy as np
import os
import PIL 
from tensorflow.keras import layers
import time 

from IPython import display 

# Load and prepare the dataset 
# We wil use the MNIST dataset to train the generator and the discriminator, the generator will generate handwritten digits resembing the MNIST Data
(train_images, train_labels) , (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5   # Normalize the images to [-1, 1]

# Shuffling data for training
BUFFER_SIZE = 60000
BATCH_SIZE = 256 

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""
Create the Models
> we are defining both models generator and discriminator using the keras sequential api .

>>>>> THE GENERATOR :
> The generator uses tf.keras.Conv2DTranspose (upsampling) layers to produce an image from a seed (random noise). 
   1. start with a dense layers
   2. take the seed as input
   3. upsample several times 
   4. until we reach the desired image size of 28 x 28 x1
   5. Notice the tf.keras.layers.LeakyReLu activation for each layer
   6. except the output layer which uses tanh
   
"""

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# use untrained generator model to create an image 
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.axis('off')  # Hide the axis
plt.show()


# Discriminator model is a CNN basedimage classfier 
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
        

# lets test before training to classify the generated images as real or fake, 
# ----- for real images the values are positive values and for fake images the values are negative values

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)


"""
Defining the loss and optimizers
> define loss functions and optimizr for both models
"""

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss
"""
> Discrimator Loss 
> This method quantifies how well the discriminator is able to distinguish real images from fakes
> It compared the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fakse(Generatated) images to an array of 0s
"""

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


"""
Generator Loss
> The generator's loss qunatifies how well it was able to trick the discriminator. 
intutitively,if the generator is performing well, the discriminator will classify the fake images as real(or 1)
Hence, compare the discriminators decisions on the generated images to an array of 1s
"""

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# The discriminator and the generator optimizers are different since you will train two networks separately 
generator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.legacy.Adam(1e-4)

# Save the checkpoints
# checkpoints save and restore the models which is helpful in the case a long running training task is interrupted
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Defining the training loop
EPOCHS = 5
noise_dim = 100
num_examples_to_generatre = 16

# we will reuse the seed overtime to visualize progress in the animated GIF so it will be interesting  

"""
Steps involved in training loop
1. training loop begins with generator receiving a random seed as input.
2. seed is uaed to produce an image
3. discriminator classify the real images(drawn from the training set) and fakes (produced by the generator)
4. The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator.

A gradient simply measures the change in all weights with regard to the change in error. You can also think of a gradient as the slope of a function. The higher the gradient, 
the steeper the slope and the faster a model can learn. But if the slope is zero, the model stops learning.

"""

# Random seed for generating images
seed = tf.random.normal([num_examples_to_generatre, noise_dim])

@tf.function # this will cause the fucntion to be compiled
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  
# Generate and save images 
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
    
"""
Train the model:

"""

train(train_dataset, EPOCHS)

checkpoint.restore(tf.train_latest_checkpoint(checkpoint_dir))

# display a single image using the epoch number
def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png', format(epoch_no))

display_image(EPOCHS)
# using image.io to create an animaretd gif using the images saved during the training 

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

