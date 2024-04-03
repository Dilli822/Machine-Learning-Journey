from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

path_to_file = tf.keras.utils.get_file('shakespare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

vocab = sorted(set(text))
char2idx = { u:i for i, u in enumerate(vocab) }
idx2char = np.array( vocab )

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

def int_to_txt(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

seq_length = 100
exmples_per_epoch = len(text) // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

# spliting input target into two hello to hell and ello 
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_txt(x))
    print("\nOUTPUT")
    print(int_to_txt(y))


BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
# building models here 
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model 

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch) # ask our model for a prediction on our first batch of training data
    print(example_batch_predictions.shape, "# (btach_suze, sequence_length, vocab_size)" ) # print out the output shape

# we can see that the predictions is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)

# lets example one prediction 
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# and of course its 65 values representing the probability of each character occuring next

time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probability of each character occuring next

# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probabilities)
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1)) [0]
predicted_chars = int_to_txt(sampled_indices)

predicted_chars # this is what the model predicted for traning sequence 1
print(predicted_chars) # gives any random predicted values

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)


model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_perfix = os.path.join(checkpoint_dir, "ckpt_(epoch)")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_perfix,
    save_weights_only = True
)

# Training : Finally we will start training the model
# if this is taking a while go to Runtime > Change Runtime Type and Choose GPU under hardware accelerator

history = model.fit(data, epochs=1, callbacks=[checkpoint_callback])



# Loading the Model
# We'll rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one piece of text to the model and have it make a prediction.

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# Once the model is finished training we can find the latest checkpoint that stores the models weights the following line
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# We can load any checkpoint we want by specifying the exact file to load 
checkpoint_num = 10


# Generating the text
def generate_text(model, start_string):
    # Evaluate step (Generating text using the learned model)
    
    # Number of characters to generate
    num_generate = 800
    
    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    [9,8,7]
    input_eval = tf.expand_dims(input_eval, 0)
    
    # empty string to store our results
    text_generated = []
    
    # Low temperature results in more predictable text.
    # Higher temperature results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0
    
    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)
        
        # Using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        
        # we pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        
        text_generated.append(idx2char[predicted_id])
        
    return (start_string + ''.join(text_generated))
        
inp = "romeo"
print(generate_text(model, inp))