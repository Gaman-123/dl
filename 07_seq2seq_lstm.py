import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

num_samples = 1000
timesteps = 5
input_dim = 1
latent_dim = 32

X = np.random.rand(num_samples, timesteps, input_dim)
Y = np.flip(X, axis=1)

encoder_inputs = Input(shape=(timesteps, input_dim))
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(timesteps, input_dim))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(input_dim)
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='mse')

model.fit([X, Y], Y, epochs=10, batch_size=32, verbose=1)

pred = model.predict([X[:1], Y[:1]])
print("Input:\n", X[0].squeeze())
print("Predicted reversed:\n", pred[0].squeeze())
print("True reversed:\n", Y[0].squeeze())
