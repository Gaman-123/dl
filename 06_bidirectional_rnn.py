import tensorflow as tf
import numpy as np

seq_length = 10
num_samples = 1000

X = []
y = []
for _ in range(num_samples):
    start = np.random.randint(0, 100)
    seq = np.arange(start, start + seq_length)
    X.append(seq[:-1])
    y.append(seq[1:])

X = np.array(X, dtype=np.float32)[..., np.newaxis]
y = np.array(y, dtype=np.float32)[..., np.newaxis]


def build_simple_rnn(input_size=1, hidden_size=32, output_size=1):
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(
            hidden_size,
            return_sequences=True,
            input_shape=(seq_length - 1, input_size)
        ),
        tf.keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    return model


def build_birnn(input_size=1, hidden_size=32, output_size=1):
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.SimpleRNN(hidden_size, return_sequences=True),
            input_shape=(seq_length - 1, input_size)
        ),
        tf.keras.layers.Dense(output_size)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss='mse')
    return model


def train_model(model, X, y, epochs=50):
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return history.history['loss'][-1]


rnn_model = build_simple_rnn()
birnn_model = build_birnn()

rnn_loss = train_model(rnn_model, X, y)
birnn_loss = train_model(birnn_model, X, y)

print(f"Final Loss (RNN): {rnn_loss:.4f}")
print(f"Final Loss (BiRNN): {birnn_loss:.4f}")
