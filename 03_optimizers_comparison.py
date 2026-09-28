import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)


def create_model():
    model = keras.Sequential([
        layers.Dense(4, input_dim=2, activation='tanh'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def train_model(optimizer_name, X, y, epochs=5000, lr=0.1):
    model = create_model()
    if optimizer_name == 'SGD':
        optimizer = optimizers.SGD(learning_rate=lr)
    elif optimizer_name == 'Momentum':
        optimizer = optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        optimizer = optimizers.Adam(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    model.compile(optimizer=optimizer, loss=losses.BinaryCrossentropy())
    history = model.fit(X, y, epochs=epochs, verbose=0)
    return history.history['loss']


losses_sgd = train_model('SGD', X, y)
losses_momentum = train_model('Momentum', X, y)
losses_adam = train_model('Adam', X, y)

plt.figure(figsize=(8, 5))
plt.plot(losses_sgd, label='SGD')
plt.plot(losses_momentum, label='Momentum')
plt.plot(losses_adam, label='Adam')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimization Algorithms Comparison (XOR Problem)')
plt.show()

print(f"Final Loss (SGD): {losses_sgd[-1]:.6f}")
print(f"Final Loss (Momentum): {losses_momentum[-1]:.6f}")
print(f"Final Loss (Adam): {losses_adam[-1]:.6f}")
