import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
batch_size = 128
buffer_size = 60000
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size).batch(batch_size)


def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 128, input_dim=100),
        layers.LeakyReLU(0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(64, (5, 5), strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(1, (5, 5), strides=2, padding="same", activation="tanh")
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=2, padding="same", input_shape=[28, 28, 1]),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid")
    ])
    return model


generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
gan_output = discriminator(generator(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(loss="binary_crossentropy", optimizer="adam")

epochs = 5000
for step in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake = generator.predict(noise, verbose=0)
    real = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
    X = np.concatenate([real, fake])
    y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    d_loss, _ = discriminator.train_on_batch(X, y)

    noise = np.random.normal(0, 1, (batch_size, 100))
    y_gen = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, y_gen)

    if step % 500 == 0:
        print(f"Step {step} | D loss: {d_loss:.3f} | G loss: {g_loss:.3f}")
        z = np.random.normal(0, 1, (1, 100))
        gen_img = generator.predict(z, verbose=0)[0, :, :, 0]
        plt.imshow(gen_img * 0.5 + 0.5, cmap="gray")
        plt.axis("off")
        plt.show()
