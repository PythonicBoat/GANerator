import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Define the generator
def make_generator_model():
    model = keras.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Dense(28*28*1, use_bias=False, activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))

    return model

# Define the discriminator
def make_discriminator_model():
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(1))

    return model

# Load real data
real_data = np.load('real_data.npy')

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Define the discriminator and generator models
generator = make_generator_model()
discriminator = make_discriminator_model()

# Define the optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training loop
@tf.function
def train_step(real_images):
    # Generate random noise
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images from the noise
        fake_images = generator(noise, training=True)

        # Evaluate the discriminator on real and fake images
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        # Compute the generator loss
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        # Compute the discriminator loss
        disc_loss_real = cross_entropy(tf.ones_like(real_output), real_output)
        disc_loss_fake = cross_entropy(tf.zeros_like(fake_output), fake_output)
        disc_loss = disc_loss_real + disc_loss_fake

    # Compute the gradients and apply to the optimizer
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train the GAN
EPOCHS = 100
BATCH_SIZE = 256

for epoch in range(EPOCHS):
    for i in range(real_data.shape[0] // BATCH_SIZE):
        # Select a random batch of real images
        real_images = real_data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        # Train the GAN on the batch of real images
        train_step(real_images)

    # Print the progress
    if epoch % 10 == 0:
        print('Epoch:', epoch)

# Generate synthetic data
num_samples = 1000
noise = tf.random.normal([num_samples, 100])
synthetic_data = generator(noise, training=False).numpy()

# Save the synthetic data
np.save('synthetic_data.npy', synthetic_data)
