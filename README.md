# GANerator

This is a basic implementation of a GAN (Generative Adversarial Network) using TensorFlow 2. The code defines a generator model and a discriminator model using Keras Sequential API. The generator takes random noise as input and generates synthetic data. The discriminator takes a sample of real data and synthetic data as input, and predicts whether the input is real or synthetic. The loss function for the generator is binary cross-entropy with the target label set to 1 for all synthetic data. The loss function for the discriminator is the sum of binary cross-entropy with the target label set to 1 for real data and 0 for synthetic data.

The code also defines an Adam optimizer for both the generator and the discriminator and trains the GAN by alternating between training the generator and training the discriminator using batches of real data. After training, the code generates synthetic data by feeding random noise to the generator and saves the synthetic data to a file.
