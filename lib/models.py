import tensorflow as tf
from tensorflow.keras import layers
import time
import math
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import pandas as pd

### -----------------------------------        GAN -----------------------------------------###
def buildGenerator(numFeatures = 240):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation = 'relu', input_shape=(numFeatures,)))

    # model.add(layers.Dense(5*5*256))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Reshape((5, 5, 256)))
    # assert model.output_shape == (None, 5, 5, 256) # Note: None is the batch size
    #
    # model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    # # assert model.output_shape == (None, 7, 7, 128)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # # assert model.output_shape == (None, 14, 14, 64)
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())
    #
    # model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    # # assert model.output_shape == (None, 28, 28, 1)

    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(numFeatures, activation = 'relu'))
    model.add(tf.keras.layers.Dense(numFeatures, activation = 'relu'))
    # model.add(tf.keras.layers.Reshape((1,239), input_shape=(239,)))
    # assert model.output_shape == (239,)

    return model

def buildDiscriminator(numFeatures):
    model = tf.keras.Sequential()


    model.add(layers.Dense(512, input_shape = (numFeatures,)))
    model.add(layers.Dense(512))
    model.add(layers.Dense(256))
    model.add(layers.Dense(128))
    model.add(layers.Dense(64))

    # model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model




class GAN:
    def __init__ (self, num_features = 240):
        self.generator = buildGenerator()
        self.discriminator = buildDiscriminator(num_features)

        # optimizers
        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999
        )

        self.BATCH_SIZE = 64


    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)

    ## Training
    def train_step(self, data):
        noise = tf.random.normal([self.BATCH_SIZE, data.shape[1]])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.generator(noise, training=True)

            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_data, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def train(self, dataset, epochs, num_samples):

        number_of_rows = dataset.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_samples, replace=False)
        testsample = dataset[random_indices, :]

        # delete test sample from Training
        dataset = np.delete(dataset, random_indices, 0)

        ## Ensure dataset is disible by batch size
        len = dataset.shape[0] / self.BATCH_SIZE
        len = math.floor(len)
        len = int(len)
        dataset = dataset[:len*self.BATCH_SIZE,:]

        # train sample
        number_of_rows = dataset.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_samples, replace=False)
        trainsample = dataset[random_indices, :]
        print("Training on {} samples divided into {} batches".format(dataset.shape[0], len))
        print("Validating on {} samples".format(testsample.shape[0]))

        noise_input = tf.random.normal([num_samples, dataset.shape[1]])
        dataset_proc =  tf.data.Dataset.from_tensor_slices(dataset).batch(self.BATCH_SIZE)

        mse_train_final = []
        mse_val_final = []
        time_final = []
        for epoch in range(epochs):
            start = time.time()

            for data_batch in dataset_proc:
                self.train_step(data_batch)

            print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
            time_final.append(time.time()-start)
            genout = self.generator.predict(noise_input)
            mse_val = mean_squared_error(testsample, genout)
            mse_train = mean_squared_error(trainsample, genout)
            print('train mse = {}     val mse = {}'.format(mse_train, mse_val))
            mse_train_final.append(mse_train)
            mse_val_final.append(mse_val)
            # discout = self.discriminator.predict(genout)

        # log mse values
        save_path = "../logs/GAN/"
        dir = os.path.dirname(__file__)
        save_path = os.path.join(dir, save_path)
        d = pd.DataFrame(data = {
        'Train MSE': mse_train_final,
        'Val MSE': mse_val_final,
        'Time (seconds)': time_final,
        })
        d.to_csv(save_path + 'gen_{}_log'.format(num_samples), index=False)

        # Get predictions
        predictions = self.generate_and_save_data(self.generator, noise_input)
        return predictions

    def generate_and_save_data(self, model, test_input, save_path = '../output/'):
        predictions = model.predict(test_input)
        dir = os.path.dirname(__file__)
        save_path = os.path.join(dir, save_path)
        np.savetxt(save_path + "generated_data_{}.csv".format(test_input.shape[0]), predictions, delimiter=",")
        return predictions



## -------------------------------------------- Neural Nets -------------------------- ##
def buildNN (data, num_hidden_layers = 4, hidden_nodes = 64, act = 'relu', do = 0, regularizer = True,
                loss_function = 'mean_absolute_error'):

    if regularizer:
        reg = tf.keras.regularizers.l2(l=0.01)
    else:
        reg = None

    NN_model = Sequential()

    # The Input Layer :
    input_layer= Input(shape=(data.shape[1],))
    NN_model.add(input_layer)

    NN_model.add(Dense(hidden_nodes, kernel_initializer='normal',activation=act, kernel_regularizer = reg))
    # The Hidden Layers :
    for i in range(num_hidden_layers-1):
        if (do>0):
            NN_model.add(Dropout(do))
        NN_model.add(Dense(hidden_nodes, kernel_initializer='normal',activation=act, kernel_regularizer = reg))


    # The Output Layer :
    NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

    # Compile the network :
    NN_model.compile(loss= loss_function, optimizer='adam', metrics=['mean_absolute_error', 'mean_squared_error'])

    return NN_model


def build_autoencoder(data, num_hidden_layers = 4, num_nodes = 32, act = 'relu'):

    model=Sequential()

    input_layer= Input(shape=(data.shape[1],))
    model.add(input_layer)

    for i in range(num_hidden_layers):
        model.add(Dense(units = num_nodes, activation=act))


    model.add(Dense(units=data.shape[1], activation=act))

    model.compile(optimizer='adam', loss= 'mean_squared_error', metrics=['mse'])

    return model



def evaluateModel (model, X, y):
    y_pred = model.predict(X)

    mse = mean_squared_error(y,y_pred)
    r2 = r2_score(y,y_pred)
    mae = mean_absolute_error(y,y_pred)

    return mse, r2, mae
