# CGAN.Developed by EM Wang / 2023.6.30 / QQ:326496053
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D,Conv2DTranspose,AveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import os
from array_data_2D import *

# GPU selection 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CGAN():
    def __init__(self):
        # initial parameters
        self.img_shape = (64, 64, 1)
        self.latent_dim = 100
        self.data_num = 2400
        self.train_dir = 'D:/'
        self.batch_size = 32
        self.iter = 500001
        self.save_interval = 200
        
        # pre-training epoch 
        self.pre_training = 501
        
        # optimizer selection: Adam
        D_optimizer = Adam(5e-5)
        G_optimizer = Adam(5e-5)

        # model compile: discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=D_optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()

        # basic noise input: latent_dim=100
        noise = Input(shape=(self.latent_dim,))
        # generate the images by generator
        img = self.generator(noise)
        # self.discriminator.trainable = False
        # discriminate the real and fake images by discriminator, and return a parameter
        valid = self.discriminator(img)
        self.combined = Model(noise, valid)
        # model compile generator
        self.combined.compile(loss='binary_crossentropy', optimizer=G_optimizer)

    def build_generator(self):
        # build the generator model
        model = Sequential()
        model.add(Dense(256 * 4 * 4,  input_dim=self.latent_dim))
        # batch normalization operation
        model.add(BatchNormalization(momentum=0.9))
        # leaky ReLU function
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        model.add(UpSampling2D())

        # size 8*8
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())

        # size 16*16
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())

        # size 32*32
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(UpSampling2D())

        # size 64*64
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(1, kernel_size=3, strides=1, padding="same", activation='tanh'))
        model.summary()

        # input noise
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        
        return Model(noise, img)

    def build_discriminator(self):
        # build the discriminator model
        model = Sequential()
        # size 64*64
        model.add(Conv2D(16, kernel_size=3, strides=1, input_shape=self.img_shape, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        # drop out operation to reduce the over-fitting
        model.add(Dropout(0.25))
        model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # size 32*32
        model.add(Conv2D(32, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # size 16*16
        model.add(Conv2D(64, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # size 8*8
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # size 4*4
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.9))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        # flatten and dense layer
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        # input images
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self):
        # train the CGAN model
        # basic file input and initial parameters
        train_img = []
        print('---File input---')
        # image input and resize
        for i in range(1, self.data_num + 1):
            img = Image.open("{}.png".format(self.train_dir + str(i)))
            img = img.resize((64,64))
            train_img.append(img_to_array(img))

        # data normalization
        model_training_data = np.array(train_img)
        print(model_training_data.shape)
        X_train = model_training_data / 127.5 - 1.

        # generate 1 & 0 matrix to discriminate the real and fake images
        valid = np.zeros((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))

        # pre-training for discriminator
        self.discriminator.trainable = True

        for i in range(self.pre_training):
            # batch images selection
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            # input random noise to generator to generate fake images
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # discriminator loss calculation: d_loss_real, d_loss_fake and total d_loss
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # discriminator loss visualization
            if i % self.save_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] " % (i, d_loss[0], 100 * d_loss[1]))

        print('---Pre-training complete---')
        print('---Generator training start---')

        # basic training
        self.discriminator.trainable = False

        for i in range(self.iter):
            # batch images selection
            idx = np.random.randint(0, X_train.shape[0], self.batch_size)
            imgs = X_train[idx]

            # input random noise to generator to generate fake images
            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # discriminator loss calculation: d_loss_real, d_loss_fake and total d_loss
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # generator loss calculation
            g_loss = self.combined.train_on_batch(noise, valid)

            # loss visualization
            if i % self.save_interval == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100 * d_loss[1], g_loss))
                self.img_export(i)

    def img_export(self,iter):
        # export the images every 200 epoches during training
        # set image number: 25 (5*5)
        r,c = 5,5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        # images regenerate by img matrix
        gen_imgs = 0.5 * gen_imgs + 0.5
        width, height = self.img_shape[0],self.img_shape[1]
        # arrange the generated images in one image set
        new_image = Image.new('L', (20+5 * width + 4 * 10, 20+5 * height + 4 * 10), 255)
        # generate images arrangement
        for i in range(5):
            for j in range(5):
                index = i * 5 + j
                if index < len(gen_imgs):
                    x = j * (width + 10)+10
                    y = i * (height + 10)+10
                    new_arr = (gen_imgs[index, :, :, 0] * 255.).astype(np.uint8)
                    img =  Image.fromarray(new_arr,mode='L')
                    new_image.paste(img, (x, y,x+width,y+height))
        # image sets save
        new_image.save("{}.png".format(str(iter)))

if __name__ == '__main__':
    cgan = CGAN()
    cgan.train()
