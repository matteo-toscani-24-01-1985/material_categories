from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Conv1DTranspose,Layer
from keras.models import Model
from random import seed,random
import numpy as np
import matplotlib.pyplot as plt
#import argparse
import os
import tensorflow as tf




###############################
# DEFINE COMMON VARIABLES
CropSize=1500; #### FOR RANDOM CROPPING DATA AUGMENTATION

desired_im_sz = (  int(CropSize) ,1) # 
nt = 1 # sequence length (i.e. single frames), not relevant but used in splitting up data
scene_data = range(0,2000 ) # how many/which of your scenes you want to use here
ntrain=19000
train_cutoff = 2000 *nt # index at which to switch from training to validation images

#############################
img_dir = './LMT_FORMATTED/' # DATA FOLDER
#############################


#####################################
# load randomly croppoed database
#####################################
all_frames=os.listdir(img_dir)
all_frames = sorted(all_frames)

seed(1789) # 14/7
RandStartMax = 47872-CropSize
NSamps=20000
X = np.zeros((int(NSamps),) + desired_im_sz);
for i in range(NSamps):
 imind=int(random() *len(all_frames))  
 im_file=  all_frames[imind] 
 im = np.loadtxt(os.path.join(img_dir, im_file))
 start=int(random()*RandStartMax)
 end=start+CropSize
 im=im[start:end]
 ###### notmalize (0 1)
 im = im-min(im)
 im = im / max(im)
 im=np.expand_dims(im, 1);
 X[i] = im
 


#############
# assign train and test datasets
x_train=X[0:ntrain]
x_test=X[ntrain:]


##############################
## DEFINE THE CUSTOM SAMPLIGN LAYER
class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



###################################
# BUILD NETWORK
###################################
latent_dim =16

#encoder
encoder_inputs = Input(shape=desired_im_sz)
x = Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
x = Flatten()(x)
x = Dense(16, activation="relu")(x)
z_mean = Dense(latent_dim, name="z_mean")(x)
z_log_var = Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder =Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

# decoder
latent_inputs = Input(shape=(latent_dim,))
x = Dense((int(CropSize/4)) * 64, activation="relu")(latent_inputs)
x = Reshape((int(CropSize/4) , 64))(x)
x = Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()



#############################################################
## Define the VAE as a `Model` with a custom `train_step`
#############################################################
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 47872
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


vae = VAE(encoder, decoder)

# format data according to the definition of VAE
data = np.concatenate([x_train, x_test], axis=0)


#######################################################
############ COMPILE AND TRAIN THE VAE
#######################################################
vae.compile(optimizer=keras.optimizers.Adam())

vae.fit(data, epochs=30, batch_size=8)

subset= data[0:10] # subset to plot and look at recontructions
code=encoder(subset)
decoded=decoder(code)
for i in range(10):
    plt.figure()
    plt.plot(subset[i,1:1000],'k')
    plt.plot(decoded[i,1:1000],'r')
    
###########################################
# save MODEL KERAS
decoder.save('decoder_model.h5')
encoder.save('encoder_model.h5')
vae.save_weights('vae_model.h5')

###########################################
# SAVE RECONSTRCUTED (FOR COMPUTING Rsq later) and latent representation  

### load original data (not augmented)
ALL_DATA=np.zeros((len(all_frames),) + desired_im_sz)
for i in range(len(all_frames)):
 im_file=  all_frames[i] 
 im = np.loadtxt(os.path.join(img_dir, im_file))
 start=int((im.shape[0]/2)-CropSize/2)
 end=start+CropSize
 im=im[start:end]
 ###### notmalize (0 1)
 im = im-min(im)
 im = im / max(im)
 im=np.expand_dims(im, 1);
 ALL_DATA[i]=im  
 
Code = encoder(ALL_DATA) 
Reconstructed =decoder(Code)

## write order filed as they are saved 
with open('imageListLMT.txt', 'w') as f:
    for item in all_frames:
        f.write("%s\n" % item)
        
for i in range(10):
    plt.figure()
    plt.plot(ALL_DATA[i],'k')
    plt.plot(Reconstructed[i],'r')

# create output folders
if not os.path.exists('./CodeVAE16D/'): os.mkdir('./CodeVAE16D/')
#### save latent    
import scipy.io   
for i in range(np.shape(Code)[1]):
    tmp=Code[2][i]
    scipy.io.savemat("{}{}{}".format('./CodeVAE16D/', i, '.mat'), mdict={'arr': tmp.numpy()})

for i in range(np.shape(Reconstructed)[1]):
    tmp=Reconstructed[i]
    scipy.io.savemat("{}{}{}".format('./ReconstructedVAE16D/', i, '.mat'), mdict={'arr': tmp.numpy()})


