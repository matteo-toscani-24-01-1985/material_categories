import numpy as np
import os
import time
import scipy.io
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D 
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger
from numpy.random import seed
seed(1)



################################################################################
# COMMONLY SET PARAMETERS
weights_dir = './model_data_LMT/'
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
desired_im_sz = (  47872 ,1) # channels_last format here, RGB images
nt = 1 # sequence length (i.e. single frames), not relevant but used in splitting up data
scene_data = range(0,2160 ) # how many/which of your scenes you want to use here
#proportionTrain=.9;
ntrain=1944
train_cutoff = 2160 *nt # index at which to switch from training to validation images

############################################
img_dir = './LMT_FORMATTED/' ### DATA FOLDER
###########################################

epochs = 50
batch_size = 32 #

################################################################################

tic = time.time()

################################################################################
# DEFINE DNN
################################################################################

if not os.path.exists(weights_dir): os.mkdir(weights_dir)

input_img = Input(shape=desired_im_sz)
#print "shape of input", K.int_shape(input_img)


x = Conv1D(64, kernel_size=(3), strides=(1), padding="same", activation="relu", data_format="channels_last")(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)
x = Conv1D(32, kernel_size=(3), strides=(1), padding='same', activation='relu', data_format="channels_last")(x)
x = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)
x = Conv1D(16, kernel_size=(3), strides=(1), padding="same", activation="relu", data_format="channels_last")(x)
x = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)
x = Conv1D(16, kernel_size=(3), strides=(1), padding="same", activation="relu", data_format="channels_last")(x)
encoded = MaxPooling1D(pool_size=(2), strides=(4), padding='same', data_format="channels_last")(x)

x = Conv1D(16, kernel_size=(3), strides=( 1), padding='same', activation='relu', data_format="channels_last")(encoded)
x = UpSampling1D(size=(4))(x)
# x = UpSampling2D(size=(4, 4), data_format="channels_first")(x) # for bottleneck of 25x25
x = Conv1D(16, kernel_size=(3), strides=( 1), padding='same', activation='relu', data_format="channels_last")(x)
x = UpSampling1D(size=(4))(x)
x = Conv1D(32, kernel_size=( 3), strides=( 1), padding='same', activation='relu', data_format="channels_last")(x)
x = UpSampling1D(size=( 4))(x)
x = Conv1D(64, kernel_size=(3), strides=( 1), padding="same", activation="relu", data_format="channels_last")(x)
x = UpSampling1D(size=( 4))(x)

decoded = Conv1D(1, kernel_size=(5), strides=(1), padding="same", activation="sigmoid", data_format="channels_last")(x)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(loss='mean_absolute_error', optimizer='adam')

print(autoencoder.summary())



################################################################################
# LOAD DATABASE
################################################################################

splits =scene_data

all_frames=os.listdir(img_dir) # read file names
all_frames = sorted(all_frames)
X = np.zeros((len(all_frames),) + desired_im_sz); # allocate space


for i in range(len(all_frames)):
 im_file=  all_frames[i] 
 im = np.loadtxt(os.path.join(img_dir, im_file))
 
 ###### notmalize (0 1)
 im = im-min(im)
 im = im / max(im)
 
 im=np.expand_dims(im, 1);
 X[i] = im

 
 
print(X.shape)
seed(1701) # Enterprise 1701D 
np.random.shuffle(X) # so that validation data is not all from one category



######################################
# Train model
######################################

X_val= X[ntrain:,:,:]
X= X[:ntrain,:,:]

weights_file = os.path.join(weights_dir, 'autoencoder_texture_weights_LMT.hdf5')  # where weights will be saved
json_file = os.path.join(weights_dir, 'autoencoder_texture_model_LMT.json')

csv_logger = CSVLogger(os.path.join(weights_dir, 'training_log_LMT.csv'), append=True, separator=';') # added by KS
callbacks = [csv_logger]
callbacks.append(ModelCheckpoint(filepath=weights_file, monitor='val_loss', save_best_only=True))

autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size,
               shuffle=True, validation_data=(X_val, X_val), callbacks=callbacks, verbose=1)


#### save model keras
autoencoder.save('autoencoder_model_LMT_16FEATURES.h5')
encoder.save('encoder_model_LMT_16FEATURES.h5')
#print("Training finished, took {} hours.".format((time.time() - tic)/3600))


######################## 
# RECONTRUCT SIGNALS and save them as matlab arrays
# create output folders
if not os.path.exists('./recontructed_LMT_16FEATURES/'): os.mkdir('./recontructed_LMT_16FEATURES/')
if not os.path.exists('./Code_LMT_16FEATURES/'): os.mkdir('./Code_LMT_16FEATURES/')

# Save text file with ordered file names, as they are ordered when reconstructed - for later analyses
with open('imageListLMT.txt', 'w') as f:
    for item in all_frames:
        f.write("%s\n" % item)

# decode 
decoded_imgs = autoencoder.predict(X)
### NOTE: you need the "recontructed_LMT" folder to save the signals 
for i in range(X.shape[0]):
    scipy.io.savemat("{}{}{}".format('./recontructed_LMT_16FEATURES/', i, '.mat'), mdict={'arr': decoded_imgs[i]})



############################
# SAVE LATENT REPRESENTATION
code=encoder.predict(X)
for i in range(2160):
    scipy.io.savemat("{}{}{}".format('./Code_LMT_16FEATURES/', i, '.mat'), mdict={'arr': code[i]})

