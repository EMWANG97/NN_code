# 2D-CNN.Developed by EM Wang / 2023.6.30 / QQ:326496053
import csv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Activation,Dropout,Flatten,Dense,AveragePooling2D,MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras import regularizers
from array_data_2D import *
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from PIL import Image

# GPU selection
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# initial parameters
train_dir = 'D:/training_data/'
working_path='D:/2D-CNN/'
os.chdir(working_path)
batch_size = 32
twoD_model_size = 64
num_channels = 1
epochs = 20000
data_num = 2400
para_num = 200
multi_model = True
ultra_multi_model = False

model = Sequential()
# size 64*64
model.add(Conv2D(input_shape=(twoD_model_size,twoD_model_size,num_channels),filters=16,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=16,kernel_size=3,padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# size 32*32
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# size 16*16
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# size 8*8
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=128,kernel_size=3,padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

# size 4*4
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(Conv2D(filters=256,kernel_size=3,padding='same',activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(2048,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(2048,activation='relu',kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(200,activation='linear'))

# optimizer
adam = Adam(learning_rate=1e-5)

# compile
model.compile(optimizer=adam,loss='mae')

# file input
print('\n'+'-----File input-----')

# basic training
train_img=[]
for i in range (1,data_num+1):
    img = Image.open("{}.png".format(train_dir+str(i)))
    img = img.resize((64, 64))
    train_img.append([img_to_array(img)/255.])
model_training_data = train_img
model_training_data = np.concatenate(model_training_data)
para_training_data = array_para_data("{}.csv".format(train_dir +'training_data_para'),para_num)
model_training_data,para_training_data = shuffle(model_training_data, para_training_data)
print('-----File input complete-----')
print('The shape of batch:',model_training_data.shape)
print('The shape of para:',para_training_data.shape)

# model training by using "fit" function
model_fit = model.fit(model_training_data, para_training_data, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.2)
model.evaluate(model_training_data, para_training_data, batch_size=batch_size)

# model evaluate 
# export the training loss to csv. files
loss_list = model_fit.history['loss']
val_loss_list = model_fit.history['val_loss']
loss = zip(loss_list,val_loss_list)
with open('loss.csv',"w",newline='') as loss_csv:
    writer = csv.writer(loss_csv)
    for row in loss:
        writer.writerow(row)

# model save
model.save('model_2Dcnn.h5')
