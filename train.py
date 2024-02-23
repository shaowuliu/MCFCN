
import time
# import torch
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow import keras
# import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers

import tensorflow as tf
# from sklearn.utils import class_weight
# from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt
import pydicom
import cv2
#from lungmask import mask #lung segmentation model
#import SimpleITK as sitk
# # import pandas as pd
# from PIL import Image
data_path = r'autodl-nas/covid/P019'
# Set the cut-off probability for the classification output (Default : 0.5)
cutoff = 0.5

K.set_image_data_format('channels_last') # 彩色图像的性质一般包括：width、height、channels 选择channels_last：返回(256,256,3)

def squash(x, axis=-1):
    print("squash")
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x

def focal_loss(y_true, y_pred):
    print("focal loss")
    loss=0
    a= 0.25
    r = 2
    for i in range(2):
        loss+= -a*(1-y_pred[:,i]) ** r * y_true[:,i] * tf.log(y_pred[:,i]) - (1-a) * y_pred[:,i]**r*(1-y_true[:,i])*tf.log(1-y_pred[:,i])
    return loss

def softmax(x, axis=-1):
    print("softmax")
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)



def margin_loss(y_true, y_pred):
    print("loss")
    loss = 0
    lamb, margin = 0.5, 0.1
    for i in range(2):
        loss+= K.sum((y_true[:,i] * K.square(K.relu(1 - margin - y_pred[:,i])) + lamb * (
        1 - y_true[:,i]) * K.square(K.relu(y_pred[:,i] - margin))), axis=-1)
    return loss



class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
def sel(filenameDCM):
    ArrayDicom = np.zeros((1,512,512)) # 这里面存的才是基础的ct的图片
    lung_mask = np.uint8(np.zeros((1,512,512))) # unit8创建图像容器
    #model = mask.get_model('unet','R231CovidWeb')
    i=0
    segmentation = np.zeros((1,512,512))
    #ds = sitk.ReadImage(filenameDCM)
    
    print(filenameDCM)
    segmentations = cv2.imread(filenameDCM);
    # plt.imshow(segmentations)
    # plt.savefig('/root/data/test.png')
    # plt.close
    print(segmentations.shape)
    segmentation[0] = segmentations[:,:,0]
    #segmentation = mask.apply(ds, model) # segmentation 0 or 1 (1, 512, 512)
        # 这个模型只是把肺部的位置提取出来了，提取出只包含 -1 0 1的数组
    lung_mask[0,:,:] = np.uint8(((segmentation>0)*1)[0]) #uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
        # lung_mask就是把 -1 去掉，然后以图片的形式保存这个数组
    #ArrayDicom[i, :, :] = sitk.GetArrayFromImage(segmentation)
    lungs = np.zeros((1,256,256))

    ct = normalize_image(segmentation)
    #print('ct',ct.shape)
    mask_l = lung_mask[0,:,:]
    #print(mask_l.shape)
    seg = mask_l * ct[0] #apply mask on the image  seg.shape (512, 512)
        # mask_l 是分割后的肺部阴影图  ct是 每个肺部ct
    #print(seg.shape)
    #print(seg.shape)
    seg = normalize_image(seg)
    img = cv2.resize(seg,(256,256)) # 512X512 到 256X256
    seg = cv2.resize(seg,(128,128))
     #img.shape (256, 256)
    #print(img.shape)
    #lungs[0,:,:] = img
	
    return img,seg    
# normalization function
def normalize_image(x): #normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))
print('GPU找到了马，能不能使用呢')
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_logical_devices('GPU')
strategy = tf.distribute.MirroredStrategy(gpus)
# with strategy.scope():  
with tf.device('/gpu:2'):     
    input256 = Input(shape=(None, None, 1), name = 'input256')
    #  输入的大小是(116, 256, 256, 1)
    x = Conv2D(64, (3, 3), activation='relu',trainable = True)(input256)
    #  卷积一下(116, 254, 254, 64)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x)
    #   batch之后(116, 254, 254, 64)
    x = Conv2D(64, (3, 3), activation='relu',trainable = True)(x)
    #   第二次卷积 (116, 252, 252, 64)
    x = MaxPooling2D((2, 2))(x)
    #   最大池化(116, 126, 126, 64)
    x = Conv2D(128, (3, 3), activation='relu',trainable = True)(x)
    #   再卷积一次(116, 124, 124, 128)
    x = Conv2D(128, (3, 3), activation='relu',trainable = True)(x)
    #   (116, 122, 122, 128)  先不看116这个数   变成了 122x122x128
    x = Reshape((-1, 128))(x)
    #   不知行数，分成128列 输出x_capsule.shape (116, 14884, 128)  为什么显示出来的是（？，？，128）这个14884为什么不显示呢？
    #   把16个特征图展开成为一维  所以应该是 14884x8的高,16的底的长方形 然后合并成 14884x8个胶囊,每个胶囊维度为16
    x = Capsule(32, 16, 3, True)(x)
    #  (116, 32, 16) 1行就是一个胶囊  16维胶囊  32个胶囊
    x = Capsule(32, 16, 3, True)(x)
    # output 是 (116, 32, 16)
    # capsule = Capsule(2, 16, 3, True)(x)
    # # 输出是(116, 2, 16)
    # output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule) # (116,2)
    #下面是测试三分类的效果
    capsule = Capsule(3, 16, 3, True)(x)

    input512 = Input(shape=(None, None, 1), name = 'input512')
    #  输入的大小是(116, 256, 256, 1)
    x2 = Conv2D(64, (3, 3), activation='relu',trainable = True)(input512)
    #  卷积一下(116, 254, 254, 64)
    x2 = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)(x2)
    #   batch之后(116, 254, 254, 64)
    x2 = Conv2D(64, (3, 3), activation='relu',trainable = True)(x2)
    #   第二次卷积 (116, 252, 252, 64)
    x2 = MaxPooling2D((2, 2))(x2)
    #   最大池化(116, 126, 126, 64)
    x2 = Conv2D(128, (3, 3), activation='relu',trainable = True)(x2)
    #   再卷积一次(116, 124, 124, 128)
    x2 = Conv2D(128, (3, 3), activation='relu',trainable = True)(x2)
    #   (116, 122, 122, 128)  先不看116这个数   变成了 122x122x128
    x2 = Reshape((-1, 128))(x)
    #   不知行数，分成128列 输出x_capsule.shape (116, 14884, 128)  为什么显示出来的是（？，？，128）这个14884为什么不显示呢？
    #   把16个特征图展开成为一维  所以应该是 14884x8的高,16的底的长方形 然后合并成 14884x8个胶囊,每个胶囊维度为16
    x2 = Capsule(32, 16, 3, True)(x2)
    #  (116, 32, 16) 1行就是一个胶囊  16维胶囊  32个胶囊
    x2 = Capsule(32, 16, 3, True)(x2)
    # output 是 (116, 32, 16)
    # capsule = Capsule(2, 16, 3, True)(x)
    # # 输出是(116, 2, 16)
    # output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule) # (116,2)
    #下面是测试三分类的效果
    capsule2 = Capsule(3, 16, 3, True)(x2)
    # 输出是(116, 2, 16)
    output = keras.layers.Add()([capsule,capsule2])
    outputs = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)), name='outputs')(output) # (116,2)

    model = Model(inputs=[input256,input512], outputs=[outputs])
    adam = optimizers.Adam(lr=1e-4)
    model.compile(loss = {'outputs':"binary_crossentropy"},
            optimizer=adam, metrics=['accuracy'])
    model.summary()        
    all_slices = np.zeros(shape=(19810,256,256,1)) #20679
    all_slices_512 = np.zeros(shape=(19810,128,128,1))
    all_index = np.zeros(shape=(19810,3)) 
    slices_index = 0
    covid_path = 'database/curated_data/curated_data/2COVID'
    covid_text = '../database/meta_data_covid.csv'
    files = os.listdir(covid_path)
    for file in files:
            segmentations = cv2.imread(covid_path+"/"+file)
            if segmentations.shape[0]==512:
		#normal_img = cv2.imread(covid_path+"/"+file);
		#normal_img2 = cv2.resize(normal_img[:,:,0],(256,256))
		#normal_img2 = normalize_image(normal_img2)
                normal_img2,normal_img_512 = sel(covid_path+"/"+file)
                all_slices[slices_index,:,:,0] = normal_img2
                all_slices_512[slices_index,:,:,0] = normal_img_512
                all_index[slices_index,2] = 1
                slices_index+=1
    normal_path = 'database/curated_data/curated_data/1NonCOVID'
    normal_text = '../database/meta_data_normal.csv'
    slices = 0
    files = os.listdir(normal_path)
    for file in files:
            normal_img2,normal_img_512 = sel(normal_path+"/"+file)
            all_slices[slices_index,:,:,0] = normal_img2
            all_slices_512[slices_index,:,:,0] = normal_img_512
            all_index[slices_index,0] = 1
            slices_index+=1
    cap_path = 'database/curated_data/curated_data/3CAP'
    cap_text = '../database/meta_data_cap.csv'
    files = os.listdir(cap_path)
    for file in files:
            normal_img2,normal_img_512 = sel(cap_path+"/"+file)
            all_slices[slices_index,:,:,0] = normal_img2
            all_slices_512[slices_index,:,:,0] = normal_img_512
            all_index[slices_index,1] = 1 
            slices_index+=1	
    history = model.fit(x={'input256':all_slices,'input512':all_slices_512},
    y={'outputs':all_index},
        batch_size=16,
        epochs=100,
        shuffle=True,
        validation_split= 0.1,
        #verbose=2,
        #       validation_data = ([x_train], [y_train])
    )

    #1114/1115 [============================>.] - ETA: 0s - loss: 0.0203 - accuracy: 0.90272022-12-22 00:54:14.426627: 
    #1114/1115 [============================>.] - ETA: 0s - loss: 0.0290 - accuracy: 0.91112023-01-12 19:39:39.627
    model.save_weights("new_data_train_512&256.h5")
    import matplotlib.pyplot as plt
    plt.plot(focal_history.history['acc'])
    plt.plot(focal_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('acc.jpg')
    plt.show()
    # summarize history for loss
    plt.plot(focal_history.history['loss'])
    plt.plot(focal_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.jpg')
    plt.show()
    # # import torch

    time_in = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("jieshu",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
