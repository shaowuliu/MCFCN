#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT-CAPS binary test code implementation.

!! Note: CT-CAPS framework is in the research stage. Use only for research ourposes at this time.
Don't use CT-CAPS as a replacement of the clinical test and radiologist review.

Created by: Shahin Heidarian, Msc. at Concordia University
E-mail: s_idari@encs.concordia.ca

** The code for the Capsule Network implementation is adopted from https://keras.io/examples/cifar10_cnn_capsule/.
"""

#%% Libraries
from __future__ import print_function
import time

import torch
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
from keras import optimizers
from keras.utils import to_categorical
import tensorflow as tf
# from sklearn.utils import class_weight
# from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt
import pydicom
import cv2
from lungmask import mask #lung segmentation model
import SimpleITK as sitk
import SimpleITK as sitk
# import pandas as pd
from PIL import Image

# Set the path based on your data directory
data_path = r'autodl-nas/covid/P019'
# Set the cut-off probability for the classification output (Default : 0.5)
cutoff = 0.5

K.set_image_data_format('channels_last') # 彩色图像的性质一般包括：width、height、channels 选择channels_last：返回(256,256,3)

def squash(x, axis=-1):
    print("squash")
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (1 + s_squared_norm)
    return scale * x


def softmax(x, axis=-1):
    print("softmax")
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


def margin_loss(y_true, y_pred):
    print("loss")
    lamb, margin = 0.5, 0.1
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


class Capsule(Layer):


    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        'num_capsule':  self.num_capsule,
        'dim_capsule' : self.dim_capsule,
        'routings':  self.routings,
        'share_weight':self.share_weights,



        })
        return config

    def build(self, input_shape):
        print("build")
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        print("call")

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)  # 跑这个
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        print('hat1',hat_inputs.shape)
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))
        print('hat2',hat_inputs.shape)
        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            print('c',c.shape)
            o = self.activation(keras.backend.batch_dot(c, hat_inputs, [2, 2]))
            print('o',o.shape)
            if i < self.routings - 1:
                b = keras.backend.batch_dot(o, hat_inputs, [2, 3])
                print('b',b.shape)
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)
                    print('o2',o.shape)

        return o

    def compute_output_shape(self, input_shape):
        print("compute_output_shape")
        return (None, self.num_capsule, self.dim_capsule)

# normalization function
def normalize_image(x): #normalize image pixels between 0 and 1
        if np.max(x)-np.min(x) == 0 and np.max(x) == 0:
            return x
        elif np.max(x)-np.min(x) == 0 and np.max(x) != 0:
            return x/np.max(x)
        else:
            return (x-np.min(x))/(np.max(x)-np.min(x))


def segment_lung(mask,model,volume_path): #mask 是 模型的得到的方法 model是unet+R321covid

    # model = mask.get_model('unet','R231CovidWeb')
    #loop through all dcm files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(volume_path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))

    dataset = pydicom.dcmread(lstFilesDCM[0]) # a sample image
    slice_numbers = len(lstFilesDCM) #number of slices
    # print('Slices:',slice_numbers)
    #print("dataset",dataset)
    #输出dataset后，发现里面居然除了图片的信息之外还有 比如的信息
    # (0009, 0010) Private Creator LO: 'SIEMENS CT VA1 DUMMY'
    # (0010, 0020) Patient ID      LO: 'P169'
    # (0010, 0040) Patient's Sex   CS: 'M'
    # (0010, 1010) Patient's Age   AS: '075Y'
    #(7fe0, 0010) Pixel Data   OB: Array of 262374 elements
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        # print('Image size:',rows,cols)

    slice_z_locations = []
    for filenameDCM in lstFilesDCM:
        ds = pydicom.dcmread(filenameDCM)
        slice_z_locations.append(ds.get('SliceLocation'))

    #sorting slices based on z locations
    slice_locations = list(zip(lstFilesDCM,slice_z_locations))
    sorted_slice_locations = sorted(slice_locations, key = lambda x: x[1])[-1::-1]

    # Saving Slices in a numpy array
    ArrayDicom = np.zeros((slice_numbers,rows,cols)) # 这里面存的才是基础的ct的图片
    lung_mask = np.uint8(np.zeros((slice_numbers,rows,cols))) # unit8创建图像容器
    # 这里面存的是 这个位置是否存在肺部组织 是 --1  否  --0
    # loop through all the DICOM files
    i = 0
    for filenameDCM, z_location in sorted_slice_locations:
        # read the file
        ds = sitk.ReadImage(filenameDCM)
        segmentation = mask.apply(ds, model) # segmentation 0 or 1 (1, 512, 512)
        # 这个模型只是把肺部的位置提取出来了，提取出只包含 -1 0 1的数组
        lung_mask[i,:,:] = np.uint8(((segmentation>0)*1)[0]) #uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
        # lung_mask就是把 -1 去掉，然后以图片的形式保存这个数组
        ArrayDicom[i, :, :] = sitk.GetArrayFromImage(ds)
        # ArrayDicom 就是 原来这个ct没见过处理的样子
        # 可用于将SimpleITK对象转换为ndarray  就是把图像变成数组
        # 使用GetArrayFromImage()方法后，X轴与Z轴发生了对调，输出形状为：(Depth, Height, Width)
        i = i+1
    # print("输出第一张切片的分割结果")
   # showpicture(lung_mask[0])
    lungs = np.zeros((ArrayDicom.shape[0],256,256,1))
    # resizing the data
    for i in range(ArrayDicom.shape[0]):
        ct = normalize_image(ArrayDicom[i,:,:])
        mask_l = lung_mask[i,:,:]
        seg = mask_l * ct #apply mask on the image  seg.shape (512, 512)
        # mask_l 是分割后的肺部阴影图  ct是 每个肺部ct
        img = cv2.resize(seg,(256,256)) # 512X512 到 256X256
        img = normalize_image(img) #img.shape (256, 256)
        lungs[i,:,:,:] = np.expand_dims(img,axis = -1)
    # print('Successfully segmented.')
    # print(lung_mask.shape,ArrayDicom.shape,lungs.shape)
    # (121, 512, 512) (121, 512, 512) (121, 256, 256, 1)  这里就是121个dcm文件的512*512矩阵
    # 输出代码：a[1] 第二行 a[:,1] 第二列  默认二维矩阵
    # print(lung_mask[0]) 全是0和1
    # print(lungs[onepic,124, 124, 0]) # 0.0
    # expand_dims:如果设置axis为1，矩阵大小就变成了（2,1,3），变成了一个2*1*3的三维矩阵 -1 就是最后一个维度 aixs从0开始
    return lung_mask, ArrayDicom, lungs

def max_vote(x):
    v = np.max(x, axis=0) #aixs = 0 代表 列
    # 多个维度相比 取出切片数目个维度的每行每列最大值（我可以认为是特征最明显吗？） 这是取出最大 理解为消去 axis=0 就是从（a，b，c） 到 （b，c）
    return v

def tet_one_dicom(model,X_test):
    X_test_normal = np.zeros(X_test.shape)
    for i in range(X_test.shape[0]):
        X_test_normal[i,:,:,:] = normalize_image(X_test[i,:,:,:])

    # X-test 的model(121, 256, 256, 1) 121和168是不同人的切片数目
    sum_seg = np.sum(np.sum(X_test,axis=1),axis=1) #(168, 1) 算是综合出来判断这个部位有没有信息 to find out if lung exists or not
    # 当加入axis=1以后就是将一个矩阵的每一行向量相加
    a = np.where(sum_seg[:,0] != 0)
    # 这个就是显示之前判断的有肺部部位的数据的地址（168，1） 【0，0，0，4，5，6.。。。】这种的内容  找出了不能用的切片吗？
    # where现在没有x和y了,最终结果返回的就是判断结果为true的元素所在的位置信息.
    X_test = X_test_normal[a]

    capsules = np.zeros((1,32,16))
    # print("得出的特征图的灰度图")
    #wpicture(capsules[0])
    if len(X_test_normal)==0:
        capsules[0] = np.zeros((32,16))-1
    else:
        x_capsule = model.predict(X_test)
        #  输入第一个模型的X_test.shape (152, 256, 256, 1) 这个152原来是切片数量168，但是前面判断存在16个空的部位，所有只剩下152个
        # 所有的切片都送到了
        # print("x_capsule.shape",x_capsule.shape) # 模型预测的结果x_capsule.shape (116, 32, 16)
        capsules[0] = max_vote(x_capsule) # [32,16] 我觉得又问题
    # print("capsules.shape",capsules.shape) #[1,32,16]
    return capsules


# def showpicture(a):
#     image = Image.open(a)
#     mat = np.array(image)
#     plt.imshow(mat)
#     plt.show()

print("kaishi",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#%% Model1 (Feature Extractor)

input_image = Input(shape=(None, None, 1))
#  输入的大小是(116, 256, 256, 1)
x = Conv2D(64, (3, 3), activation='relu',trainable = True)(input_image)
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
# output = Capsule(32, 16, 3, True)(x)
x = Capsule(32, 16, 3, True)(x)
# output 是 (116, 32, 16)
capsule = Capsule(2, 16, 3, True)(x)
# 输出是(116, 2, 16)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule) # (116,2)
#  上面这不算是一层网络
model = Model(inputs=[input_image], outputs=[output])
# adam = optimizers.Adam(lr=1e-4)
# model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# # model.save_weights("model.h5")
model.load_weights('weights-2class-v1-71.h5')
model.summary()
model2 = Model(inputs=[input_image], outputs=model.layers[-3].output)

import numpy
# 预测图片类别并获取特征图
# predict = np.array(pred)
img = 'data/covid/covid6.npy' # 28 6 39
# img  = cz
x = numpy.load(img)
# cz2 = numpy.expand_dims(cz, -1)
x2 = numpy.zeros((1,256,256,1))
#76  78  87  57  31
a = [40,45,50]
b = [1,1,1,]
for cc in range(0,len(a)):
    qiepian = a[cc]
    x2[0] = x[qiepian-1]
    output = model.predict(x2)
    print('output',output)
    predict = np.array(output[0])

    # 获取预测向量中的推测元素、输出特征图
    heisemei_output = model.output[:,np.argmax(predict)]
    print('heisemei_output ',heisemei_output,heisemei_output .shape)
    last_conv_layer = model.get_layer('conv2d_4')  # 最后一个卷积,可以换成别的层名，用来获取每一层的类激活图、
    # 计算相对梯度，并对前三个维度取平均值
    grads = K.gradients(heisemei_output,last_conv_layer.output)[0]  # 计算特征图与预测元素之间的梯度 可能报错
    # 解决方案1 替换成with tf.GradientTape() as gtape:
    #     grads = gtape.gradient(african_elephant_output, last_conv_layer.output)
    # 解决方案2：使用tf1的compat模式禁用eager-execution约束表单tf2，添加如下代码：
    # import tensorflow as tf
    # tf.compat.v1.disable_eager_execution()
    pooled_grads = K.mean(grads, axis=(0,1,2))
    print('pool',pooled_grads)
    # 对于给定的样本图像， pooled_grads和最终的输出特征图
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

    pooled_grads_value, conv_layer_output_value = iterate([x2])
    print('pool shape',pooled_grads_value.shape) #一个矩阵 (128,1)
    print(conv_layer_output_value.shape) #(122, 122, 128) 这个矩阵的输出的东西
    for i in range(128):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    # 得到特征图的逐通道平均值，即为类激活的热力图
    heatmap = np.mean(conv_layer_output_value, axis=2)
    print('tongdaoquanzhong',heatmap.shape)
    # 特征图可视化
    # 将热力图标准化到0-1之间, 并可视化Invalid argument: transpose expects a vector of size 3. But input(1) is a vector of size 4 	 [[{{node conv2d_5/convolution-0-TransposeNHWCToNCHW-LayoutOptimizer}}]] 	
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # cmap = plt.get_cmap('Spectral')
    print('权值图')
    # plt.imshow(numpy.squeeze(heatmap))
    # plt.show()
    # 热力图与原图进行叠加
    # 重新读取原图像
    img = x2
    # 将热力图的大小调整与原图一致
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    print('all == 0?',np.all(heatmap==0))
    # 将热力图转换为RGB格式
    heatmap2 = np.uint8(255 * heatmap)
    print('heat',heatmap.shape,x2.shape)
    # 将热利用应用于原始图像
    import cv2
    heatmap = cv2.applyColorMap(heatmap2, 4)
    # 　这里的热力图因子是０.４
    print('最初始的结果')
    print(x2.shape)
    new_yuan = np.uint8(225*np.squeeze(x2))
    # new_yuan = new_yuan/225
    yuan = cv2.applyColorMap(new_yuan, 1)

    plt.imshow(np.squeeze(x2))
    plt.show()
    # 存储热力图
    # 保存图像
    print('热力图')


    #for c in range(0,11):
    c = 2
    yuan2 = cv2.applyColorMap(new_yuan, c)
    plt.imshow(yuan2)
    plt.show()

    heatmap3 = cv2.applyColorMap(heatmap2,c)
    plt.imshow(heatmap3)
    plt.show()

    # superi = (heatmap3*0.2+x2) # 以前是两个直接相加，就没别得了
    # if b[cc] == 0:
    #     plt.title('no covid-19 lesions slice'+'+number'+str(qiepian))
    #     plt.imshow(superi)
    #     plt.show()
    # else:
    #     plt.title('covid-19 lesions slice'+'+number'+str(qiepian))
    #     plt.imshow(superi)
    #     plt.show()