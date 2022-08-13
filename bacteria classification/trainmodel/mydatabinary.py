import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import tensorflow as tf
from skimage import io
from tensorflow.keras.utils import Sequence


c=pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'bacinfo.csv'))
#c=pd.read_csv('bacinfo.csv')
label_values=[]
for i in range(34):
    label_values.append([c["R"][i],c["G"][i],c["B"][i]])

print(label_values)




num_classes = len(label_values)
print(num_classes)

def cleanit(image):
    l=[]
    for i in range(4):

        image1=np.uint8(image[i,:,:,:])
        imagegr = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        ind = np.argwhere(imagegr == imagegr.min())
        cl=image1[ind[0,0],ind[0,1]]
        ot, o = cv2.threshold(imagegr, 255, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,)
        o=np.uint8(np.stack((o,o,o),axis=-1))
        b=o==[0,0,0]
        v=np.uint8(b*cl)+o
        l.append(v)
        t=np.stack(tuple(l))

    del l
    return t


def one_hot(mask):
    print(mask.shape)
    semantic_map = []
    for colour in label_values:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = (np.stack(semantic_map, axis=-1)).astype(float)
    return semantic_map

def adjustData(img,mask):
    img = img / 255
    mask = one_hot(mask)
    return (img,mask)


# data generator for semi-supervised learning
def dataGenerator(batch_size,path1,dict1,size,seed=10):
    image_datagen = ImageDataGenerator(**dict1)
    image_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size, size),
        classes = ["images"],
        class_mode = None,
        batch_size = batch_size,

        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size, size),
        classes = ["masks"],
        class_mode = None,
        batch_size = batch_size,

        seed = seed)
    data_generator = zip(image_generator, mask_generator)
    for (image, mask) in data_generator:
        mask=cleanit(mask)
        image, mask = adjustData(image,mask)
        train = (image, mask)
        #print(train[0].shape)
        yield train
        
def dataGeneratorv(batch_size,path1,size,seed=1):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size, size),
        classes = ["images"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    mask_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(size, size),
        classes = ["masks"],
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    data_generator = zip(image_generator, mask_generator)
    for (image, mask) in data_generator:
        image, mask = adjustData(image,mask)
        train = (image, mask)
        #print(train[0].shape)
        yield train


        
def sharpen1(p, T):
    return np.power(p, 1/T) / np.mean(np.power(p, 1/T), axis=-1, keepdims=True)
    
def sharpen(p, T):
    return tf.pow(p, 1/T) / tf.reduce_sum(tf.pow(p, 1/T), axis=1, keepdims=True)


def num_of_images(path):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = ["images"],
        class_mode = None)
    return image_generator.samples

def num_of_images2(path):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path,
        classes = ["ul__val_images"],
        class_mode = None)
    return image_generator.samples

def validation_g(image_path,mask_path,image_prefix = ".tif",mask_prefix = "_gt.tif"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))

    for index,item in enumerate(image_name_arr):
        img = cv2.cvtColor(io.imread(item),cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        mask = mask[:,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        val = (img,mask)
        yield val

#mask name should end with '_gt' following the image name
def validation(image_path,mask_path,image_prefix = ".tif",mask_prefix = "_gt.tif"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.cvtColor(io.imread(item),cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        mask = mask[:,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr, image_name_arr

def validation_arr(image_path,mask_path,image_prefix = ".tif",mask_prefix = "_gt.tif"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))

    for index,item in enumerate(image_name_arr):
        img = cv2.cvtColor(io.imread(item),cv2.COLOR_BGR2RGB)
        mask= cv2.cvtColor(io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        mask = mask[:,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr=np.stack((image_arr,img),axis=-4)
        mask_arr=np.stack((mask_arr,mask),axis=-4)
    return image_arr,mask_arr, image_name_arr


def validation_tif(image_path,mask_path,image_prefix = ".tif",mask_prefix = "_gt.tif"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))

    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)

        mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:1532,:2048,:3]
        mask = mask[:1532,:2048,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr, image_name_arr

def validation_png(image_path,mask_path,image_prefix = ".png",mask_prefix = "_gt.png"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))

    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)

        mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:1184,:,:3]
        mask = mask[:1184,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        img,mask = adjustData(img,mask)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr, image_name_arr

def validation2(image_path):
    image_name_arr = glob.glob(os.path.join(image_path,"*.png"))
    image_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        img = img[:,:,:3]
        img = img/255
        image_arr.append(img)
    image_arr = np.array(image_arr)
    return image_arr, image_name_arr
    

def validation2_tif(image_path):
    image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    image_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)
        img = img[:1184,:,:3]
        img = img/255
        image_arr.append(img)
    image_arr = np.array(image_arr)
    return image_arr, image_name_arr

