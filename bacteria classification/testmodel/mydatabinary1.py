from ast import Index
import os,cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import tensorflow as tf
from skimage import io
from PIL import Image
import pandas as pd


c=pd.read_csv(os.path.join(os.path.dirname(os.getcwd()),'bacinfo.csv'))
# c=pd.read_csv('bacinfo.csv')
label_values=[]
for i in range(34):
    label_values.append([c["R"][i],c["G"][i],c["B"][i]])

print(label_values)


check_path = os.path.dirname(os.getcwd()) + "/Results/check"
write_path = os.path.dirname(os.getcwd())



num_classes = len(label_values)
print(num_classes)

def one_hot(mask):
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
    
def adjustData_new(img):
    img = img / 255
    return (img)
    
#https://github.com/jkjung-avt/keras-cats-dogs-tutorial/blob/master/train_cropped.py
def random_crop(img,random_crop_size):
    # Note: image_data_format is 'channel_last'
    #assert img.shape[2] == 3
    height, width = 512,512
    dy_img = random_crop_size
    #np.random.seed(1)
    width_new = 512-dy_img
    height_new = 512-dy_img
    x = np.random.randint(low=0, high=width_new+1,size = None)
    y = np.random.randint(low=0, high=height_new+1,size = None)
    img_new = img[y:int(y+dy_img), x:int(x+dy_img), :]
    return img_new

def crop_generator_img(batches,batch_size,seed_new,i1,scale, path_count):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    size = 512
    while True:
        #batch_x, batch_y = next(batches)
        batch_x = next(batches)
        
        for i in range(batch_size):
            rand_idx = np.random.randint(0, len(scale)-1)
            random_num = scale[rand_idx]
    
            #calculate the crop length
            crop_length = int(np.ceil(random_num*size))
            batch_crops_img = np.zeros((batch_size,crop_length,crop_length,3)) 
            batch_crops_new_img= np.zeros((batch_size,size,size,3))
            batch_crops_img[i] = random_crop(batch_x[i], crop_length)
            batch_crops_new_img[i] = cv2.resize(batch_crops_img[i],(size,size),interpolation=cv2.INTER_NEAREST)
            # if (path_count==0):
            #     i1=i
            # else:
            #     i1 = i+batch_size
            # write_batch = write_path + "/Data/DiBaS_dataset_gt/both_gp_gn/random_cropped_image/"+"img_"+str(i1)+".jpg"
            # cv2.imwrite(write_batch,batch_crops_new_img[i])
        yield (batch_crops_new_img)


# data generator for semi-supervised learning
def dataGenerator(batch_size,path1,aug_dict,size,seed=1):
    #CROP_LENGTH = np.ceil(0.8*size)
    image_datagen = ImageDataGenerator(**aug_dict)
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
    seed_new = np.random.randint(0,100,size=None)
    
    scale = np.random.uniform(0.8,1,size = 100)
    i1=0
    path_count = 0
    image_generator = crop_generator_img(image_generator,batch_size,seed_new,i1 ,scale,path_count=0)
    mask_generator = crop_generator_img(mask_generator,batch_size,seed_new,i1,scale,path_count=1)
    data_generator = zip(image_generator, mask_generator)
    for (image, mask) in data_generator:
        image, mask = adjustData(image,mask)
        train = (image, mask)
        yield train
        
def dataGeneratorv(path1):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(2048, 1504),
        classes = ["images"],
        class_mode = None,
        batch_size = 1,
        seed = 1)
    mask_generator = image_datagen.flow_from_directory(
        path1,
        target_size=(2048, 1504),
        classes = ["masks"],
        class_mode = None,
        batch_size = 1,
        seed = 1)
    data_generator = zip(image_generator, mask_generator)
    for (image, mask) in data_generator:
        image, mask = adjustData(image,mask)

        train = (image,)
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

#mask name should end with '_gt' following the image name
def validation(image_path,mask_path,image_prefix = ".jpg",mask_prefix = "_gt.jpg"):
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


def validation_tif(image_path,mask_path,image_prefix = ".tif",mask_prefix = "_gt.tif"):
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
    
    
def validation_tif_dibas(image_path,image_prefix = ".tif"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))
    image_arr = []

    for index,item in enumerate(image_name_arr):
      if index <5:
        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)

        img = img[:1504,:,:3]
        image_arr.append(img/255)

    image_arr = np.array(image_arr)

    return image_arr,image_name_arr
def testdatagen(path):
        image_datagen = ImageDataGenerator()
        image_generator = image_datagen.flow_from_directory(
        path,
        target_size=(1504, 2048),
        classes = ["images"],
        batch_size=1,
        class_mode = None,
        seed = 1)
        # for i in enumerate(image_generator):
        #     print(i[1])
        #     yield i
        return image_generator

#This function will predict semantically segmented images without taking the gt images into accout  
def validation_tif_new(image_path,image_prefix = ".tif"):
    image_name_arr = glob.glob(os.path.join(image_path,"*%s"%image_prefix))

    image_arr = []
    
    for index,item in enumerate(image_name_arr):

        img = cv2.cvtColor(cv2.imread(item),cv2.COLOR_BGR2RGB)

        # mask= cv2.cvtColor(cv2.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix)),cv2.COLOR_BGR2RGB)
        img = img[:1184,:,:3]
        # mask = mask[:1184,:,:3] if mask.ndim==3 else np.repeat(mask[:,:,np.newaxis],3,axis=-1)
        imgss = adjustData_new(img)
        image_arr.append(img)
        # mask_arr.append(mask)
    image_arr = np.array(image_arr)
    # mask_arr = np.array(mask_arr)
    return image_arr, image_name_arr

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
        #img = img[:,:,:3]
        img = img[:1184,:,:3]
        img = img/255
        image_arr.append(img)
    image_arr = np.array(image_arr)
    return image_arr, image_name_arr

