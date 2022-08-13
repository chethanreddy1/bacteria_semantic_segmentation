import os
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
# from sklearn.cluster import MeanShift

def morph_dil(original_image,kernel_size):       
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
        dilation = cv.dilate(original_image,kernel,iterations = 1)
        return dilation

def morph_ero(original_image,kernel_size):    
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
        erosion = cv.erode(original_image,kernel,iterations = 1)
        return erosion

def imgsv(imgpath,img):
        cv.imwrite(imgpath,img)

def imgr(imgpath):    
        return cv.imread(imgpath)

def image_show(img):    
        cv.imshow('image',img)
        cv.waitKey(0)
        cv.destroyAllWindows()

def showimg(img,img2):
        figure_size = 15
        plt.figure(figsize=(figure_size,figure_size))
        plt.subplot(1,2,1),plt.imshow(img)
        plt.title('img1'), plt.xticks([]), plt.yticks([])
        plt.subplot(1,2,2),plt.imshow(img2)
        plt.title('img2'), plt.xticks([]), plt.yticks([])
        plt.show()
    
   
def morph_open(original_image,kernel_size):
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
        openedimage = cv.morphologyEx(original_image, cv.MORPH_OPEN, kernel)
        return openedimage

def morph_close(original_image,kernel_size):
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,kernel_size)
        openedimage= cv.morphologyEx(original_image, cv.MORPH_CLOSE, kernel)
        return openedimage

def morph_close_k(original_image,size):
        kernel=get_kernel(size)
        closedimage= cv.morphologyEx(original_image, cv.MORPH_CLOSE, kernel)
        return closedimage


# def meanshiftmask(original_image,bacteria_color):
#         img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
#         vectorized = img.reshape((-1,3))
#         ms=MeanShift()
#         ms.fit(vectorized)
#         label=ms.labels_
#         v=ms.cluster_centers_
#         if np.sqrt(v[0].dot(v[0]))>np.sqrt(v[1].dot(v[1])):
#            new_color=np.array([[255,255,255],bacteria_color])
#         else:
#            new_color = np.array([bacteria_color,[255,255,255]])
#         res = new_color[label]
#         result_image = res.reshape((img.shape))
#         return result_image

def kmeans_mask(original_image,color):
        vectorized = original_image.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        v=center
        if np.sqrt(v[0].dot(v[0]))>np.sqrt(v[1].dot(v[1])):
           new_color=np.array([[255,255,255],color])
        else:
           new_color = np.array([color,[255,255,255]])
        res = new_color[label]
        result_image = res.reshape((original_image.shape))
        return result_image

def kmeans_mask_k(original_image,color,K):
        l=[]
        vectorized = original_image.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts=10
        ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        v=center
        new_color=np.array([[255,255,255]]*K)
        for i in range(len(v)):
                l.append(np.linalg.norm(v[i]))
        v=np.array(l)
        new_color[np.argmin(v)]=np.array(color)
        res = new_color[label]
        result_image = res.reshape((original_image.shape))
        return result_image

# def gaus_mask(original_image,color):
#         img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
#         vectorized = img.reshape((-1,3))
#         gmm = GaussianMixture(n_components=2)
#         gmm.fit(vectorized)
#         label = gmm.predict(vectorized)
#         v=gmm.means_
#         if np.sqrt(v[0].dot(v[0]))>np.sqrt(v[1].dot(v[1])):
#            new_color=np.array([[255,255,255],color])
#         else:
#            new_color = np.array([color,[255,255,255]])
#         res = new_color[label]
#         result_image = res.reshape((img.shape))
#         return result_image

def otsu(image):
        image=np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return image_result



get_kernel_bin = lambda size : cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))


def get_kernel(size):
        a=np.zeros((size,size))
        r=int((size-1)/2)
        for i in range(-r,r+1):
           for j in range(-r,r+1):
                   if (i)**2 + (j)**2 <= r**2:
                           a[i+r,j+r]=1
        
        return np.uint8(a)
