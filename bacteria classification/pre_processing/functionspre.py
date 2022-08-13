import os
import cv2
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#kernel for morphological closing
def get_kernel(size):
        a=np.zeros((size,size))
        r=int((size-1)/2)
        for i in range(-r,r+1):
           for j in range(-r,r+1):
                   if (i)**2 + (j)**2 <= r**2:
                           a[i+r,j+r]=1
        
        return np.uint8(a)

#morphological closing with kernel shape as (size,size)
def morph_close_k(original_image,size):
        kernel=get_kernel(size)
        closedimage= cv.morphologyEx(original_image, cv.MORPH_CLOSE, kernel)
        return closedimage


#kmeans with bacteria color equal to 'color' and no of clusters equal to 'K'
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

def otsu(image):
        image=np.uint8(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return image_result