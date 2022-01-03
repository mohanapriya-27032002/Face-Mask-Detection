#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


img = cv2.imread("C:\\Users\\user\\Pictures\\images.jpg")


# In[3]:


img.shape


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


haar_data = cv2.CascadeClassifier("C:\\Users\\user\\Downloads\\haarcascade_frontalface_default.xml")


# In[7]:


haar_data.detectMultiScale(img)


# In[11]:


capture = cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
            face = img[y:y + h, x:x + w, :]
            face = cv2.resize(face, (50, 50))
            print(len(data))
            if len(data) < 400:
                data.append(face)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break
capture.release()
cv2.destroyAllWindows()


# In[9]:


import numpy as np


# In[10]:


np.save("without_mask.npy", data)


# In[12]:


np.save("with_mask.npy", data)


# In[ ]:




