#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[ ]:





# In[2]:


with_mask = np.load('with_mask.npy')
without_mask = np.load('without_mask.npy')


# In[3]:


with_mask.shape


# In[4]:


without_mask.shape


# In[5]:


with_mask = with_mask.reshape(200,50*50*3)
without_mask = without_mask.reshape(200,50*50*3)


# In[6]:


with_mask.shape


# In[7]:


without_mask.shape


# In[8]:


x = np.r_[with_mask, without_mask]


# In[9]:


x.shape


# In[10]:


labels = np.zeros(x.shape[0])


# In[11]:


labels[200:] = 1.0


# In[12]:


names = {0 : 'Mask', 1: 'No Mask'}


# In[13]:


#svm - support Vector Machine
from sklearn.svm import SVC


# In[14]:


from sklearn.metrics import accuracy_score


# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train, x_test, y_train, y_test = train_test_split(x,labels, test_size=0.25)


# In[17]:


x_train.shape


# In[18]:



from sklearn.decomposition import PCA


# In[19]:


pca = PCA(n_components=3)
x_train= pca.fit_transform(x_train)


# In[20]:


x_train[0]


# In[21]:


x_train.shape


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(x,labels, test_size=0.25)


# In[23]:


svm = SVC()
svm.fit(x_train, y_train)


# In[24]:


#x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[25]:


accuracy_score(y_test, y_pred)


# In[26]:


haar_data = cv2.CascadeClassifier("C:\\Users\\user\\Downloads\\haarcascade_frontalface_default.xml")
capture = cv2.VideoCapture(0)
data = []
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    flag,img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w, y+h), (255,0,255), 4)
            face = img[y:y+h,x:x+w, :]
            face = cv2.resize(face,(50,50))
            face = face.reshape(1,-1)
            #face = pca.transform(face)
            pred = svm.predict(face)
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1, (244,250,250), 2)
            print(n)
        cv2.imshow('result', img)
        if cv2.waitKey(2) == 27:
            break
capture.release()
cv2.destroyAllWindows()
            


# In[ ]:





# In[ ]:




