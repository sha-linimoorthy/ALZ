#!/usr/bin/env python
# coding: utf-8

# # Setting up the environment and uploading the data

# In[1]:


from sklearn.preprocessing import StandardScaler #mean 0, sd 1
from glob import glob #search and match the file 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg #working with img read and show img
from skimage.transform import resize
import pandas as pd
from matplotlib.image import imread #read img frm file to array
from skimage.io import imread_collection #load a collection of image
from PIL import Image #img transformation rotate
import seaborn as sns
from sklearn import decomposition, preprocessing, svm
import sklearn.metrics as metrics #confusion_matrix, accuracy_score
from time import sleep 
from tqdm.notebook import tqdm #progress bar
import os
sns.set()


# # Uploading the data set

# In[2]:


#Dataset that should go with Alzheimer label
very_mild = glob(r"E:\Alzheimersdiesease\archive\Dataset\Very_Mild_Demented\*")
mild = glob(r"E:\Alzheimersdiesease\archive\Dataset\Mild_Demented\*")
moderate = glob(r"E:\Alzheimersdiesease\archive\Dataset\Moderate_Demented\*")

#Dataset without Alzheimer
non = glob(r"E:\Alzheimersdiesease\archive\Dataset\Non_Demented\*")


# # One of the non-Alzheimer's data
Matplotlib uses a colormap to map the pixel values to colors, when plotting an img using 'imshow' func, takes pixel values and colormaps,,,, 'virdis'
# In[7]:


print(non[1])
def view_image(directory):
    img = mpimg.imread(directory)
    plt.imshow(img) #img
    plt.title(directory)
    plt.axis('off')
    print(f'Image shape:{img.shape}')
    return img #array

print('One of the data in Non Alzheimer Folder')
view_image(non[1])


# # One of the Azheimer's data

# In[4]:


print('Alzheimer Patient\'s Brain')
view_image(moderate[1])

PCA dimentionality reduction method by most variance, img to vector mean vector sub,covarianceeigen vectors, eigen valuesTranform features to normal scale, improves performance of ML alg
# # PCA for Alzheimer's disease

# In[10]:


def extract_feature(dir_path):
    img = mpimg.imread(dir_path)
    img = img / 255.0  # normalize pixel values
    img = resize(img, (128, 128, 3))  # convert all images to (128x128x3)
    img = np.reshape(img, (128, 384)) #img to vector
    return img

non_ALZ = [extract_feature(filename) for filename in non]
vmild_ALZ = [extract_feature(filename) for filename in very_mild]
mild_ALZ = [extract_feature(filename) for filename in mild]
moderate_ALZ = [extract_feature(filename) for filename in moderate]
ALZ = vmild_ALZ + mild_ALZ + moderate_ALZ

#for PCA
all_data = np.concatenate((np.array(non_ALZ),np.array(ALZ))) #add as rows
#print(all_data)
all_data = all_data.reshape(all_data.shape[0], np.product(all_data.shape[1:]))  #(m*n) 0-row, 1-col

scaler = StandardScaler() #feature-mean/sd
scaler.fit(all_data) #mean,sd computed

#standardize data to 0 mean and unit variance
X = scaler.transform(all_data) #x-mean/sd

#split the data 
from sklearn.model_selection import train_test_split
y = [0] * len(non_ALZ) + [1] * len(ALZ)    #creates a list 
X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=0.2)

scala = preprocessing.StandardScaler()
#Compressing the images into two dimensions using PCA
pca = decomposition.PCA(200)
X_proj = pca.fit_transform(X_train)

#let's first see which principal component works better
#scree plot but cumulative
# Getting the cumulative variance 1 2 3 => 1 3 6
var_cumu = np.cumsum(pca.explained_variance_ratio_)*100 #100 is multiplied for percentage
 
# How many PCs explain 90% of the variance? 0 so
k = np.argmax(var_cumu>80)
print("Number of components explaining 80% variance: "+ str(k)) #174
#print("\n")
 
plt.figure(figsize=[10,5])
plt.title('Cumulative Explained Variance explained by the components')
plt.ylabel('Cumulative Explained variance')
plt.xlabel('Principal components')
plt.axvline(x=k, color="k", linestyle="--")
plt.axhline(y=80, color="r", linestyle="--")
ax = plt.plot(var_cumu)

print(X_proj)


# In[11]:


#List where arrays shall be stored
resized_image_array=[]
#List that will store the answer if an image is female (0) or male (1)
resized_image_array_label=[]

width = 256
height = 256
new_size = (width,height) #the data is just black to white 

#Iterate over pictures and resize them to 256 by 256
def resizer(image_directory):
    for file in image_directory: #tried with os.listdir but could work with os.walk as well
        img = Image.open(file) #just putting image_directory or file does not work for google colab, interesting. 
        #preserve aspect ratio
        img = img.resize(new_size)
        array_temp = np.array(img)
        shape_new = width*height
        img_wide = array_temp.reshape(1, shape_new)
        resized_image_array.append(img_wide[0])
        if image_directory == non:
            resized_image_array_label.append(0)
        else:
            resized_image_array_label.append(1)

ALZ = very_mild + mild + moderate
resizer(non)
resizer(ALZ)


# In[12]:


print(len(non))
print(len(ALZ)) #data are well transformed. Let's conduct SVM
print(len(resized_image_array))
print(resized_image_array[1])

#split the data to test and training
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(resized_image_array, resized_image_array_label, test_size = 0.2)

SVM Classification and regression, here classification, inp feature to high dimentional feature =>kernel
# In[13]:


#train SVM model
#from sklearn import svm
clf = svm.SVC(kernel = 'linear')
clf.fit(train_x, train_y)
#store predictions and ground truth
y_pred = clf.predict(train_x)
y_true = train_y

#assess the performance of the SVM with linear kernel on Training data
print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
print('Precision : ', metrics.precision_score(y_true, y_pred))
print('Recall : ', metrics.recall_score(y_true, y_pred))
print('f1 : ', metrics.f1_score(y_true, y_pred)) 
print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred)) #The training seems to be done with high accuracy on training data.

#Now, use the SVM model to predict Test data
y_pred = clf.predict(test_x)
y_true = test_y

#assess the performance of the SVM with linear kernel on Testing data
print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
print('Precision : ', metrics.precision_score(y_true, y_pred))
print('Recall : ', metrics.recall_score(y_true, y_pred))
print('f1 : ', metrics.f1_score(y_true, y_pred)) 
print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred)) #Having high training data accuracy might mean that it is having some overfitting

Receiver Operating Curve False positive rate and true positive rate 
AUC Area Under the ROC Curve fpr and tpr
# In[15]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

fpr,tpr,thresholds=roc_curve(y_true,y_pred)
roc_auc=auc(fpr,tpr)

plt.plot(fpr,tpr,color="darkorange",lw=2,label='ROC curve')
plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.show()

So perfect classification performance in terms of TPR but still misclassify if AUC 1- perfect classifier
# In[16]:


from sklearn.metrics import roc_auc_score

auc=roc_auc_score(y_true,y_pred)
print("AUC Score : ",auc)


# In[9]:


#Train a SVM using polynomial kernel with degree of 2
clf = svm.SVC(kernel = 'poly', degree = 2)
clf.fit(train_x, train_y)

#store predictions and ground truth
y_pred = clf.predict(train_x)
y_true = train_y

#assess the performance of the SVM with linear kernel on Training data
print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
print('Precision : ', metrics.precision_score(y_true, y_pred))
print('Recall : ', metrics.recall_score(y_true, y_pred))
print('f1 : ', metrics.f1_score(y_true, y_pred)) 
print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred))

#Now, use the SVM model to predict Test data
y_pred = clf.predict(test_x)
y_true = test_y

#assess the performance of the SVM with linear kernel on Testing data
print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
print('Precision : ', metrics.precision_score(y_true, y_pred))
print('Recall : ', metrics.recall_score(y_true, y_pred))
print('f1 : ', metrics.f1_score(y_true, y_pred)) 
print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred))

RBF Radial Basis Function inpt to infinite dimentional space by guassian func,
# In[11]:


#Train a SVM using RBF kernel
clf = svm.SVC(kernel = 'rbf')
clf.fit(train_x, train_y)

#store predictions and ground truth
y_pred = clf.predict(train_x)
y_true = train_y

#assess the performance of the SVM with linear kernel on Training data
print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
print('Precision : ', metrics.precision_score(y_true, y_pred))
print('Recall : ', metrics.recall_score(y_true, y_pred))
print('f1 : ', metrics.f1_score(y_true, y_pred)) 
print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred))

#Now, use the SVM model to predict Test data
y_pred = clf.predict(test_x)
y_true = test_y

#assess the performance of the SVM with linear kernel on Testing data
print('Accuracy : ', metrics.accuracy_score(y_true, y_pred))
print('Precision : ', metrics.precision_score(y_true, y_pred))
print('Recall : ', metrics.recall_score(y_true, y_pred))
print('f1 : ', metrics.f1_score(y_true, y_pred)) 
print('Confusion matrix :', metrics.confusion_matrix(y_true, y_pred))


# In[ ]:




