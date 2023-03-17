#!/usr/bin/env python
# coding: utf-8

# # Developing a Convolutional Neural Network From Scratch for MNIST Handwritten Digit Classification.

# -->Importing required libraries to process, load and anaalyse data
# TensorFlow: TensorFlow gives us the flexibility and control with features like the Keras Functional API and Model Subclassing API for creation of complex topologies. For easy prototyping and fast debugging, use eager execution.

# In[2]:


import tensorflow as tf
from tensorflow import keras 


# --> %matplotlib inline: %matplotlib inline enables the drawing of matplotlib figures in the IPython environment. 
# 
# --> Matplotlib supports various types of graphical representations like Bar Graphs, Histograms, Line Graph, Scatter Plot, Stem Plots, etc.
# 
# --> NumPy can be used to perform a wide variety of mathematical operations on arrays. It adds powerful data structures to Python that guarantee efficient calculations with arrays and matrices and it supplies an enormous library of high-level mathematical functions that operate on these arrays and matrices.

# In[3]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 


# # Dataset
# 
# The dataset that is being used here is the MNIST digits classification dataset. Keras is a deep learning API written in Python and MNIST is a dataset provided by this API. This dataset consists of 60,000 training images and 10,000 testing images. It is a decent dataset for individuals who need to have a go at pattern recognition as we will perform in just a minute!
# 
# When the Keras API is called, there are four values returned namely- x_train, y_train, x_test, and y_test. Do not worry, I will walk you through this.
# 
# # LOADING DATASET 
# After loading the necessary libraries, load the MNIST dataset as shown below:
# 
# (X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
# 
# this dataset returns four values and in the same order as mentioned above. Also, x_train, y_train, x_test, and y_test are representations for training and test datasets.

# In[4]:


(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()


# In[5]:


len(x_train)


# In[6]:


len(x_test)


# In[7]:


x_train[0].shape


# In[8]:


len(y_train)


# In[9]:


len(y_test)


# In[10]:


y_train[0].shape


# In[11]:


x_train[0]


# Plotting the trained data set i.e; x_train and checking wether the data presented in y_train and the plotted data of x_train at the same index is same or not.

# In[12]:


plt.matshow(x_train[2])


# In[13]:


y_train[2]


# In[14]:


plt.matshow(x_train[50])


# In[15]:


y_train[50]


# In[16]:


y_train[:10]


# In[17]:


x_train.shape


# If we write X_train[0] then you get the 0th image with values between 0-255 (0 means black and 255 means white). The output is a 2-dimensional matrix (Of course, we will not know what handwritten digit X_train[0] represents. To know this write y_train[0] and you will get 5 as output. This means that the 0th image of this training dataset represents the number 5. 
# 
# So, let’s scale this training and test datasets as shown below:
# 
# X_train = X_train / 255
# X_test = X_test / 255

# In[18]:


x_train = x_train/255
x_test = x_test/255


# In[19]:


x_train[0]


# In[20]:


x_test[0]


# we reshape x_train and x_test to get a copy of an given array collapsed into one dimension.

# In[21]:


x_train_flattend = x_train.reshape(len(x_train),28*28)


# In[22]:


x_train_flattend.shape


# In[23]:


x_test_flattend = x_test.reshape(len(x_test),28*28)


# In[24]:


x_test_flattend.shape


# In[25]:


x_train_flattend[0]


# In[26]:


# create a simple neural network 
# input layer - 784
# output layer - 10 (0-9)
from keras.models import Sequential
model = keras.Sequential()


# Now that the dataset is looking good, it is high time that we create a Convolutional Neural Network.
# 
# # Creating and Training a CNN
# 
# Let’s create a CNN model using the TensorFlow library. The model is created as follows:

# --> The sequential model allows us to specify a neural network, precisely, sequential: from input to output, passing through a series of neural layers, one after the other.
# 
# --> Keras Dense layer is the layer that contains all the neurons that are deeply connected within themselves. 
# 
# --> This means that every neuron in the dense layer takes the input from all the other neurons of the previous layer. 
# 
# --> We can add as many dense layers as required.
# 
# --> model. fit() : fit training data. For supervised learning applications, this accepts two arguments: 
#    
#    the data X and the labels y (e.g. model. fit(X, y) ). 
#    For unsupervised learning applications, this accepts only a single argument, the data X (e.g. model.

# In[27]:


model = keras.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation = 'sigmoid')
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(x_train_flattend,y_train,epochs = 5)


# # Evaluating on test dataset

# ## Making Predictions
# 
# To evaluate the CNN model so created you can run:
# 
# model.evaluate(x_test_flattend,y_test)
# 
# It is time to use our test dataset to see how well the CNN model will perform.
# 
# y_predicted = model.predict(x_test_flattend)
# y_predicted[0]
# 
# The above code will use the convolutional_neural_network model to make predictions for the test dataset and store it in the y_predicted_by_model dataframe. For each of the 10 possible digits, a probability score will be calculated. The class with the highest probability score is the prediction made by the model. For example, if you want to see what is the digit in the first row of the test set:

# In[28]:


model.evaluate(x_test_flattend,y_test)


# In[29]:


plt.matshow(x_test[0])


# In[30]:


y_predicted = model.predict(x_test_flattend)
y_predicted[0]


# NumPy argmax()in Python is used to return the indices of the max elements of the given array along with the specified axis. 
# 
# Using this function gets the indices of maximum elements of single dimensions and multi-dimensional(row-wise or column-wise) of the given array

# In[31]:


np.argmax(y_predicted[0])


# In[32]:


plt.matshow(x_test[1])


# In[33]:


y_predicted[1]


# Created "y_predicted_labels" using np.argmax to see all values present in data_indices. 
# It also helps to create a confusion matrix.

# In[34]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[35]:


y_test[:5]


# Above we can see the predicted labels and the truth labels both are same. 

# Now we create a confusion matrix.
# 
# cm = tf.math.confusion_matrix(labels = y_test,predictions = y_predicted_labels)
# 
# A confusion matrix presents a table layout of the different outcomes of the prediction and results of a classification problem and helps visualize its outcomes. 
# 
# It plots a table of all the predicted and actual values of a classifier.
# 
# It helps us to identify the errors.
# 
# We will create confusion_matrix to compare actual y_test and predicted ones that is y_predicted_labels.

# In[37]:


cm = tf.math.confusion_matrix(labels = y_test,predictions = y_predicted_labels)
cm


# Plotting the heatmap of confusion matrix using seaborn to know how many times our model predicted error and how many times it predicted correctly

# In[45]:


import seaborn as sns 
plt.figure(figsize = (10,8))
sns.heatmap(cm,annot = True,fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # IMPROVING MODEL 

# Adding keras dense hidden layers improve the performance of model.
# specifying how many neurons we want in our hidden layer.which is lessthan input shape. it is a sort of trail and error.
# 
# Using 'Relu' function in activation in Hidden layer. We can add as many layers as we want. 

# In[40]:


model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation = 'relu'),
    keras.layers.Dense(10,activation = 'sigmoid')
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(x_train_flattend,y_train,epochs = 5)


# ### EVALUATING THE IMPROVED MODEL

# In[66]:


model.evaluate(x_test_flattend,y_test)


# ### Above we can see that after adding one hidden dense layer our model accuracy improved from  [0.2659195363521576, 0.9261000156402588] here to [0.08179225772619247, 0.9742000102996826] here.
# 
# ### which means our model improved its accuracy from 92.6% to 97.4% which can be seen as good improvement.

# In[41]:


y_predicted = model.predict(x_test_flattend)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels = y_test,predictions = y_predicted_labels)
plt.figure(figsize = (10,8))
sns.heatmap(cm,annot = True,fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# #### We can directly use keras flatten function to flatten the shape of datasets and passing only x_train and y_train in model.fit()

# In[42]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)),
    keras.layers.Dense(100,activation = 'relu'),
    keras.layers.Dense(10,activation = 'sigmoid')
])
model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
model.fit(x_train, y_train,epochs = 5)


# In[43]:


model.evaluate(x_test,y_test)


# In[44]:


y_predicted = model.predict(x_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels = y_test,predictions = y_predicted_labels)
plt.figure(figsize = (10,8))
sns.heatmap(cm,annot = True,fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # CONCLUSION 

# In this project, we begin by Importing necessary libraries and assigning data to variables & how a dataset is divided into training and test dataset. As an example, a popular dataset called MNIST was taken to make predictions of handwritten digits from 0 to 9. The dataset was cleaned, scaled, and shaped. Using TensorFlow, a CNN model was created and was eventually trained on the training dataset. Finally, predictions were made using the trained model.
# 
# After that we improved the model by adding hidden dense layers and keras flatten functions by which we got a hike of 5% accuracy  inprovement in our model. We can try this out by tweaking the model hyperparameters a bit to see if we are able to achieve higher accuracies or not.
# 
# We got losses between "0.2 to 0.8 " by which we can conclude that it is a preferable model to make predictions.There remains much room for improvement

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




