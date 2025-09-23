#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Notebook inspired by https://github.com/PacktPublishing/Deep-Learning-with-TensorFlow-and-Keras-3rd-edition/blob/main/Chapter_2/multiple_linear_regression_using_keras_API.ipynb


# In[36]:


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, Normalization
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data' 
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin']


# In[6]:


data = pd.read_csv(url, names=column_names,
                    na_values='?', comment='\t',
                    sep=' ', skipinitialspace=True)


# In[7]:


data = data.drop('origin', axis=1)
print(data.isna().sum())
data = data.dropna()


# In[8]:


data.head()


# In[9]:


data.info()


# In[10]:


train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)


# In[11]:


sns.pairplot(train_dataset[
	['mpg', 'cylinders', 'displacement','horsepower', 'weight', 'acceleration', 'model_year']
	], diag_kind='kde')


# In[12]:


train_dataset.describe().transpose()[['mean', 'std']]


# In[15]:


train_features = train_dataset.copy()
test_features = test_dataset.copy() 

train_labels = train_features.pop('mpg')
test_labels = test_features.pop('mpg')


# In[17]:


# Normalise features
data_normalizer = Normalization(axis=1)
data_normalizer.adapt(np.array(train_features))


# In[20]:


model = K.Sequential([
                      data_normalizer,
                      Dense(64,  activation='relu'),
                      Dense(32,  activation='relu'),
                      Dense(1,  activation=None)
])
model.summary()


# In[21]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[22]:


history = model.fit(x=train_features,y=train_labels, epochs=100, verbose=1, validation_split=0.2)


# In[30]:


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('epoch')
plt.ylabel('error [mpg]')
plt.legend()
plt.grid(True)


# In[28]:


y_pred = model.predict(test_features).flatten()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, y_pred)
plt.xlabel('True values [mpg]')
plt.ylabel('Predictions [mpg]')
lims = [0, 50]
plt.title('Predicted versus true values of \nthe  multivariate machine learning regression')
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims)
plt.show()


# In[34]:


prediction_error = y_pred - test_labels
plt.hist(prediction_error, bins=30)
plt.title('Distribution of errors')
plt.xlabel('prediction error [mpg]')
plt.ylabel('count')
plt.show()


# In[37]:


# Calculate the prediction error (residuals)
prediction_error = y_pred - test_labels

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(test_labels, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(test_labels, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")


# In[ ]:




