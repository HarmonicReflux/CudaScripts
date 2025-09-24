#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[4]:


model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)  # this neural network has only one neuron in it
])


# In[5]:


model.compile(optimizer='sgd', loss='mean_squared_error')
model.summary()


# In[6]:


xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
plt.plot(xs, ys)
plt.show()
np.corrcoef(xs, ys)


# In[11]:


class myCallback(tf.keras.callbacks.Callback):
    '''
    Hals training when the loss balls below a certain defined number.

    Atgs:
        epoch (integer) - index of epoch. Required but unused in teh function definition below.
        logs (dict) - metric results from the training epoch.
    '''
    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < 0.1:
            print('loss is lower than required threshold. Cancelling training in consequence.')
            self.model.stop_training = True


# In[12]:


model.fit(xs, ys, epochs=100, callbacks=[myCallback()])


# In[13]:


# evaluate the trained model
model.predict(np.array([10.0]))  # should be 19 in an ideal world


# In[41]:


# compute error metrics of trained model 

# Ideal relationship: y = 2x - 1
x_value = 10.0
actual = 2 * x_value - 1  # Calculate the true value from the ideal relationship

# Prediction from the model
prediction = model.predict(np.array([x_value]))  # Prediction might be a numpy array

# Access the scalar value from prediction array

# Calculate the errors
errors = prediction_value - actual

# Calculate MSE (Mean Squared Error)
mse = np.square(errors).mean()

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(mse)

# Calculate MAE (Mean Absolute Error)
mae = np.abs(errors).mean()

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.abs(errors / x_value).mean()

# Print the error metrics
print(f"true value (y): {actual}")
print(f"predicted Value: {prediction}")  # Now using prediction_value, which is a scalar
print(f"prediction error: {errors}")
print(f"mean squared Error (mse): {mse}")
print(f"root mean squared error (rmse): {rmse}")
print(f"mean absolute error (): {mae}")
print(f"mean absolute percentage error (mape): {mape}%")


# In[40]:


# convert the .ipynb to a .py file
# !jupyter nbconvert --to script TFLearningLinearRegression.ipynb


# In[ ]:




