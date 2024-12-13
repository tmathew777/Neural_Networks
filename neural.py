#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

data = pd.read_csv(r"C:\Users\tessa\Downloads\Student_data.csv") 

print(data.info())  

data = data.fillna(0)  

X = data.drop('FirstYearPersistence {1,0}', axis=1) 
y = data['FirstYearPersistence {1,0}']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  
model.add(Dense(64, activation='relu'))  
model.add(Dense(32, activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

print("Model Summary:")
model.summary()

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

y_pred = (model.predict(X_test) > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

model.save(r'C:\Users\tessa\Downloads\neuralnetworks\first_year_persistence_nn_model.keras')



# In[ ]:




