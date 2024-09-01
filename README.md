# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

To build a neural network regression model for predicting a continuous target variable, we will follow a systematic approach. The process includes loading and pre-processing the dataset by addressing missing data, scaling the features, and ensuring proper input-output mapping. Then, a neural network model will be constructed, incorporating multiple layers designed to capture intricate patterns within the dataset. We will train this model, monitoring performance using metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE), to ensure accurate predictions. After training, the model will be validated and tested on unseen data to confirm its generalization ability. The final objective is to derive actionable insights from the data, helping to improve decision-making and better understand the dynamics of the target variable. Additionally, the model will be fine-tuned to enhance performance, and hyperparameter optimization will be carried out to further improve predictive accuracy. The resulting model is expected to provide a robust framework for making precise predictions and facilitating in-depth data analysis.

## Neural Network Model

![nn ex 1](https://github.com/user-attachments/assets/5ac98df1-0b2b-40a4-84b9-a1abba63a49c)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Prema Latha S
### Register Number: 212222230112
```python


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from google.colab import auth
import gspread
from google.auth import default
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl_ex1').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input':'float'})
dataset1 = dataset1.astype({'Output':'float'})
dataset1.head()
X = dataset1[['Input']].values
y = dataset1[['Output']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
    ])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(X_train1,y=y_train,epochs=2000)
loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test1 = Scaler.transform(X_test)
ai_brain.evaluate(X_test1,y_test)
X_n1 = [[4]]
X_n1_1 = Scaler.transform(X_n1)
ai_brain.predict(X_n1_1)



```
## Dataset Information

![dataset](https://github.com/user-attachments/assets/c92fa336-f5d4-4db5-95ea-1c1006161c0e)


## OUTPUT

### Training Loss Vs Iteration Plot

![output](https://github.com/user-attachments/assets/5973e5e6-a392-48b4-acd2-4dceee59d90c)


### Test Data Root Mean Squared Error

![output](https://github.com/user-attachments/assets/7031e3ee-70ca-4ed2-af4d-251eec1449bc)

### New Sample Data Prediction

![output](https://github.com/user-attachments/assets/68d14b8c-82d0-4bcb-9605-8cfb8c48f18d)

## RESULT

Thus a neural network regression model for the given dataset is developed and the prediction for the given input is obtained accurately.
