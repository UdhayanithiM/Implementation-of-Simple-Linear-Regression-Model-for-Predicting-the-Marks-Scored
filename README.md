# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.



## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Udhayanithi M
RegisterNumber: 212222220054
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Read the CSV data
df = pd.read_csv('/content/student_scores.csv')

# View the beginning and end of the data
df.head()
df.tail()

# Segregate data into variables
x = df.iloc[:, :-1].values
y = df.iloc[:, 1].values

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Create a linear regression model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict values using the model
y_pred = regressor.predict(x_test)

# Display predicted and actual values
print("Predicted values:", y_pred)
print("Actual values:", y_test)

# Visualize the training data
plt.scatter(x_train, y_train, color="black")
plt.plot(x_train, regressor.predict(x_train), color="red")
plt.title("Hours VS scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Visualize the test data
plt.scatter(x_test, y_test, color="cyan")
plt.plot(x_test, regressor.predict(x_test), color="green")
plt.title("Hours VS scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

mae = mean_absolute_error(y_test, y_pred)
print('MAE = ', mae)

rmse = np.sqrt(mse)
print('RMSE = ', rmse)

```

## Output:

## df.head():

![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/3a916bce-527a-4c74-97ff-797b43aa074a)

## df.tail():
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/cec2c102-27c2-4c3d-a806-15414b7599ff)

## Array value of X:
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/784c4710-b0e1-4a33-a257-2b932cf16e83)

## Array value of y:
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/0f1753c9-bc70-443c-a03f-b309b206ecc6)

## Values of Y prediction:
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/6b0c49fc-0903-49f5-9e1d-3fe19687c5b2)

## Values of Y test:
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/2ddf45d5-fe83-4a46-b8bc-8dd90cf0433e)

## Training Set Graph:
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/863c3b2c-ac64-4571-91e8-589a90601f42)

## Test Set Graph:
![image](https://github.com/UdhayanithiM/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/127933352/efc382b6-4290-4dcc-af55-5323484f7809)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
