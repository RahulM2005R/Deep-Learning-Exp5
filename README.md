# Exp-5: Recurrent Neural Network model for Stock Price Prediction

## **AIM:**

To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## **THEORY**

### **Neural Network Model**

<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/a9c2d0eb-4d5c-4c6d-9a9f-9fc4154a4488" />


**DESIGN STEPS**

**Step-1** Read the CSV file and create the DataFrame using pandas.

**Step-2** Select the “Open” column (or any desired column) and scale the values using MinMaxScaler.

**Step-3** Create two lists — X_train and y_train — where X_train stores 60 readings as input and the 61st reading as output in y_train.

**Step-4** Build an LSTM model with the desired number of neurons and a single output neuron.

**Step-5** Combine the training and test data, then prepare X_test using the same 60-step sequence logic.

**Step-6** Use the trained model to make predictions on the test data and inverse transform the results to their original scale.

**Step-7** Plot the graph comparing the Actual and Predicted stock prices using matplotlib.

##**PROGRAM**

**Name:Rahul M R**

**Register Number: 2305003005**

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from keras.models import Sequential

dataset_train = pd.read_csv('trainset.csv')

dataset_train.columns

dataset_train.head()

train_set = dataset_train.iloc[:,1:2].values
type(train_set)
train_set.shape

sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(train_set)
training_set_scaled.shape

X_train_array = []
y_train_array = []
for i in range(60, 1259):
  X_train_array.append(training_set_scaled[i-60:i,0])
  y_train_array.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train_array), np.array(y_train_array)
X_train1 = X_train.reshape((X_train.shape[0], X_train.shape[1],1))

X_train.shape

length = 60
n_features = 1

model = Sequential([layers.SimpleRNN(40,input_shape=(60,1)),
                    layers.Dense(1)])
model.compile(optimizer='adam',loss='mse')
model.summary()

model.compile(optimizer='adam', loss='mse')

model.fit(X_train1,y_train,epochs=25, batch_size=64)

model.summary()
dataset_test = pd.read_csv('testset.csv')
test_set = dataset_test.iloc[:,1:2].values
test_set.shape
dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)

inputs = dataset_total.values
inputs = inputs.reshape(-1,1)
inputs_scaled=sc.transform(inputs)
X_test = []
for i in range(60,1384):
  X_test.append(inputs_scaled[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0], X_test.shape[1],1))

X_test.shape
predicted_stock_price_scaled = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price_scaled)

plt.plot(np.arange(0,1384),inputs, color='red', label = 'Test(Real) Google stock price')
plt.plot(np.arange(60,1384),predicted_stock_price, color='blue', label = 'Predicted Google stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
```


## **OUTPUT**

**Epoch Training:**

<img width="524" height="336" alt="image" src="https://github.com/user-attachments/assets/574814bd-2653-427b-9ac3-6d4f36e7dc14" />

---

**Model Training Loss Across Epochs:**

<img width="730" height="470" alt="image" src="https://github.com/user-attachments/assets/276214c5-9141-4777-90b5-6f7033a829c6" />

---

**True Stock Price, Predicted Stock Price vs time**

<img width="580" height="455" alt="image" src="https://github.com/user-attachments/assets/9f620edc-f467-4312-8aeb-3455bbdf41b2" />

---

**Predicted Value**

<img width="441" height="49" alt="image" src="https://github.com/user-attachments/assets/ebd75e5a-8cf4-4ca8-af56-e54741153b10" />

## **RESULT**
Thus, a reccurrent neural network for Stock Price Prediction developed successfully.
