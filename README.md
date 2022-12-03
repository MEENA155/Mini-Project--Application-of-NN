# Mini-Project--Application-of-NN


(Expected the following details in the report )
## Project Title:
Stock market prediction
## Project Description:
We can observe that the accuracy achieved by the state-of-the-art ML model is no better than simply guessing with a probability of 50%. Possible reasons for this may be the lack of data or using a very simple model to perform such a complex task as Stock Market prediction.
## Algorithm:
1. Import the necessary pakages.
2. Install the csv file
3. Using the for loop and predict the output
4. Plot the graph
5. Analyze the regression bar plot
### Colaboratory Link :
https://colab.research.google.com/drive/1rknMNlbLphgS6ObhFSfGUWMfB-S_kPIE?usp=sharing
## Program:
### Import the necessary pakages
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
```
### Install the csv file
```
df = pd.read_csv('/content/Tesla.csv')
df.head()df.shape
df.describe()
df.info()
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()
df.head()
df[df['Close'] == df['Adj Close']].shape
df = df.drop(['Adj Close'], axis=1)
df.isnull().sum()
features = ['Open', 'High', 'Low', 'Close', 'Volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.distplot(df[col])
plt.show()
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
plt.subplot(2,3,i+1)
sb.boxplot(df[col])
plt.show()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
plt.pie(df['target'].value_counts().values,labels=[0, 1], autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10, 10))

# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()
features = df[['open-close', 'low-high']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)
models = [LogisticRegression(), SVC(
kernel='poly', probability=True), XGBClassifier()]

for i in range(3):
  models[i].fit(X_train, Y_train)

print(f'{models[i]} : ')
print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
print()
metrics.plot_confusion_matrix(models[0], X_valid, Y_valid)
plt.show()


```


## Output:
![image](https://user-images.githubusercontent.com/94677128/205443186-91c4b8b1-2de8-4f50-b43b-8ecdb087bc09.png)

![image](https://user-images.githubusercontent.com/94677128/205443214-edd0ec9d-3c50-407e-ab6b-df8ee14d27c0.png)

![image](https://user-images.githubusercontent.com/94677128/205443246-474d8641-2f43-48d2-88ad-ff878bd3e0d2.png)

![image](https://user-images.githubusercontent.com/94677128/205443273-b0229d74-af0d-4f46-a1e9-7fab4e5ce0e4.png)

![image](https://user-images.githubusercontent.com/94677128/205443292-8ae591e9-629f-4a63-a512-124c434ccc0a.png)

![image](https://user-images.githubusercontent.com/94677128/205443317-59510226-3c16-479f-b3cd-4cfa55cc9f23.png)

![image](https://user-images.githubusercontent.com/94677128/205443339-fbdd7469-c5b7-474e-ba76-2295ab2ac920.png)

![image](https://user-images.githubusercontent.com/94677128/205443349-d74dd7a9-10c8-4912-b0fd-58ee4c078957.png)

![image](https://user-images.githubusercontent.com/94677128/205443367-4c54ccc0-7468-4751-b948-e5142ef30185.png)

![image](https://user-images.githubusercontent.com/94677128/205443382-82bd44ac-ea35-4329-8ab2-d3dabfa346fe.png)

![image](https://user-images.githubusercontent.com/94677128/205443422-bde79f8e-7f64-4b66-b4ff-4b83950413cc.png)

![image](https://user-images.githubusercontent.com/94677128/205443443-184aa7b8-5870-4a17-8d60-e47783d6f501.png)

![image](https://user-images.githubusercontent.com/94677128/205443464-31751399-6296-48f5-9374-988fc83e54b6.png)

![image](https://user-images.githubusercontent.com/94677128/205443494-1b0c5362-1480-4350-9018-77da3769e2f9.png)

![image](https://user-images.githubusercontent.com/94677128/205443516-1fbae57b-a766-4c21-85e1-91f9c1dfb357.png)


## Advantage :
Python is the most popular programming language in finance. Because it is an object-oriented and open-source language, it is used by many large corporations, including Google, for a variety of projects. Python can be used to import financial data such as stock quotes using the Pandas framework.
## Result:
Thus, stock market prediction is implemented successfully.
