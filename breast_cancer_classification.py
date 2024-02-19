
#Importing the libraries
import pandas as pd
import numpy as np

#Uploading the Dataset
from google.colab import files
uploaded = files.upload()

#Reading the data
dataset = pd.read_csv("data.csv")

#Summarize the dataset
print(dataset.shape)
dataset.head()

#Mapping Class String Values to Varaibles
dataset['diagnosis'] = dataset['diagnosis'].map({"M":0,"B":1}).astype(int)

#Segreting the Dataset into X
X = dataset.iloc[:, 2:32].values
X

#Segreting the Dataset into Y
Y = dataset.iloc[:,1]
Y

#Splitting Dataset into Train & Test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.25, random_state=0)

# Future Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Selecting the Algorithm
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='liblinear', multi_class='ovr')

# Predicting the algorithm
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
value = [[12.45,15.7,82.57,	477.1,	0.1278,	0.17,	0.1578,	0.08089,	0.2087,	0.07613, 0.3345,	0.8902,	2.217,	27.19,	0.00751,	0.03345,	0.03672,	0.01137,	0.02165,	0.005082,	15.47,	23.75,	103.4,	741.6,	0.1791,	0.5249,	0.5355,	0.1741,	0.3985,	0.1244]]
y_pred = model.predict(value)
print(y_pred)

#Evaluation Metrics
from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix: ")
print(cm)
