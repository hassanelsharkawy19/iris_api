import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import pickle

dataset = datasets.load_iris()
Xx = dataset.data
Yy = dataset.target

X,x_test,y,y_test = train_test_split(Xx,Yy,test_size=.2,random_state=20)

model = LogisticRegression()

model.fit(X,y)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)
print(model.score(x_test,y_test))
