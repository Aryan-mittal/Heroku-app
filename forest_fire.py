
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")
import math

data = pd.read_csv("forestfires.csv")
data = np.array(data) 

data1 = pd.read_csv("Forest_fire.csv")
data1 = np.array(data1)

X = data[1:, 4:13] 
y = data[1:, -1]
y = y.astype('float')
X = X.astype('float')
# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
log_reg = LogisticRegression()

X1 = data1[1:, 1:-1]
Y1 = data1[1:, -1]
Y1 = Y1.astype('float')
X1 = X1.astype('float')

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=0.4, random_state=0)
log_reg1 = LogisticRegression()


log_reg.fit(X_train, y_train)
log_reg1.fit(X1_train, y1_train)

# inputt=[int(x) for x in "86 26 94 5 8 51 6 0 0".split(' ')]
inputt=[int(x) for x in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = log_reg1.predict_proba(final)
print(b)


pickle.dump(log_reg1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))

y_pred = log_reg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(X_test, y_test)))


