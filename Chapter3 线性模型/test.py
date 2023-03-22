import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

spambase = np.loadtxt('data/spambase/spambase.data', delimiter=",")
dota2results = np.loadtxt('data/dota2Dataset/dota2Train.csv', delimiter=',')

spamx = spambase[:, :57]
spamy = spambase[:, 57]

dota2x = dota2results[:, 1:]
dota2y = dota2results[:, 0]

model1 = LogisticRegression(max_iter=10000)
prediction1 = cross_val_predict(model1, spamx, spamy, cv=10)

model2 = LogisticRegression(max_iter=10000)
prediction2 = cross_val_predict(model2, dota2x, dota2y, cv=10)

acc_s = []
pre_s = []
rec_s = []
f1_s = []

print(acc_s.append(accuracy_score(spamy, prediction1)))
