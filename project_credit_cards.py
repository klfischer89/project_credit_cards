import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, plot_confusion_matrix


df = pd.read_csv("./data/creditcard.csv.bz2")

# sns.heatmap(df.corr())
# plt.show()


X = df[["V17", "V14", "V12", "V10", "V16", "V3", "V7", "V11", "V4", "V18"]]
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

model = LogisticRegression(class_weight = "balanced")
model.fit(X_train, y_train)

print(model.score(X_test, y_test))

y_test_pred = model.predict(X_test)

print("Recall: " + str(recall_score(y_test, y_test_pred)))
print("Precision: " + str(precision_score(y_test, y_test_pred)))

plot_confusion_matrix(model, X_test, y_test, normalize = "all")