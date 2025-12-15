import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("./data/diabetes.csv")

X = df[["Pregnancies", "Glucose", "BloodPressure", "Age", "BMI"]]
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75)

pca = PCA(n_components = 2)

# pca.fit(X_train)
# X_train_transformed = pca.transform(X_train)

X_train_transformed = pca.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_transformed, y_train)

X_test_transformed = pca.transform(X_test)
print(model.score(X_test_transformed, y_test))

X_pred = pd.DataFrame([
    [5, 100, 80, 30, 33]
], columns = ["Pregnancies", "Glucose", "BloodPressure", "Age", "BMI"])

model.predict(pca.transform(X_pred))
pca.transform(X_pred)