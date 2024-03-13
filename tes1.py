import pandas as pd
import sklearn
import seaborn as sns
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
df = pd.read_csv(r'drug200.xls')
df.describe()

df.info()
categorical_cols = df.select_dtypes(include=object).columns.to_list()

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes
df = df.rename(columns={'Drug': 'target'})
x = df.drop('target', axis=1)
y = df['target']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=5, shuffle=True)
rforest = RandomForestClassifier()
rforest.fit(x_train, y_train)
y_pred = rforest.predict(x_test)
train_predict = rforest.predict(x_train)
accuracy = accuracy_score(y_train, train_predict)
print("Accuracy : ", accuracy*100, '%')
