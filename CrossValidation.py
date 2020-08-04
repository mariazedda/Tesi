import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from matplotlib import pyplot as plt

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import seaborn as sns
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

# Load the Diabetes dataset
train = pd.read_csv("C:\\Users\\pc\\Desktop\\Tesi\\TestAndTraining\\futsal\\P1\\training.csv")
test = pd.read_csv("C:\\Users\\pc\\Desktop\\Tesi\\TestAndTraining\\futsal\\P1\\test.csv")

classi = ["1000 metri", "Navetta 5x10", "Scatto 10m", "Triplo salto in lungo"]

y = train.Activity
X = train.drop('Activity', axis=1)

#fit a model (Random Forest)
model = DecisionTreeClassifier(random_state=None)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train)

#Predizione sui dati di training
p_train = model.predict(X_train)

#Predizione sui dati di test
p_test = model.predict(X_test)

#Calcolo della accuratezza
acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)

print("\nAccuracy")
print(f'Train {acc_train}, Test {acc_test}')


#Predizioni con il classificatore addestrato con dati di training
predictions = cross_val_predict(model, X, y)
arr = precision_recall_fscore_support(y, predictions, average='weighted')

print("\nPrecision: ", arr[0], "\nRecall: ", arr[1], "\nF-1:", arr[2])

# Confusion Matrix
C = confusion_matrix(predictions, y)
df_cm = pd.DataFrame(C, classi, classi)
sns.set(font_scale=1.4)
graphic = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap="YlGnBu", fmt='g')
plt.show()



