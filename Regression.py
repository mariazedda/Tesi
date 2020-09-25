import csv

import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sport = input("sport : c/f \n")

if sport == "c":
    stringPath = "TesterTrainingFile\\calcio"

else:
    stringPath = "TesterTrainingFile\\futsal"


for p in range(1, 6):

    train = pd.read_csv(stringPath + "\\P" + str(p) + "\\training.csv")
    test = pd.read_csv(stringPath + "\\P" + str(p) + "\\test.csv")

    Y_train = train["Score"] #.values  # memorizzo la variabile dipendente per il train
    #Y_train = train.Score
    X_train = train.drop(['Score'], axis=1)  #.values  # memorizzo le variabili indipendenti per il train
    print(X_train)
    Y_test = test["Score"]  #.values  # memorizzo la variabile dipendente per il test
    #Y_test = test.Score
    X_test = test.drop('Score', axis=1) #.values  # memorizzo le variabili idipendenti per il test

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    # standardizzo le variabili dipendenti di train e di test
    ss = StandardScaler()
    X_train_std = ss.fit_transform(X_train)
    X_test_std = ss.transform(X_test)

    # imposto il modello
    model = LinearRegression()

    # Addestro il classificatore con i dati di training
    model.fit(X_train_std, Y_train)

    # Predizione sui dati di test
    Y_pred_test = model.predict(X_test_std)

    # esprime quanto bene il modello descrive il dataset utilizzato
    m_square_test = mean_squared_error(Y_test, Y_pred_test)
    print("Errore quadratico medio test: %f" % m_square_test)

    r_test = r2_score(Y_test, Y_pred_test)
    print("R^2 test:", r_test)
    with open('prediction.csv', 'a', newline='') as f:
        wtr = csv.writer(f)
        wtr.writerow(X_train)
        wtr.writerow(X_test)

