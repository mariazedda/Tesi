import pandas as pd
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sport = input("sport : c/f \n")

if sport == "c":
    stringPath = "TesterTrainingFile/calcio"

else:
    stringPath = "TesterTrainingFile/futsal"


for p in range(1, 6):

    p_train = []
    p_test = []

    train = pd.read_csv(stringPath + "/P" + str(p) + "/training.csv")
    test = pd.read_csv(stringPath + "/P" + str(p) + "/test.csv")

    Y_train = train["Score"].values  # memorizzo la variabile dipendente per il train
    X_train = train.drop('Score', axis=1).values  # memorizzo le variabili indipendenti per il train

    X_test = test.drop('Score', axis=1)  # memorizzo le variabili idipendenti per il test

    Y_test = test["Score"].values  # memorizzo la variabile dipendente per il test

    # standardizzo le variabili dipendenti di train e di test
    ss = StandardScaler()
    X_train_std = ss.fit_transform(X_train)
    X_test_std = ss.transform(X_test)

    # imposto il modello
    model = LinearRegression()

    # Addestro il classificatore con i dati di training
    model.fit(X_train_std, Y_train)

    # predizione dati di train
    p_temp_train = model.predict(X_train_std)

    # Predizione sui dati di test
    p_temp_test = model.predict(X_test_std)

    # memorizzo le predizioni negli array
    p_train.extend(p_temp_train)
    p_test.extend(p_temp_test)

    # esprime quanto bene il modello descrive il dataset utilizzato
    m_square_train = mean_squared_error(Y_train, p_train)
    m_square_test = mean_squared_error(Y_test, p_test)
    print("Errore quadratico medio train:", m_square_train)
    print("Errore quadratico medio test:", m_square_test)

    r_train = r2_score(Y_train, p_train)
    r_test = r2_score(Y_test, p_test)
    print("R^2 train:", r_train)
    print("R^2 test:", r_test)

