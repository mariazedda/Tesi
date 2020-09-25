import pandas as pd
from numpy import mean, var, fft, diff, sign, correlate
from scipy.stats import kurtosis, skew
import csv

stringSens = "dQ dV Mag Quat FreeAcc".split()
stringAx = "w x y z".split()
stringAxi = "x y z".split()
stringFeature = "max min avg var skew kurt aut pks fpk".split()

clm = []
for s in range(5):
    if stringSens[s] == 'dQ' or stringSens[s] == 'Quat':
        for a in range(4):
            for f in range(9):
                if stringFeature[f] == "aut":
                    for u in range(1, 11):
                        clm.append(stringSens[s] + "-" + stringAx[a] + "-" + stringFeature[f] + str(u))
                else:
                    if stringFeature[f] == "pks" or stringFeature[f] == "fpk":
                        for u in range(1, 6):
                            clm.append(stringSens[s] + "-" + stringAx[a] + "-" + stringFeature[f] + str(u))
                    else:
                        clm.append(stringSens[s] + "-" + stringAx[a] + "-" + stringFeature[f])

    if stringSens[s] == 'Mag' or stringSens[s] == 'FreeAcc':
        for a in range(3):
            for f in range(9):
                if stringFeature[f] == "aut":
                    for u in range(1, 11):
                        clm.append(stringSens[s] + "-" + stringAxi[a] + "-" + stringFeature[f] + str(u))
                else:
                    if stringFeature[f] == "pks" or stringFeature[f] == "fpk":
                        for u in range(1, 6):
                            clm.append(stringSens[s] + "-" + stringAxi[a] + "-" + stringFeature[f] + str(u))
                    else:
                        clm.append(stringSens[s] + "-" + stringAxi[a] + "-" + stringFeature[f])

    if stringSens[s] == 'dV':
        for a in range(3):
            for f in range(9):
                if stringFeature[f] == "aut":
                    for u in range(1, 11):
                        clm.append(stringSens[s] + "-" + str(a + 1) + "-" + stringFeature[f] + str(u))
                else:
                    if stringFeature[f] == "pks" or stringFeature[f] == "fpk":
                        for u in range(1, 6):
                            clm.append(stringSens[s] + "-" + str(a + 1) + "-" + stringFeature[f] + str(u))
                    else:
                        clm.append(stringSens[s] + "-" + str(a + 1) + "-" + stringFeature[f])

sport = input("sport : c/f \n")
activity = input("attività : y/n \n")
quality = input("score : y/n \n")
position = input("sensor position : y/n \n")
athleteID = input("athlete ID : y/n \n")

if activity == "y":
    clm.append("Activity")

if quality == "y":
    clm.append("Score")

if position == "y":
    clm.append("Sensor")

if athleteID == "y":
    clm.append("ID")

# Calcolo delle feature e scrittura sul file
if sport == "c":
    stringPath = "data2/calcio"
    fileName = 'featuresCalcio'
else:
    stringPath = "data2/futsal"
    fileName = 'featuresFutsal'


    #with open(fileName + 'P' + str(p) + '.csv', 'w', newline='') as f:
with open('featuresCalcio.csv', 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(clm)
    #if p == 1:
    for p in range(1, 6):
        for s in range(1, 12):
            auxStr = ""
            if s < 10:
                auxStr = "0"
                auxStr += str(s)
                stringPathFile = stringPath + "/p" + str(p) + "/s" + auxStr + ".txt"

                waste = 0

                for X in pd.read_csv(stringPathFile, sep="\t", header=None, chunksize=60):
                    if X[waste + 1].size < 60:
                        break
                    row = []

                    for i in range(2, X.shape[1] - 3):
                        aux = []
                        # Valore massimo
                        aux.append(max(X[i]))

                        # Valore minimo
                        aux.append(min(X[i]))

                        # Valore medio
                        aux.append(mean(X[i]))

                        # Varianza
                        aux.append(var(X[i], ddof=1))

                        # Assimetria
                        aux.append(skew(X[i]))

                        # Curtosi
                        aux.append(kurtosis(X[i]))

                        # Autocorrelazione
                        acf = correlate(X[i], X[i], 'full')[-len(X[0]):]
                        aut = []
                        for j in range(acf.size):
                            if j % 8 == 0:
                                aut.append(acf[j])

                        while len(aut) < 10:
                            tmp = acf[X[i].size - (10 - len(aut))]
                            aut.append(tmp)
                        aux.extend(aut)

                        # Trasformata discreta di Fourier
                        fourier = fft.rfft(X[i] - X[i].mean())
                        freq = fft.rfftfreq(X[i].size, d=1. / 15)
                        inflection = diff(sign(diff(fourier)))
                        peaks = (inflection < 0).nonzero()[0] + 1

                        # primi 5 picchi trasformata di fourier
                        peak = fourier[peaks]

                        # Frequenza dei picchi della trasformata discreta di fourier
                        signal_freq = freq[peaks]

                        aux.extend(peak[:5])
                        aux.extend(signal_freq[:5])
                        row.extend(aux)

                        if i == X.shape[1] - 4:
                            if activity == "y":
                                row.extend([max(X[i + 1])])  # Activity
                            if quality == "y":
                                row.extend([max(X[i + 2])])  # Score
                            if position == "y":
                                row.extend([max(X[i + 3])])  # Position
                            if athleteID == "y":
                                row.extend(["p" + str(p) + ""])  # Athlete

                        waste = i

                    if len(clm) == len(row):
                        wtr.writerow(row)
