# CED For Random Forests.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import plot_roc_curve

def main():

    # -------- #
    attack = 'DoS'
    # attack = 'Fuzzy'
    # attack = 'Gear'
    # attack = 'RPM'
    # -------- #

    startCP = 1
    maxCP = 5   # Looks at n number of changepoints.
    nTrees = 10 # Use of only 10 trees for speed.
    nrSections = 2 # 2 for SCH, CP presumed in perfect center of dataset. Always presumes 50% in control, 50% out control.
    totalObservations = 40000
    nrSectionObservations = int(totalObservations / nrSections)
    rstate = 42
    maxSim = 1000000    # Nr of simulations to produce mean.

    attackData = np.load('attackData.npy',allow_pickle='TRUE').item()

    for ARL in (100, 200):  # Test for ARL0 100 and ARL0 200.
        for totalWindowTime in (0.015, 0.03, 0.045):
            totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
            attackMeanMu = attackData[attack][totalWindowTime]['mean']['mu']
            attackMeanSigma = attackData[attack][totalWindowTime]['mean']['sigma']

            # Train model on a simulated dataset. Take note parameters are the same as tested values.
            df = get_simulatedDataset(attackMeanMu, attackMeanSigma, totalWindowTime, nrSections, nrSectionObservations)
            x = df[['mean']]  # Features
            y = df['flagAttack'] # Labels
            clf = RandomForestClassifier(random_state = rstate, n_estimators=nTrees)
            clf.fit(x, y)

            threshold = 1 - (1 / ARL)   # Threshold based on ARL.
            count = 0
            delay = []
            for CP in range(startCP, maxCP+1):
                for i in range(1, maxSim+1):
                    while True:
                        count += 1
                        if count >= CP:
                            X = np.random.normal(loc=attackMeanMu, scale=attackMeanSigma, size=None)
                        else:
                            X = np.random.normal(loc=0, scale=1, size=None)
                        y_pred = clf.predict_proba([[X]]) # [[prob_0, prob_1]]
                        # Alarm and if CP has occured.
                        if y_pred[0][1] > threshold and count > CP:
                            delay.append(count - CP)
                            count = 0
                            break
                        # Alarm and if CP has not occured yet.
                        elif y_pred[0][1] >= threshold and count <= CP:
                            count = 0
                            break
                    # print('CED:', sum(delay)/len(delay), 'Depth(%):',round((i/maxSim)*100, 2), end="\r")

                CED = sum(delay) / len(delay)
                print()
                print('Attack:', attack)
                print('totalWindowTime:', totalWindowTime)
                print('ARL:', ARL, 'Threshold:', round(1-threshold, 2))
                print('CP:', CP)
                print('CED:', CED)


def get_simulatedDataset(attackMeanMu, attackMeanSigma, totalWindowTime, nrSections, nrSectionObservations):
    # counts
    counts = []
    # windowTimes
    windowTimes = []
    windowTimeStep = totalWindowTime
    windowTime = 0
    # mean
    mean = []
    # flagAttack
    flagAttack = []
    turn = True # Start generate normal data.
    count = 0   # Observation count.
    for i in range(nrSections):
        # Normal data.
        if turn == True:
            # Simulate normal data.
            for j in range(nrSectionObservations):
                # counts
                count += 1
                counts.append(count)
                # windowTimes
                windowTime += windowTimeStep
                windowTimes.append(windowTime)
                # mean
                mean.append(np.random.normal(loc=0, scale=1, size=None))
                # flagAttack
                flagAttack.append(0)    # 0 if normal.
            turn = False
        # Attack data.
        else:
            # Simulate attack data.
            for j in range(nrSectionObservations):
                # counts
                count += 1
                counts.append(count)
                # windowTimes
                windowTime += windowTimeStep
                windowTimes.append(windowTime)
                # mean
                mean.append(np.random.normal(loc=attackMeanMu, scale=attackMeanSigma, size=None))
                # flagAttack
                flagAttack.append(1)
            turn = True
    df = pd.DataFrame(list(zip(counts, windowTimes, mean, flagAttack)), columns=['counts', 'windowTimes', 'mean', 'flagAttack'])
    return df


if __name__ == '__main__':
    main()
