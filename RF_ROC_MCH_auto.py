# RF ROC Curve for Multiple Changepoint Hypothesis, use of Scikit function: roc_curve.

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def main():
    rstate = 42 # For reproducibility.
    nTrees = 10 # Number of trees for RF model.
    dataPath = 'Data/Simulated/'
    attackData = np.load('attackData.npy',allow_pickle='TRUE').item()
    for nrSections in (2, 5):
        for totalWindowTime in (0.015, 0.03, 0.045):
            totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
            for maxObservations in (100, 500, 1000):
                nrSectionObservations = int(maxObservations/nrSections)
                # Graph over ROC.
                ax = plt.gca()
                plt.title('RF MCH ' + str(totalWindowTime) + ', ' + str(maxObservations) + ', ' + str(nrSections))
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linestyle='dashed')
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                colors = ['green', 'blue', 'orange', 'red']
                for attack in ('DoS', 'Fuzzy', 'Gear', 'RPM'):
                    attackMeanMu = attackData[attack][totalWindowTime]['mean']['mu']
                    attackMeanSigma = attackData[attack][totalWindowTime]['mean']['sigma']
                    # Read in data
                    filename = dataPath + attack + '_' + str(maxObservations) + '_' + totalWindowTimeString + '_' + str(nrSections) + '.csv'
                    df = pd.read_csv(filename)
                    # Create a training dataset, similar to observation.
                    df_train = get_simulatedDataset(attackMeanMu, attackMeanSigma, totalWindowTime, nrSections, nrSectionObservations)
                    clf = RandomForestClassifier(random_state = rstate, n_estimators=nTrees)   # RF model.
                    clf.fit(df_train[['mean']], df_train['flagAttack'])   # Train model on training dataset.

                    truths = []
                    predictions = []
                    count = 0
                    for X in df['mean']:
                        count += 1  # Running observation count.
                        truths.append(df['flagAttack'][count-1]) # True flag of observation. [Ground truth].
                        predictionAttack = clf.predict_proba([[X]])[0][1] # Test observation, returns [[prob_0, prob_1]]
                        predictions.append(predictionAttack)
                    FPR, TPR, t = roc_curve(truths, predictions, pos_label=1)
                    # Calculate AUC.
                    AUC = round(auc(FPR, TPR), 2)
                    # Add results to ROC.
                    plt.plot(FPR, TPR, label=attack+ ' (AUC = ' + str(AUC) + ')', color=colors[0])
                    colors.pop(0)
                plt.legend()
                plt.savefig('RF_MCH_' + totalWindowTimeString + '_' + str(maxObservations) + '_' + str(nrSections) + '_AutoROC.png')
                plt.clf()
                plt.close()



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
