# RF ROC Curve for Single Changepoint Hypothesis.

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
import math

def main():
    rstate = 42 # For reproducibility.
    nTrees = 10 # Number of trees for RF model.
    dataPath = 'Data/Simulated/'
    attackData = np.load('attackData.npy',allow_pickle='TRUE').item()
    maxSim = 100   # Nr value simulations.
    nrSections = 2  # Only 2 sections for SCH, CP in the middle.
    for totalWindowTime in (0.015, 0.03, 0.045):
        totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
        maxObservations = 200
        nrSectionObservations = int(maxObservations/nrSections)
        CP = maxObservations/nrSections  # Change point in the middle of observation space.
        # Graph over ROC.
        plt.title('RF SCH ' + str(totalWindowTime) + ', ' + str(maxObservations))
        ax = plt.gca()
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
            df_train = get_simulatedDataset(attackMeanMu, attackMeanSigma, totalWindowTime, nrSections, 5*nrSectionObservations)
            clf = RandomForestClassifier(random_state = rstate, n_estimators=nTrees)   # RF model.
            clf.fit(df_train[['mean']], df_train['flagAttack'])   # Train model on training dataset.

            FPR_set = [0, 1]    # Start with points (0, 0) and (1, 1).
            TPR_set = [0, 1]
            for threshold in range(0, 1001, 5): # ARL 200.
                threshold = threshold/1000
                FPR_avg = []
                TPR_avg = []
                for i in range(maxSim): # Looking for an average FPR respective TPR of n simulations.
                    count = 0   # init, and reset.
                    for X in df['mean']: # Run through the dataset.
                        count += 1  # Running observation count.
                        predictionAttack = clf.predict_proba([[X]])[0][1] # Test observation, returns [[prob_0, prob_1]]
                        # Alarm.
                        if count == len(df['mean']):    # No alarm at all.
                            FPR_avg.append(0)   # FPR = 1-TNR=1-(TN/N)
                            TPR_avg.append(0)   # TPR = 1-FNR=1-(FN/p)
                            break   # Redundant.
                        elif predictionAttack >= threshold and count < CP:    # Alarm before CP. [Early detection]
                            FPR_avg.append((maxObservations-CP-count)/(maxObservations-CP))
                            TPR_avg.append((maxObservations-CP)/(maxObservations-CP))
                            break
                        elif predictionAttack >= threshold and count == CP: # Alarm at CP. [Perfect detection]
                            FPR_avg.append(0)
                            TPR_avg.append(1)
                            break
                        elif predictionAttack >= threshold and count > CP:  # Alarm after CP. [Late detection]
                            FPR_avg.append(0)
                            TPR_avg.append((maxObservations-count)/(maxObservations-CP))
                            break
                FPR_set.append(sum(FPR_avg)/len(FPR_avg))
                TPR_set.append(sum(TPR_avg)/len(TPR_avg))

            d = {'FPR': FPR_set, 'TPR': TPR_set}
            p = pd.DataFrame(d).sort_values('FPR')
            # p = p.groupby('FPR').mean().reset_index()
            # Calculate AUC.
            AUC = round(auc(p['FPR'], p['TPR']), 2)
            # Add results to ROC.
            plt.plot(p['FPR'], p['TPR'], label=attack+ ' (AUC = ' + str(AUC) + ')', color=colors[0])
            plt.scatter(p['FPR'], p['TPR'], color=colors[0])
            colors.pop(0)
        plt.legend()
        plt.savefig('RF_SCH_' + totalWindowTimeString + '_' + str(maxObservations) + '.png')
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
