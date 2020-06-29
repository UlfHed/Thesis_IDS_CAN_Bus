# CUSUM ROC Curve for Single Changepoint Hypothesis.

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import math
from decimal import *
getcontext().prec = 3


def main():
    dataPath = 'Data/Simulated/'
    attackData = np.load('attackData.npy',allow_pickle='TRUE').item()
    maxSim = 1000   # Nr CUSUM value simulations.
    nrSections = 2  # Only 2 sections for SCH, CP in the middle.
    for totalWindowTime in (0.015, 0.03, 0.045):
        totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
        maxObservations = 200
        nrSectionObservations = int(maxObservations/nrSections)
        CP = maxObservations/nrSections  # Change point in the middle of observation space.
        # Graph over ROC.
        plt.title('CUSUM SCH ' + str(totalWindowTime) + ', ' + str(maxObservations))
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
            # CUSUM model parameters.
            d_mu = abs(attackData[attack][totalWindowTime]['mean']['mu']-0)  # Presumed IC N(0, 1).
            d_sigma = abs(attackData[attack][totalWindowTime]['mean']['sigma']-1)
            FPR_set = [0, 1]    # Start with points (0, 0) and (1, 1).
            TPR_set = [0, 1]
            maxC = attackData[attack][totalWindowTime]['200C']  # 200 ARL.
            for C in range(0, round(maxC*101), 1):
                C = C/100
                FPR_avg = []
                TPR_avg = []
                for i in range(maxSim): # Looking for an average FPR respective TPR of n simulations.
                    count = 0   # init, and reset.
                    G = 0   # init, and reset.
                    for X in df['mean']: # Run through the dataset.
                        count += 1  # Running observation count.
                        if attack in ('DoS', 'Fuzzy'):  # CUSUM Mean Model.
                            G = max(G - (abs(d_mu)*(X + (abs(d_mu)/2))), 0) # CUSUM Mean Model Negative.
                        elif attack in ('Gear', 'RPM'): # CUSUM Variance Model.
                            G = max(G - math.log(1+d_sigma) - (((X**2) / 2) * ((1 / ((1 + d_sigma) ** 2)) - 1)), 0)
                        # Alarm.
                        if G >= C and count < CP:    # Alarm before CP. [Early detection]
                            FPR_avg.append((maxObservations-CP-count)/(maxObservations-CP))
                            TPR_avg.append((maxObservations-CP)/(maxObservations-CP))
                            break
                        elif G >= C and count == CP: # Alarm at CP. [Perfect detection]
                            FPR_avg.append(0)
                            TPR_avg.append(1)
                            break
                        elif G >= C and count > CP:  # Alarm after CP. [Late detection]
                            FPR_avg.append(0)
                            TPR_avg.append((maxObservations-count)/(maxObservations-CP))
                            break
                FPR_set.append(float(Decimal(sum(FPR_avg))/Decimal(len(FPR_avg))))
                TPR_set.append(float(Decimal(sum(TPR_avg))/Decimal(len(TPR_avg))))

            # Sort by FPR.
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
        plt.savefig('CUSUM_SCH_' + totalWindowTimeString + '_' + str(maxObservations) + '.png')
        plt.clf()
        plt.close()



if __name__ == '__main__':
    main()
