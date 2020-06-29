# CUSUM ROC Curve for Multiple Changepoint Hypothesis..

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
    nrSections = 5
    for totalWindowTime in (0.015, 0.03, 0.045):
        totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
        maxObservations = 200
        nrSectionObservations = int(maxObservations/nrSections)
        # Graph over ROC.
        plt.title('CUSUM MCH ' + str(totalWindowTime) + ', ' + str(maxObservations) + ', ' + str(nrSections))
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
            # Get IC and OC observations [Ground truth].
            IC = [] # In Control.
            for ic in df.loc[df['flagAttack']==0, ['counts']].values.tolist():
                IC.append(ic[0])
            OC = [] # Out Control.
            for oc in df.loc[df['flagAttack']==1, ['counts']].values.tolist():
                OC.append(oc[0])
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
                    alarms = [] # Alarm at observation t (count).
                    for X in df['mean']: # Run through the dataset.
                        count += 1  # Running observation count.
                        if attack in ('DoS', 'Fuzzy'):  # CUSUM Mean Model.
                            G = max(G - (abs(d_mu)*(X + (abs(d_mu)/2))), 0) # CUSUM Mean Model Negative.
                        elif attack in ('Gear', 'RPM'): # CUSUM Variance Model.
                            G = max(G - math.log(1+d_sigma) - (((X**2) / 2) * ((1 / ((1 + d_sigma) ** 2)) - 1)), 0)
                        # Alarm.
                        if G >= C:
                            alarms.append(count)
                            G = C/2 # Fast Initial Restart (FIR).
                    FP = 0
                    TP = 0
                    for alarm in alarms:
                        if alarm in OC: # If the alarm is TP.
                            TP += 1
                        else:   # An alarm that is not TP, is FP.
                            FP += 1
                    FPR_avg.append(FP/len(IC)) # FPR = FP/N.
                    TPR_avg.append(TP/len(OC)) # TPR = TP/P
                FPR_set.append(Decimal(sum(FPR_avg))/Decimal(len(FPR_avg)))
                TPR_set.append(Decimal(sum(TPR_avg))/Decimal(len(TPR_avg)))
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
        plt.savefig('CUSUM_MCH_' + totalWindowTimeString + '_' + str(maxObservations) + '_' + str(nrSections) + '.png')
        plt.clf()
        plt.close()




if __name__ == '__main__':
    main()
