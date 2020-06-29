
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import math
import statistics as stat
import matplotlib.transforms as transforms

def main():
    dataPath = 'Data/Simulated/'
    attackData = np.load('attackData.npy',allow_pickle='TRUE').item()
    totalWindowTime = 0.03
    nrSections = 2
    totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
    maxObservations = 100
    nrSectionObservations = int(maxObservations/nrSections)
    for attack in ('DoS', 'Fuzzy', 'Gear', 'RPM'):
        attackMeanMu = attackData[attack][totalWindowTime]['mean']['mu']
        attackMeanSigma = attackData[attack][totalWindowTime]['mean']['sigma']
        # Read in data
        filename = dataPath + attack + '_' + str(maxObservations) + '_' + totalWindowTimeString + '_' + str(nrSections) + '.csv'
        df = pd.read_csv(filename)
        # CUSUM model parameters.
        d_mu = attackMeanMu-0  # Presumed IC N(0, 1).
        d_sigma = attackMeanSigma-1
        C = attackData[attack][totalWindowTime]['200C']  # 200 ARL.
        G = 0
        count = 0
        xValues = []
        yValues = []
        yS = []
        nrAlarms = 0
        seen = False
        S = 0
        lowest_S = [10, 0]
        for X in df['mean']: # Run through the dataset.
            count += 1  # Running observation count.
            if attack in ('DoS', 'Fuzzy'):  # CUSUM Mean Model.
                G = max(G - (abs(d_mu)*(X + (abs(d_mu)/2))), 0) # CUSUM Mean Model Negative.
                if G <= C and seen == False:
                    S = S - abs(d_mu) * (X+(abs(d_mu)/2))
                    if lowest_S[0] > S:
                        lowest_S[0] = S
                        lowest_S[1] = count
            elif attack in ('Gear', 'RPM'): # CUSUM Variance Model.
                G = max(G - math.log(1+d_sigma) - (((X**2) / 2) * ((1 / ((1 + d_sigma) ** 2)) - 1)), 0)
                if G <= C and seen == False:
                    S = S + (-math.log(1+abs(d_sigma)) - (((X**2)/2)*((1/((1+d_sigma)**2))-1)))
                    if lowest_S[0] > S:
                        lowest_S[0] = S
                        lowest_S[1] = count
            # Alarm.
            if G >= C:
                if seen == False:
                    xCP = count
                    yCP = G
                    seen = True
                nrAlarms += 1
            if nrAlarms == 3:
                break
            xValues.append(count)
            yValues.append(G)
            yS.append(S)


        print('Attack:', attack, 'Lowest S:', lowest_S[1])

        CP = (xCP, round(yCP, 2))
        realCP = 51

        fig, ax=plt.subplots()
        plt.plot(df['counts'], df['mean'], color = 'royalblue', label='$X_t$')
        ax.axvline(realCP, color="red", linestyle='dashed', label="$\\tau$ = " + str(realCP))
        plt.title('Full Observation ' + attack + ', Changepoint ' + 't = ' + str(realCP))
        plt.xlabel('Observed window (t)')
        plt.ylabel('$X_t$')
        plt.legend()
        plt.savefig(attack + '_' + totalWindowTimeString + '_SimulatedDataFullSet.png')
        plt.clf()   # Clear for next plot.


        # CUSUM

        fig, ax=plt.subplots()
        # G
        ax.plot(xValues, yValues, ls="", marker="o", markersize=3, color='darkblue')
        ax.plot(xValues, yValues, color='royalblue', label='G')
        ax.plot(xCP, yCP, color='red', label='Alarm ' + str(CP), marker='o', markersize=5)
        # ax.plot(xValues, yS, ls="", marker="o", markersize=3, color='darkorange')
        # ax.plot(xValues, yS, color='orange', label='S')

        # Horizontal line: C.
        ax.axhline(y=C, color="red", linestyle='dashed', label="$C_y$ = " + str(C))
        # # CP.
        # CP = (xCP, round(yCP, 2))
        # ax.annotate('CP ' + str(CP), xy=CP, xytext=CP, xycoords='data')
        xAxis = list(range(0, len(xValues)+5, 5))
        ax.set_xticks(xAxis)

        ax.axvline(x=lowest_S[1], color="orange", linestyle='dashed', label="$MLE_t$ = " + str(lowest_S[1]))


        plt.title('CUSUM ' + attack + ', Changepoint ' + 't = ' + str(realCP) + ', ARL 200')
        plt.xlabel('Observed window (t)')
        plt.ylabel('G')
        plt.legend()
        plt.savefig(attack + '_' + totalWindowTimeString + '_CUSUM_SimulatedData.png')
        plt.clf()   # Clear for next plot.



def standardized(X, mu, sigma):
    return (X-mu)/sigma



if __name__ == '__main__':
    main()
