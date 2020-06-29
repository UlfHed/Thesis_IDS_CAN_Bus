# Create the simulated data.

import pandas as pd
import numpy as np
import csv

def main():
    dataPath = 'Data/Simulated/'

    attackData = np.load('attackData.npy',allow_pickle='TRUE').item()   # Load data from analysis.
    for nrSections in (2, 5, 20):   # Two separate datasets, 2 sections: 1 in control, 1 out control. 5 sections: 3 in control, 2 out control.
        for totalWindowTime in (0.015, 0.03, 0.045):    # 3 seperate windowsizes are studied.
            totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
            for maxObservations in (50, 75, 100, 150, 200, 500, 1000, 40000):  # Arbitrary observationspace (nr of observations).
                nrSectionObservations = int(maxObservations/nrSections)
                for attack in ('DoS', 'Fuzzy', 'Gear', 'RPM'):  # Each seperate attack studied.
                    attackMeanMu = attackData[attack][totalWindowTime]['mean']['mu']
                    attackMeanSigma = attackData[attack][totalWindowTime]['mean']['sigma']
                    df = get_simulatedDataset(attackMeanMu, attackMeanSigma, totalWindowTime, nrSections, nrSectionObservations)
                    # Write file.
                    filename = dataPath + attack + '_' + str(maxObservations) + '_' + totalWindowTimeString + '_' + str(nrSections) + '.csv'
                    df.to_csv(filename, encoding='utf-8', index=False)

# Simulate data.
def get_simulatedDataset(attackMeanMu, attackMeanSigma, totalWindowTime, nrSections, nrSectionObservations):
    counts = []
    windowTimes = []
    windowTimeStep = totalWindowTime
    windowTime = 0
    mean = []
    flagAttack = []
    turn = True
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
