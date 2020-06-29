# ROC plot for the Simulated data by Random Forests.

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn import tree

def main():
    rstate = 42 # For reproducibility.
    nTrees = 100 # Number of trees for RF model.
    nrObservations = 40000
    dataPath = 'Data/Simulated/'
    # for nrSections in (2, 5):
    for nrSections in (2, 5, 20):
        for totalWindowTime in (0.015, 0.03, 0.045):
            totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
            ax = plt.gca()
            plt.title('RF Simulated Dataset ' + str(totalWindowTime))
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linestyle='dashed')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            colors = ['green', 'blue', 'orange', 'red']
            for attack in ('DoS', 'Fuzzy', 'Gear', 'RPM'):
                filename = attack + '_' + str(nrObservations) + '_' + str(totalWindowTimeString) + '_' + str(nrSections) + '.csv'
                df = pd.read_csv(dataPath + filename)

                x = df[['mean']]    # Features.
                y = df['flagAttack'] # Labels

                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=rstate)
                x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=rstate)

                clf = RandomForestClassifier(random_state = rstate, n_estimators = nTrees)
                trainedModel = clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)

                print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

                plot_roc_curve(clf, x_test, y_test, ax=ax, alpha=0.75, name=attack)

            plt.savefig('RF_SimulatedDataset_' + totalWindowTimeString + '_' + str(nrObservations) + '_' + str(nrSections) + '.png')
            plt.clf()
            plt.close()




if __name__ == '__main__':
    main()
