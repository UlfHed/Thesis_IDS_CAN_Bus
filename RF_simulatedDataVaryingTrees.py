# ROC plot for simulated data to evaluate variation of trees for the Random Forests method.

import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix

def main():
    rstate = 42 # For reproducibility.
    nrObservations = 40000
    dataPath = 'Data/Simulated/'
    for nrSections in (2, 5, 20):
        for totalWindowTime in (0.015, 0.03, 0.045):
            totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
            for nTrees in (5, 10, 100):
                # ax = plt.gca()
                # plt.title('RF Simulated Dataset ' + str(totalWindowTime)+ ' ' + str(nTrees))
                # ax.set_xlim([0,1])
                # ax.set_ylim([0,1])
                # ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linestyle='dashed')
                # plt.xlabel('FPR')
                # plt.ylabel('TPR')
                # colors = ['green', 'blue', 'orange', 'red']
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

                    # Confusion Matrix
                    np.set_printoptions(precision=2)
                    disp = plot_confusion_matrix(clf, x_test, y_test,
                                                 display_labels=['No attack', 'Attack'],
                                                 cmap=plt.cm.Blues,
                                                 normalize='true')
                    disp.ax_.set_title('CM ' + attack + ' ' + str(totalWindowTime) + ' Simulated Data, nTrees:' + str(nTrees))
                    # Save graph
                    plt.savefig('RF_CM_' + attack + '_' + totalWindowTimeString + '_nTrees_' + str(nTrees) + '_SimulatedData.png')
                    plt.clf()
                    plt.close()


                #     plot_roc_curve(clf, x_test, y_test, ax=ax, alpha=0.75, name=attack)
                #
                # plt.savefig('RF_SimulatedDataset_' + totalWindowTimeString + '_' + str(nrObservations) + '_' + str(nrSections) + '_' + str(nTrees) + '.png')
                # plt.clf()
                # plt.close()




if __name__ == '__main__':
    main()
