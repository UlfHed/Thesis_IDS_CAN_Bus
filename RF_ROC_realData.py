# ROC plot for the real data.

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
    dataPath = 'Data/Real/'
    parentWindow = 5
    for childWindow in (0.003, 0.006, 0.009):
        totalWindowTime = childWindow * parentWindow
        childWindowTimeString = str(childWindow).replace('.','') # Remove dot from float, as string.
        totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
        ax = plt.gca()
        plt.title('RF Real Dataset ' + str(totalWindowTime))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='grey', linestyle='dashed')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        colors = ['green', 'blue', 'orange', 'red']
        for attack in ('DoS', 'Fuzzy', 'Gear', 'RPM'):
            filename = attack + '_' + str(childWindowTimeString) + '_' + str(parentWindow) + '.csv'
            df = pd.read_csv(dataPath + filename)
            df = df.head(40000) # Endast första 40,000 observationer.
            df['flagAttack'] = (df['nrAttackPackets'] > 0).astype(int)

            x = df[['mean']]    # Features.
            y = df['flagAttack'] # Labels

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=rstate)
            x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=rstate)

            clf = RandomForestClassifier(random_state = rstate, n_estimators = nTrees)
            trainedModel = clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)

            print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

            plot_roc_curve(clf, x_test, y_test, ax=ax, alpha=0.75, name=attack)

        plt.savefig('RF_RealDataset_' + totalWindowTimeString + '.png')
        plt.clf()
        plt.close()

            # estimator = clf.estimators_[0]
            # from sklearn.tree import export_graphviz
            # export_graphviz(estimator, out_file='tree.dot', rounded=True, proportion=False, precision = 2, filled=True)
            # from subprocess import call
            # filename = 'RF_RealDataset_Tree_' + attack + '_' + totalWindowTimeString + '.png'
            # call(['dot', '-Tpng', 'tree.dot', '-o', filename, '-Gdpi=300'])
            # exit()



if __name__ == '__main__':
    main()
