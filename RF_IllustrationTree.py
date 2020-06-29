# Simple illustration of a small tree.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

def main():
    rstate = 42 # For reproducibility.
    nrObservations = 1000
    # dataPath = 'Data/Simulated/'
    dataPath = 'Data/Real/'

    nrSections = 2
    totalWindowTime = 0.03
    nTrees = 10
    totalWindowTimeString = str(totalWindowTime).replace('.','') # Remove dot from float, as string.
    attack = 'DoS'
    # filename = attack + '_' + str(nrObservations) + '_' + str(totalWindowTimeString) + '_' + str(nrSections) + '.csv'
    childWindowTimeString = '0006'
    parentWindow = 5
    filename = attack + '_' + str(childWindowTimeString) + '_' + str(parentWindow) + '.csv'
    df = pd.read_csv(dataPath + filename)
    df = df.head(40000) # Endast första 40,000 observationer.
    df['flagAttack'] = (df['nrAttackPackets'] > 0).astype(int)

    x = df[['mean']]    # Features.
    y = df['flagAttack'] # Labels

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=rstate)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=rstate)

    clf = RandomForestClassifier(random_state = rstate, n_estimators = nTrees, max_depth=2)
    trainedModel = clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    # Extract single tree
    estimator = clf.estimators_[5]

    # Export as dot file
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names = ['X'],
                    class_names = ['0', '1'],
                    rounded = True, proportion = False,
                    precision = 2, filled = True)

    # Convert to png using system command (requires Graphviz)
    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



if __name__ == '__main__':
    main()
