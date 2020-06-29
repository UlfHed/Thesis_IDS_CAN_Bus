# Derive characteristics from data necessary for simulation.

import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def main():
    # Parent directory.
    dataPath = 'Data/Real/'

    # Limit read in observations. limit 40 000. 0.045 RPM and Gear data past 40 000 is not reliable.
    n = 40000

    parentWindowSize = 5
    for childWindowSize in (0.003, 0.006, 0.009):
        totalWindowTime = childWindowSize * parentWindowSize
        childWindowSizeString = str(childWindowSize).replace('.','') # Remove dot from float, as string.
        dataDirectoryString = str(totalWindowTime).replace('.','') + '/' # Remove dot from float, as string.

        # Datasets:
        data_DoS = dataPath + 'DoS_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_Fuzzy = dataPath + 'Fuzzy_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_Gear = dataPath + 'Gear_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_Normal = dataPath + 'Normal_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_RPM = dataPath + 'RPM_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'

        # Header.
        print()
        print('*'*50)
        print('Data information from:')
        print('childWindowSize:', childWindowSize)
        print('parentWindowSize:', parentWindowSize)
        print('Resulting time window:', childWindowSize * parentWindowSize)
        print('*'*50)

        # Start.
        datasetNames = ('DoS', 'Fuzzy', 'Gear', 'Normal', 'RPM')
        count = 0
        for dataset in (data_DoS, data_Fuzzy, data_Gear, data_Normal, data_RPM):
            datasetName = datasetNames[count]
            count += 1

            # Get data, dataframe.
            df = pd.read_csv(dataset)
            # Remove for full.
            df = df.head(n)

            print()
            print('-'*20 + ' ' + datasetName + ' ' + '-'*20)
            print('Total nr packets:', sum(df['nrPackets']))
            print('Total nr Observations:', len(df['nrPackets']))

            # ---- Stochastic variable: vector mean. ---- #
            print()
            print('Vector mean (stochastic variable):')
            # Scaling parameters for standardization; mean and std from in control observations.
            normalObservations = df.loc[df['nrAttackPackets'] == 0, ['mean']]
            mean = normalObservations['mean'].mean()
            std = normalObservations['mean'].std()

            print(' --- Scaling parameters (Normal observations):')
            print(' ---  --- Mean:', mean)
            print(' ---  --- Std:', std)
            print()

            # Standardize vector means.
            df['meanStandardized'] = standardize(df['mean'], mean, std)

            # Get in control observations of mean (after standardization).
            normalObservations = df.loc[df['nrAttackPackets'] == 0, ['meanStandardized']]
            # Get out control observations of mean (after standardization).
            attackObservations = df.loc[df['nrAttackPackets'] > 0, ['meanStandardized']]

            print(' --- Normal observations (in control) after standardization:')
            print(' ---  --- Mean:', normalObservations['meanStandardized'].mean())
            print(' ---  --- Std:', normalObservations['meanStandardized'].std())
            print(' ---  --- Variance:', normalObservations['meanStandardized'].std() ** 2)
            print(' ---  --- Nr observations:', len(normalObservations['meanStandardized']), ' |', round((len(normalObservations['meanStandardized'])/len(df['mean'])*100), 2), '%')
            print()
            print(' --- Attack observations (out control) after standardization:')
            print(' ---  --- Mean:', attackObservations['meanStandardized'].mean())
            print(' ---  --- Std:', attackObservations['meanStandardized'].std())
            print(' ---  --- Variance:', attackObservations['meanStandardized'].std() ** 2)
            print(' ---  --- Nr observations:', len(attackObservations['meanStandardized']), ' |', round((len(attackObservations['meanStandardized'])/len(df['mean'])*100), 2), '%')
            print()

            # # Stochastic variable: vector stdev.
            # print('Vector stdev (stochastic variable):')
            #
            # # Scaling parameters for standardization; mean and std from in control observations.
            # normalObservations = df.loc[df['nrAttackPackets'] == 0, ['stdev']]
            # mean = normalObservations['stdev'].mean()
            # std = normalObservations['stdev'].std()
            #
            # print(' --- Scaling parameters (Normal observations):')
            # print(' ---  --- Mean:', mean)
            # print(' ---  --- Std:', std)
            # print()
            #
            # # Standardize vector stdevs.
            # df['stdevStandardized'] = standardize(df['stdev'], mean, std)
            #
            # # Get in control observations of stdev (after standardization).
            # normalObservations = df.loc[df['nrAttackPackets'] == 0, ['stdevStandardized']]
            # # Get out control observations of stdev (after standardization).
            # attackObservations = df.loc[df['nrAttackPackets'] > 0, ['stdevStandardized']]
            #
            # print(' --- Normal observations (in control) after standardization:')
            # print(' ---  --- Mean:', normalObservations['stdevStandardized'].mean())
            # print(' ---  --- Std:', normalObservations['stdevStandardized'].std())
            # print(' ---  --- Variance:', normalObservations['stdevStandardized'].std() ** 2)
            # print(' ---  --- Nr observations:', len(normalObservations['stdevStandardized']), ' |', round((len(normalObservations['stdevStandardized'])/len(df['stdev'])*100), 2), '%')
            # print()
            # print(' --- Attack observations (out control) after standardization:')
            # print(' ---  --- Mean:', attackObservations['stdevStandardized'].mean())
            # print(' ---  --- Std:', attackObservations['stdevStandardized'].std())
            # print(' ---  --- Variance:', attackObservations['stdevStandardized'].std() ** 2)
            # print(' ---  --- Nr observations:', len(attackObservations['stdevStandardized']), ' |', round((len(attackObservations['stdevStandardized'])/len(df['stdev'])*100), 2), '%')
            # print()


def standardize(data, mean, std):
    # Standardize n, based on know mean and stdev of the population n belongs to.
    zValues = []
    for i in data:
        zValues.append( (i - mean) / std )
    return zValues


if __name__ == '__main__':
    main()
