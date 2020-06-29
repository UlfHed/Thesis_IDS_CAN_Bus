# Get visual representation of dataset.

import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def main():
    # Parent directory.
    dataPath = 'Data/'

    # Limit read in observations. limit 40 000. 0.045 RPM and Gear data past 40 000 is not reliable.
    n = 40000

    # Each windowsize.
    for childWindowSize in (0.003, 0.006, 0.009):
        parentWindowSize = 5
        totalWindowTime = childWindowSize * parentWindowSize

        childWindowSizeString = str(childWindowSize).replace('.','') # Remove dot from float, as string.
        dataDirectoryString = 'Real/'

        # Datasets:
        data_DoS = dataPath + dataDirectoryString + 'DoS_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_Fuzzy = dataPath + dataDirectoryString + 'Fuzzy_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_Gear = dataPath + dataDirectoryString + 'Gear_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_Normal = dataPath + dataDirectoryString + 'Normal_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
        data_RPM = dataPath + dataDirectoryString + 'RPM_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'

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

            # Histogram of full normal dataset.
            if datasetName == 'Normal':
                # Histogram observed packets.
                mu, std = norm.fit(df['nrPackets'])
                plt.hist(df['nrPackets'], bins=15, density=True, alpha=0.6, color='blue', edgecolor='black', linewidth=1)
                # Plot the PDF.
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'g', linewidth=2)
                fit = "Fit: mu = %.2f,  std = %.2f" % (mu, std)
                plt.title(datasetName + ' Observation Packets, ' + fit)
                plt.xlabel('Observation # Packets')
                plt.ylabel('Density')
                plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + '_histogramObservationPackets.png')
                plt.clf()   # Clear for next plot.

                # Histogram observed mean.
                mu, std = norm.fit(df['mean'])
                plt.hist(df['mean'], bins=15, density=True, alpha=0.6, color='blue', edgecolor='black', linewidth=1)
                # Plot the PDF.
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'g', linewidth=2)
                fit = "Fit: mu = %.2f,  std = %.2f" % (mu, std)
                plt.title(datasetName + ' Observation Mean, ' + fit)
                plt.xlabel('Observation Vector Means')
                plt.ylabel('Density')
                plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + '_histogramObservationMean.png')
                plt.clf()   # Clear for next plot.

                # Histogram observed stdev.
                mu, std = norm.fit(df['stdev'])
                plt.hist(df['stdev'], bins=25, density=True, alpha=0.6, color='blue', edgecolor='black', linewidth=1)
                # Plot the PDF.
                xmin, xmax = plt.xlim()
                x = np.linspace(xmin, xmax, 100)
                p = norm.pdf(x, mu, std)
                plt.plot(x, p, 'g', linewidth=2)
                fit = "Fit: mu = %.2f,  std = %.2f" % (mu, std)
                plt.title(datasetName + ' Observation Stdev, ' + fit)
                plt.xlabel('Observation Vector Stdevs')
                plt.ylabel('Density')
                plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + '_histogramObservationStdev.png')
                plt.clf()   # Clear for next plot.

            # Observed packets.
            plt.title(datasetName + ' Observation # Packets')
            plt.plot(df['Counts'], df['nrPackets'], color='blue')
            plt.xlabel('Observed window (t)')
            plt.ylabel('# Packets')
            plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + '_observationPackets.png')
            plt.clf()   # Clear for next plot.

            # Observed mean.
            plt.title(datasetName + ' Observation Vector Mean')
            plt.plot(df['Counts'], df['mean'], color='blue')
            plt.xlabel('Observed window (t)')
            plt.ylabel('Vector Mean')
            plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + '_observationMean.png')
            plt.clf()   # Clear for next plot.

            # Observed stdev.
            plt.title(datasetName + ' Observation Vector Stdev')
            plt.plot(df['Counts'], df['stdev'], color='blue')
            plt.xlabel('Observed window (t)')
            plt.ylabel('Vector Stdev')
            plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + '_observationStdev.png')
            plt.clf()   # Clear for next plot.

            # Attack observation graph.
            plt.title(datasetName + ' Observation # Attack Packets')
            plt.plot(df['Counts'], df['nrAttackPackets'], color='red')
            plt.xlabel('Observed window (t)')
            plt.ylabel('# Attack Packets')
            plt.savefig(dataPath +  'Graphs/' + 'Real/' + datasetName + 'observationAttackPackets.png')
            plt.clf()   # Clear for next plot.


if __name__ == '__main__':
    main()
