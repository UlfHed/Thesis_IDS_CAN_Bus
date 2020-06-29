# Process the raw dataset into data of use.
# 'windowNrs', 'windowTimes', 'windowOccurences', 'windowFlags', 'windowFlagOccurences'

import csv
import pandas as pd
import statistics

def main():
    # CarHacking Data.
    # -------------------------------------------------------------------------------------------------------------------------------
    dataPath = '/home/phed/Projects/Thesis_dataset/CarHacking/Original/'
    # CSV data is as follows:
    # timestamp, CAN ID, DLC, DATA[0], DATA[1], DATA[2], DATA[3], DATA[4], DATA[5], DATA[6], DATA[7], Flag
    data_rawDoS = dataPath + 'DoS_dataset.csv'
    data_rawFuzzy = dataPath + 'Fuzzy_dataset.csv'
    data_rawGear = dataPath + 'gear_dataset.csv'
    data_rawRPM = dataPath + 'RPM_dataset.csv'
    data_rawNormal = dataPath + 'normal_run_data.txt'   # Take note .txt file.
    # -------------------------------------------------------------------------------------------------------------------------------

    maxRead = 'all'    # nr lines (packets), or 'all' for the whole dataset.
    maxWindowNr = 'all'    # Past approx 200 000 observations each dataset is unreliable.

    # Each 3 child window sizes.
    for windowSize in (0.003, 0.006, 0.009):
        childWindowSizeString = str(windowSize).replace('.','') # Remove dot from float, as string.
        parentWindowSize = 5  # Size of parent window.

        datasetNames = ('DoS', 'Fuzzy', 'Gear', 'RPM', 'Normal')
        datasetCount = 0
        for dataset in (data_rawDoS, data_rawFuzzy, data_rawGear, data_rawRPM, data_rawNormal):
            datasetName = datasetNames[datasetCount]
            datasetCount += 1
            print('Processing ' + datasetName + '...')

            # Normal data file handled as .txt file not .csv.
            if datasetName == 'Normal':
                data = readInDataNormal(dataset, maxRead)
                windows = get_windowsNormal(data, windowSize, maxWindowNr)
            else:
                data = readInData(dataset, maxRead)
                windows = get_windows(data, windowSize, maxWindowNr)

            parentWindows = get_parentWindows(windows, parentWindowSize)

            data = 0 # Memory flush.
            windows = 0 # Memory flush.

            # Pandas dataframe.
            df = pd.DataFrame(
                list(zip(parentWindows[0], parentWindows[1], parentWindows[2], parentWindows[3], parentWindows[4], parentWindows[5])),
                columns=['Counts', 'windowTimes', 'nrPackets', 'mean', 'stdev', 'nrAttackPackets']
            )

            # Write .csv file.
            fileName = 'Data/' + 'Real/' + datasetName + '_' + childWindowSizeString + '_' + str(parentWindowSize) + '.csv'
            df.to_csv(fileName, encoding='utf-8', index=False)
            df = 0 # Memory flush.



def get_parentWindows(windows, parentWindowSize):
    # parentWindowCounts, parentWindowTimes, parentWindowTotalPackets, parentWindowMeans, parentWindowStdevs, parentWindowTotalAttackPackets
    timeStep = windows[1][0]
    observedChildWindows = [] # List of child windows, total nr packets for each child.
    count = 0
    parentWindowCount = 0
    # Data returned
    parentWindowCounts = []
    parentWindowTimes = []
    parentWindowTotalPackets = []
    parentWindowMeans = []
    parentWindowStdevs = []
    for i in windows[2]:
        count += 1   # Discrete count child windows.
        observedChildWindows.append(i)
        # Parent window.
        if count == parentWindowSize:
            # Discrete count of parent windows.
            parentWindowCount += 1
            parentWindowCounts.append(parentWindowCount)
            # Time of parent windows.
            parentWindowTimes.append(round(windows[1][parentWindowCount-1] * parentWindowSize, 3))
            # Total observations in parent windows.
            observedChildWindowsSum = sum(observedChildWindows)
            parentWindowTotalPackets.append(observedChildWindowsSum)
            # Mean and stdev of observation.
            observedChildWindowsMean = statistics.mean(observedChildWindows)
            parentWindowMeans.append(observedChildWindowsMean)
            observedChildWindowsStdevs = statistics.stdev(observedChildWindows)
            parentWindowStdevs.append(observedChildWindowsStdevs)
            # Reset for next parent window.
            observedChildWindows = []
            count = 0
    # Attack packets.
    observedChildWindows = [] # List of child windows, total nr packets for each child.
    parentWindowTotalAttackPackets = []
    count = 0
    for i in windows[4]:
        count += 1   # Discrete count child windows.
        observedChildWindows.append(i)
        # Parent window.
        if count == parentWindowSize:
            parentWindowTotalAttackPackets.append(sum(observedChildWindows))
            # Reset for next parent window.
            observedChildWindows = []
            count = 0
    return parentWindowCounts, parentWindowTimes, parentWindowTotalPackets, parentWindowMeans, parentWindowStdevs, parentWindowTotalAttackPackets


def get_windowsNormal(data, windowSize, maxWindowNr):
    TCount = 0  # The count of bad packets.
    time_c = 1  # The count of discrete windows.
    count = 0   # The count of packets.
    # list element is the order of window, first element in each list is the first window values for each list.
    windowNr = []
    windowTimes = []
    windowOccurence = []
    windowFlags = []
    windowFlagOccurences = []
    for order in data:
        count += 1  # Count of packets.
        packet = data[order]
        timestamp = packet[0]
        # The current window we are observing.
        windowTime = round(time_c * windowSize, 6) # E.g. = 0.01, 0.02, 0.03, ...
        # If observed packet belongs to next window.
        if timestamp >= windowTime:
            # Create the window, and save it.
            windowNr.append(time_c)
            windowTimes.append(windowTime)
            windowOccurence.append(count)
            windowFlags.append(0)
            windowFlagOccurences.append(0)
            count = 0   # Reset count of packets for next window.
            time_c += 1 # Next discrete window.
        if maxWindowNr != 'all' and time_c == maxWindowNr+1:
            break
    return windowNr, windowTimes, windowOccurence, windowFlags, windowFlagOccurences


def get_windows(data, windowSize, maxWindowNr):
    TCount = 0  # The count of bad packets.
    time_c = 1  # The count of discrete windows.
    count = 0   # The count of packets.
    # list element is the order of window, first element in each list is the first window values for each list.
    windowNr = []
    windowTimes = []
    windowOccurence = []
    windowFlags = []
    windowFlagOccurences = []
    for order in data:
        count += 1  # Count of packets.
        packet = data[order]
        timestamp = packet[0]
        flag = packet[3]
        # The current window we are observing.
        windowTime = round(time_c * windowSize, 6) # E.g. = 0.01, 0.02, 0.03, ...
        # Count number of bad packets in the observed window.
        if flag == 'T':
            TCount += 1
        # If observed packet belongs to next window.
        if timestamp >= windowTime:
            # Classify windows containing attack packets.
            if TCount > 0:
                windowFlag = 1
            else:
                windowFlag = 0
            # Create the window, and save it.
            windowNr.append(time_c)
            windowTimes.append(windowTime)
            windowOccurence.append(count)
            windowFlags.append(windowFlag)
            windowFlagOccurences.append(TCount)
            count = 0   # Reset count of packets for next window.
            time_c += 1 # Next discrete window.
            TCount = 0 # Reset count of packets with flag T for next window.
        if maxWindowNr != 'all' and time_c == maxWindowNr+1:
            break
    return windowNr, windowTimes, windowOccurence, windowFlags, windowFlagOccurences


def readInDataNormal(filePath, maxRead):
    # Textfile (dataset without attack is in .txt format).
    data = {}
    count = 0
    with open(filePath, 'r') as textfile:
        for line in textfile:
            line = line.split()
            count += 1
            if count == 1:
                timeStart = float(line[1])  # Used to downscale time to start at 0.
            try:
                data[count] = [float(line[1]) - timeStart, line[3], line[7:]] # ex. {1: [timestamp, ID, dataframe]}
            except IndexError:  # The OTIDS normal data have some packets with empty dataframe.
                data[count] = [float(line[1]) - timeStart, line[3], 'None'] # ex. {1: [timestamp, ID, dataframe]}
            if maxRead != 'all' and count == maxRead:
                break
    return data


def readInData(filePath, maxRead):
    # CSV files.
    data = {}
    with open (filePath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        count = 0 # Nr lines count.
        timeStart = 0
        for line in reader:
            count += 1
            # Used to downscale time to start at 0.
            if count == 1:
                timeStart = float(line[0])
            data[count] = [float(line[0]) - timeStart, line[1], line[3:len(line)-1], line[-1]] # ex. {1: [timestamp, ID, dataframe, flag]}
            if maxRead != 'all' and count == maxRead:
                break
    return data


if __name__ == '__main__':
    main()
