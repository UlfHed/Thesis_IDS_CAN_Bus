# Illustration of the three alarm scenarios for the SCH case; early detection, perfect detection, and late detection.

import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create data.
    CP = 101
    alarmEarly = 50
    alarmLate = 150

    maxObservations = 200
    nrSections = 2
    sectionSize = int(maxObservations/nrSections)
    normal = True   # Start with in control data.
    yObservation = []
    xObservation = []
    mu = -5 # out control data.
    sigma = 1
    count = 0
    for i in range(nrSections):
        for i in range(sectionSize):    # Each section.
            count += 1
            xObservation.append(count)
            if normal == True:  # Create in control data.
                yObservation.append(np.random.normal(loc=0, scale=1, size=1))
            else:   # Create out control data.
                yObservation.append(np.random.normal(loc=mu, scale=sigma, size=1))
        if normal == True:  # Set up for next section data.
            normal = False
        else:
            normal = True

    # Plot (early detection).
    plt.title('Illustration of SCH, Early Detection')
    plt.plot(xObservation, yObservation, label='Observations')
    plt.plot(xObservation, yObservation, ls="", marker="o", markersize=2, color='darkblue')
    plt.axvline(CP, label='Changepoint (t = ' + str(CP) + ')', color='r', linestyle='dashed')
    plt.axvline(alarmEarly, label='Alarm, ED (t = ' + str(alarmEarly) + ')', color='r')
    plt.xlabel('Observed window (t)')
    plt.ylabel('Observation ($X_t$)')
    plt.legend()
    plt.savefig('Illustration_SCH_Early_Detection.png')
    plt.clf()   # Clear for next plot.

    # plot (perfect detection).
    plt.title('Illustration of SCH, Perfect Detection')
    plt.plot(xObservation, yObservation, label='Observations')
    plt.plot(xObservation, yObservation, ls="", marker="o", markersize=2, color='darkblue')
    plt.axvline(CP, label='Changepoint (t = ' + str(CP) + ')', color='r', linestyle='dashed')
    plt.axvline(CP, label='Alarm, PD (t = ' + str(CP) + ')', color='r')
    plt.xlabel('Observed window (t)')
    plt.ylabel('Observation ($X_t$)')
    plt.legend()
    plt.savefig('Illustration_SCH_Perfect_Detection.png')
    plt.clf()   # Clear for next plot.

    # Plot (late detection).
    plt.title('Illustration of SCH, Late Detection')
    plt.plot(xObservation, yObservation, label='Observations')
    plt.plot(xObservation, yObservation, ls="", marker="o", markersize=2, color='darkblue')
    plt.axvline(CP, label='Changepoint (t = ' + str(CP) + ')', color='r', linestyle='dashed')
    plt.axvline(alarmLate, label='Alarm, LD (t = ' + str(alarmLate) + ')', color='r')
    plt.xlabel('Observed window (t)')
    plt.ylabel('Observation ($X_t$)')
    plt.legend()
    plt.savefig('Illustration_SCH_Late_Detection.png')
    plt.clf()   # Clear for next plot.


if __name__ == '__main__':
    main()
