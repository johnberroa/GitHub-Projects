"""
Plots how fast people can hit a button.  Cool to see where the rate can't be maintained anymore/saturation.
"""
import numpy as np
import matplotlib.pyplot as plt
import time


def trial():
    """
    Records how fast the user presses the enter key
    :return rates; list of rates:
    """
    user = ''
    rates = []
    rates_toHit = 0
    count = 0
    print("\n\nTask starts now")
    while True:
        if count % 10 == 0:
            rates_toHit += 1
            if rates_toHit == 9: # stop automatically after the user advances from 8/second
                break
            print("Press the key at a rate of {} per second".format(rates_toHit))
        count += 1
        inputs = []
        start = time.time()
        while time.time() - start <= .5:
            # records a datapoint every .5 seconds (if this is to be changed, count must be changed)
            before = time.time()
            user = input()
            after = time.time()
            inputs.append(after - before)
        rates.append(np.mean(inputs))
    return rates

def plot_rates(rates):
    """
    Plots the rates as a graph and displays it
    :param rates:
    """
    print("Plotting...")
    x_vals = np.arange(len(rates))
    print(x_vals, rates)
    plt.plot(x_vals, rates, c='m')

    plt.axhline(1, linewidth=1, color='c')
    plt.axhline(1/2, linewidth=1, color='c')
    plt.axhline(1/3, linewidth=1, color='c')
    plt.axhline(1/4, linewidth=1, color='c')
    plt.axhline(1/5, linewidth=1, color='c')
    plt.axhline(1/6, linewidth=1, color='c')
    plt.axhline(1/7, linewidth=1, color='c')
    plt.axhline(1/8, linewidth=1, color='c')
    plt.axhline(.032, linewidth=1, color='r', label="Holding")

    plt.text(75, 1 / 2, "2/s")
    plt.text(75, 1 / 3, "3/s")
    plt.text(75, 1 / 4, "4/s")
    plt.text(75, 1 / 5, "5/s")
    plt.text(75, 1 / 6, "6/s")
    plt.text(75, 1 / 7, "7/s")
    plt.text(75, 1 / 8, "8/s")

    for window in range(len(rates)):
        if window % 10 == 0:
            plt.axvline(window, linewidth=1, color='b')

    plt.legend()
    plt.title("Button Pressing Rate of a Human")
    plt.ylabel("Hits per second")
    plt.xlabel("Onset of new rate")
    plt.xticks(np.arange(0,len(rates), 10), ['Start\n1/s','Start\n2/s',
                                             'Start\n3/s','Start\n4/s',
                                             'Start\n5/s','Start\n6/s',
                                             'Start\n7/s','Start\n8/s'])
    plt.show()


if __name__ == '__main__':
    print("This is a test of button pressing speed")
    print("For the task, the 'enter' button should be pressed")
    print("Try to follow the directions on screen")
    print("The task will take 40 seconds to complete")
    print("\nPlease wait for the task to start...")
    time.sleep(2)
    print("\nNote: Holding down the 'enter' key will not tell you how fast you can hit it ;)")
    time.sleep(4)
    r = trial()
    plot_rates(r)
