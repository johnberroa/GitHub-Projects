"""
This code plays around with indexes and number control in for loops.  The first allows the index to stay at the same
number for x runs through the for loop.  Probably has an application somewhere...
The second repeats a set of numbers within a range.
"""

########################################################################################################################
def forCycle(x, loopSize):
    cycles = 0
    for i in range(loopSize):
        j = i % x  # counts by x; in the default case, 3
        k = i - j
        k = int(k / x)
        if cycles == (x):
            cycles = 0
        cycler(k, cycles)
        cycles += 1

def cycler(k, cycles):
    '''
    Where the code that gets done in the for loop gets placed.  Need a 'if cycles' for each loop if you want something
    done differently each time.  Currently it is set up for 3.  If this isn't changed to fit the number you input,
    it won't work properly for higher desired numbers, and will be inefficient for smaller desired numbers.
    '''
    if cycles == 0:
        print(k, 'One')
    elif cycles == 1:
        print(k, 'Two')
    elif cycles == 2:
        print(k, 'Three...restart!')
    else:
        print('There is no code to perform during this iteration.')
########################################################################################################################
def repeatSequence(len, seq_max):
    cycle = 0
    for i in range(len):
        if cycle == 0:
            cycle = 1
        elif cycle % seq_max != 0:
            cycle += 1
        else:
            cycle = 1
        print(cycle)
########################################################################################################################

forCycle(3, 100) #count for 100 iterations repeating the index 3 times
repeatSequence(100, 5) #repeat for 100 iterations 1-5