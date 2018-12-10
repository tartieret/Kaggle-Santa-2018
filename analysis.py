"""Analysis of a solution"""

from aco import load_cities, INPUT_FILE, OUTPUT_FILE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def distance(X, Y, primes):
    """Calculate the tour distance - Vectorized implementation

    Args:
        X (numpy.ndarray): X coordinates
        Y (numpy.ndarray): Y coordinates
        primes (numpy.ndarray): 0 or 1

    Returns:
        float: total distance

    """
    distances = np.hypot(X-np.roll(X, shift=-1), Y-np.roll(Y, shift=-1))
    penalties = 0.1*distances[9::10]*(1-primes[9::10])
    print('Nb penalties: {}'.format(len(penalties)))
    print('% penalties: {}'.format(100*len(penalties)/len(distances)))
    return np.sum(distances)+np.sum(penalties)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Analyze a solution')
    parser.add_argument('-f', help='name of the solution file', default=OUTPUT_FILE, action="store")
    args = parser.parse_args()
    filename = args.f

    print('*'*60 + '\nKaggle Sant 2018 - Solution analysis\n' + '*'*60)

    # load all the cities with indices
    ids, X, Y, primes = load_cities(INPUT_FILE)

    # load the solution
    print('Load solution from "{}"'.format(filename))
    submission = pd.read_csv(filename)
    cities = submission.Path
    print('Found {} cities'.format(len(cities)))

    subX = X[cities]
    subY = Y[cities]
    subPrimes = primes[cities]
    score = distance(subX, subY, subPrimes)
    print('Total distance: ', score)

    fig = plt.figure(figsize=(20,20))
    plt.plot(subX, subY, '.-', color='lightblue', alpha=0.6, label='Cities')
    plt.plot(subX[0], subY[0], 'o', color='fuchsia', label='North Pole')
    plt.axis('off')
    plt.legend()
    plt.show()

