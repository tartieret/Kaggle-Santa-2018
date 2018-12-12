"""Analysis of a solution"""

import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from run import INPUT_FILE, OUTPUT_FOLDER
from utils import load_cities


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
    parser.add_argument('-n', '--name', help='name of the solution file', action="store")
    args = parser.parse_args()
    name = args.name

    print('*'*60 + '\nKaggle Sant 2018 - Solution analysis\n' + '*'*60)
    print('    ANALYSE Solution {}\n'.format(name))

    # load all the cities with indices
    ids, X, Y, primes = load_cities(INPUT_FILE)

    # load the solution
    submission_folder = os.path.join(OUTPUT_FOLDER, name)
    submission_file = os.path.join(submission_folder, 'submission.csv')
    print('Load solution from "{}"'.format(submission_file))
    submission = pd.read_csv(submission_file)
    cities = submission.Path
    print('Found {} cities'.format(len(cities)))

    subX = X[cities]
    subY = Y[cities]
    subPrimes = primes[cities]
    score = distance(subX, subY, subPrimes)
    print('Total distance: ', score)

    # Load the stats
    stats = pd.read_csv(os.path.join(submission_folder, 'stats.csv'))

    fig = plt.figure(figsize=(15,15))
    plt.plot(subX, subY, '.-', color='lightblue', alpha=0.6, label='Cities')
    plt.plot(subX[0], subY[0], 'o', color='fuchsia', label='North Pole')
    plt.axis('off')
    plt.legend()

    fig = plt.figure(figsize=(5,5))
    plt.plot(stats.Gen, stats.Score, '.-', color='blue', alpha=0.8, label='Score')
    plt.xlabel('Generations')
    plt.ylabel('Score')
    plt.legend()

    plt.show()



