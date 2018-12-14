import numpy as np
import pandas as pd
import math

# --------------------------------------
# Utility functions
# --------------------------------------

def load_cities(filename):
    """Load the preprocessed list of cities, that already
    includes the "prime" flag

    Args:
        filename (str): name of the file to load

    Returns:
        (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray): cityID, X, Y and prime flag

    """
    cities = pd.read_csv(filename)
    ids = cities.CityId.values
    X = cities.X.values
    Y = cities.Y.values
    primes = cities.is_prime.values
    print('Load file {} - Found {} cities'.format(
        filename, len(X)
    ))
    return ids, X, Y, primes


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
    return np.sum(distances)+np.sum(penalties)


def build_distance_matrix(X, Y):
    nb_cities = len(X)
    distances = np.zeros((nb_cities, nb_cities))
    for i in range(0, nb_cities):
        for j in range(0, nb_cities):
            distances[i][j] = math.sqrt((X[i]-X[j])**2+(Y[i]-Y[j])**2)
    return distances