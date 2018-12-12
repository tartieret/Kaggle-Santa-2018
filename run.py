import argparse
import time
import os
import logging

from utils import load_cities
from aco import ACO

INPUT_FILE = "aug_cities.csv"
OUTPUT_FOLDER = "output"
OUTPUT_FILE = os.path.join(OUTPUT_FOLDER, "submission.csv")


# --------------------------------------
# Main
# --------------------------------------


if __name__ == '__main__':

    print('*'*60 + '\nKaggle Sant 2018 - ACO\n' + '*'*60)

    ids, X, Y, primes = load_cities(INPUT_FILE)

    # for now reduces to 100 cities
    ids = ids[0:50]
    X = X[0:50]
    Y = Y[0:50]
    primes = primes[0:50]

    nb_cities = len(X)
    nb_generations = 200

    # initialize the problem
    aco = ACO(X, Y, primes,
        rho=0.4,
        alpha=1,
        beta=1.5,
        Q=40,
        tau_0=0.0001
    )
    # run the solver
    start = time.time()
    try:
        aco.solve(
            nb_ants=nb_cities,
            nb_generations=nb_generations,
            nb_elites=5,
        )
    except KeyboardInterrupt:
        print('\nKEYBOARD INTERRUPTION')
    finally:
        print('ACO run {} generations'.format(aco.generation))
    end = time.time()

    print('\nBest score: ', aco.best_score)
    print('Time elapsed: {:.2f} sec'.format(end-start))
    print('Average: {:.2f} sec / generation'.format((end-start)/aco.generation))

    # save the best tour to the disk
    filename = os.path.join(OUTPUT_FOLDER, 'submission_{}.csv'.format(int(start)))
    print('\nSave best tour as {}'.format(filename))
    #with open(OUTPUT_FILE, 'w') as submission:
    with open(filename, 'w') as submission:
        submission.write('Path\n')
        for city in aco.best_tour:
            submission.write('{}\n'.format(city))

    # save the stats
    stats_filename = os.path.join(OUTPUT_FOLDER, 'stats_{}.csv'.format(int(start)))
    with open(stats_filename, 'w') as submission:
        submission.write('Gen,Score,Gain\n')
        for stat in aco.stats:
            submission.write('{},{},{}\n'.format(stat[0], stat[1], stat[2]))
