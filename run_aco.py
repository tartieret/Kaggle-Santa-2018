import argparse
import time
import os
import logging

from utils import load_cities
from aco import ACO

INPUT_FILE = "aug_cities.csv"
OUTPUT_FOLDER = "output"


# --------------------------------------
# Main
# --------------------------------------


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ACO Solver')
    parser.add_argument('-n', '--name', help='name of experiment', action="store")
    parser.add_argument('-g', '--generations', help='number of generations', action="store", type=int)
    args = parser.parse_args()
    name = args.name
    nb_generations = args.generations

    print('*'*60 + '\nKaggle Sant 2018 - ACO\n' + '*'*60)

    # load the grid
    ids, X, Y, primes = load_cities(INPUT_FILE)
    # for now reduces to 100 cities
    ids = ids[0:100]
    X = X[0:100]
    Y = Y[0:100]
    primes = primes[0:100]

    # initialize the problem
    nb_cities = len(X)
    aco = ACO(
        name=name,
        X=X, Y=Y, primes=primes,
        rho=0.4,
        alpha=1,
        beta=1.5,
        Q=60,
        tau_0=1/nb_cities*37222
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
    submission_folder = os.path.join(OUTPUT_FOLDER, name)
    if not os.path.isdir(submission_folder):
        os.mkdir(submission_folder)
    filename = os.path.join(submission_folder, 'submission.csv')
    print('\nSave best tour as {}'.format(filename))
    with open(filename, 'w') as submission:
        submission.write('Path\n')
        for city in aco.best_tour:
            submission.write('{}\n'.format(city))

    # save the stats
    stats_filename = os.path.join(submission_folder, 'stats.csv')
    with open(stats_filename, 'w') as submission:
        submission.write('Gen,Score,Gain\n')
        for stat in aco.stats:
            submission.write('{},{},{}\n'.format(stat[0], stat[1], stat[2]))
