"""Simulated Annealing implementation"""

import argparse
import sys
import os
import random
import copy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import logging
import time
import tqdm

from utils import load_cities, distance

INPUT_FILE = "aug_cities.csv"
OUTPUT_FOLDER = "output"

if not os.path.isdir(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

# -----------------------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

# -----------------------------------------------------------------------------------------
# TSP Model
# -----------------------------------------------------------------------------------------

class TSP:
    """Representation of a traveling salesman optimization problem.  The goal
    is to find the shortest path that visits every city in a closed loop path.
    """

    def __init__(self, ids, X, Y, primes):
        self.ids = copy.deepcopy(ids)
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)
        self.primes = copy.deepcopy(primes)
        self.size = self.X.size
        self.grid = np.array([ids, X, Y, primes])

    def copy(self):
        """Return a copy of the current state."""
        new_tsp = TSP(self.ids, self.X, self.Y, self.primes)
        return new_tsp

    def fitness(self):
        """Calculate the total length of the closed-circuit path of the current
        state by summing the distance between every pair of adjacent cities.

        Returns
        -------
        float
            A floating point value with the total cost of the path given by visiting
            the cities in the order according to the self.cities list

        """
        return distance(self.X, self.Y, self.primes)

    def nodes(self):
        return list(zip(self.ids, self.X, self.Y, self.primes))

    def save(self, filename):
        """Save the solution to the disk"""
        with open(filename, 'w') as file:
            file.write("CityId,X,Y,is_prime\n")
            for i in range(0, self.size):
                file.write('{},{},{},{}\n'.format(
                    self.ids[i], self.X[i], self.Y[i], self.primes[i]
                ))

    @staticmethod
    def load_from_file(filename):
        """Load the preprocessed list of cities, that already
        includes the "is_prime" flag

        Args:
            filename (str): name of the file to load

        Returns:
            (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray): cityID, X, Y and prime flag

        """
        cities = pd.read_csv(filename)
        ids = cities.CityId.values.astype(np.int32)
        X = cities.X.values
        Y = cities.Y.values
        primes = cities.is_prime.values.astype(np.int16)
        tsp = TSP(ids, X, Y, primes)
        return tsp

    def plot(self, show=True):
        """Plot a TSP path."""
        fig = plt.figure(figsize=(15, 15))
        plt.plot(np.append(self.X, self.X[0]), np.append(self.Y, self.Y[0]), '.-', color='lightblue', alpha=0.6, label='Cities')
        plt.plot(self.X[0], self.Y[0], 'o', color='fuchsia', label='North Pole')
        plt.axis('off')
        plt.legend()
        if show:
            plt.show()


# -----------------------------------------------------------------------------------------
# Simulated Annealing model
# -----------------------------------------------------------------------------------------


class SA:
    """Simulated Annealing"""

    def __init__(self, name, problem, alpha=0.995, T_min=1e-8, T_start=-1):
        self.name = name
        self.problem = problem

        self.N = self.problem.size
        self.T_start = math.sqrt(problem.size) if T_start == -1 else T_start
        self.T = self.T_start
        self.alpha = alpha
        self.current_solution = problem
        self.current_fitness = problem.fitness()
        self.best_fitness = self.current_fitness
        self.best_solution = self.current_solution.copy()
        self.iteration = 0
        self.T_min = T_min

        self.stats = []
        # create a submission folder for this experiment
        self.submission_folder = os.path.join(OUTPUT_FOLDER, self.name)
        if not os.path.isdir(self.submission_folder):
            os.mkdir(self.submission_folder)

    def euc_dist(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))

    def greedy_solution(self, problem):
        """Generate a greedy solution using the nearest neighbour approach"""
        logger.info('Start building greedy solution...')
        start_t = time.time()
        grid = problem.grid
        curr_i = 0
        new_path = np.zeros(self.N, dtype=np.int32)
        free_ind = set(np.arange(1, self.N, dtype=np.int32))
        for i in tqdm.tqdm(range(1, self.N)):
            x = grid[1, curr_i]
            y = grid[2, curr_i]
            # calculate the distances between current city and all the other cities
            neighbours = list(free_ind)
            dist = np.hypot(grid[1, neighbours]-x, grid[2, neighbours]-y)
            next_i = neighbours[np.argmin(dist)]
            free_ind.remove(next_i)
            new_path[i] = next_i
            curr_i = next_i
        end_t = time.time()

        # format the solution
        ids = problem.ids[new_path]
        X = problem.X[new_path]
        Y = problem.Y[new_path]
        primes = problem.primes[new_path]
        solution = TSP(ids, X, Y, primes)
        fitness = solution.fitness()
        logger.info('Generated greedy solution in {:2f}s with fitness {}'.format(end_t-start_t, fitness))
        return solution, fitness

    def initialize_solution(self, save_as_filename="initial.csv", load_from=None):
        """Generate an initial solution"""
        if load_from:
            # load the initial solution from the disk
            self.current_solution = TSP.load_from_file(load_from)
        else:
            # generate the initial solution
            self.current_solution, self.current_fitness = self.greedy_solution(self.problem)
            if self.current_fitness < self.best_fitness:
                self.best_solution = self.current_solution.copy()
                self.best_fitness = self.current_fitness
                logger.info('Initialization led to a better solution with fitness {}'.format(self.best_fitness))

            # save the solution to the disk
            init_filename = os.path.join(self.submission_folder, save_as_filename)
            logger.info('Save initial solution to {}'.format(init_filename))
            self.current_solution.save(init_filename)


    def schedule(self, T):
        return self.alpha*T

    def swap(self, arr, i, k):
        return np.concatenate((arr[0:i], arr[k:-self.N+i-1:-1], arr[k+1:self.N]))

    def two_opt(self, problem, fitness):
        """2-opt Algorithm"""
        improvement_threshold = 0.01
        ids = problem.ids
        X = problem.X
        Y = problem.Y
        primes = problem.primes
        best_fitness = fitness
        improvement_factor = 1
        while improvement_factor > improvement_threshold:
            # Record the distance at the beginning of the loo
            distance_to_beat = best_fitness
            # From each city except the first and last,
            for swap_first in range(1, self.N-2):
                # to each of the cities following,
                for swap_last in range(swap_first+1, self.N):
                    # 2-opt swap
                    new_X = self.swap(X, swap_first, swap_last)
                    new_Y = self.swap(Y, swap_first, swap_last)
                    new_primes = self.swap(primes, swap_first, swap_last)
                    # check the total distance with this modification.
                    new_fitness = distance(new_X, new_Y, new_primes)
                    # if the new route is better save it
                    if new_fitness < best_fitness:
                        ids = self.swap(ids, swap_first, swap_last)
                        X = new_X
                        Y = new_Y
                        primes = new_primes
                        best_fitness = new_fitness
            # Calculate how much the route has improved.
            improvement_factor = 1 - best_fitness/distance_to_beat

        new_solution = TSP(ids, X, Y, primes)
        return new_solution, new_fitness

    def get_successors(self):
        """Build the list of potential successors from the current solution"""
        # i = random.randint(1, self.N - 2)
        # l = 1#random.randint(1, self.N - l)
        # candidate = self.current_solution.copy()
        # candidate.ids[i : (i + l)] = np.flip(candidate.ids[i : (i + l)])
        # candidate.X[i : (i + l)] = np.flip(candidate.X[i : (i + l)])
        # candidate.Y[i : (i + l)] = np.flip(candidate.Y[i : (i + l)])
        # candidate.primes[i : (i + l)] = np.flip(candidate.primes[i : (i + l)])
        candidate, fitness = self.two_opt(self.current_solution, self.current_fitness)
        return [candidate]#, self.current_solution]

    def select_successor(self):
        """Randomly select a successor for the current solution"""
        successors = self.get_successors()
        return random.choice(successors)

    def transition_prob(self, candidate_fitness):
        """Probability of accepting the new candidate as the next solution
        even if its fitness is higher than the current solution"""
        return math.exp(-abs(candidate_fitness - self.current_fitness) / self.T)

    def solve(self, nb_iterations=1000):
        """Run simulated annealing"""
        logger.info('Start Simulated Annealing - T={T} - current_fitness={fit}'.format(
            T=self.T, fit=self.current_fitness
        ))
        start = time.time()
        while self.iteration < nb_iterations and self.T > self.T_min:
            # calculate the current fitness
            self.current_fitness = self.current_solution.fitness()
            # select the next potential solution
            successor = self.select_successor()
            successor_fitness = successor.fitness()

            delta_e = self.current_fitness - successor_fitness
            if delta_e > 0:
                self.current_solution = successor
                self.current_fitness = successor_fitness
                if self.current_fitness < self.best_fitness:
                    improvment = (self.best_fitness - self.current_fitness)/self.best_fitness
                    self.best_solution = self.current_solution
                    self.best_fitness = self.current_fitness

                    logger.info('{}/{} - Best fitness: {}, gain: {:.2f}%'.format(
                        self.iteration,
                        nb_iterations,
                        self.best_fitness,
                        improvment*100
                    ))

                    # save to the disk
                    self.save_solution()

            else:
                if random.random() < self.transition_prob(successor_fitness):
                    self.current_solution = successor
                    self.current_fitness = successor_fitness

            # cool down the temperature
            self.T = self.schedule(self.T)

            # record the stats
            self.stats.append((self.iteration, self.best_fitness, self.T))

            self.iteration += 1

            if self.iteration % 10 == 0:
                # estimate the remaining time
                now = time.time()
                est = 1/self.iteration*(now-start)*(math.log(self.T_min/self.T_start)/math.log(self.alpha)-1)
                logger.info("{}/{} - T={} - Est. time left: {:.2f}s".format(
                    self.iteration, nb_iterations, self.T, est
                ))

        logger.info('End Simulated Annealing - Iter={it}, T={T} - current_fitness={fit}'.format(
            it=self.iteration, T=self.T, fit=self.current_fitness
        ))
        logger.info('Best fitness={fit}'.format(
            fit=self.best_fitness
        ))
        logger.info('Total elapsed time: {:.2f}s'.format(time.time()-start))

    def save_solution(self):
        """save the best tour to the disk"""
        filename = os.path.join(self.submission_folder, 'best_tour.csv')
        self.best_solution.save(filename)

    def build_submission(self):
        """Generate a submission file for Kaggle"""
        filename = os.path.join(self.submission_folder, 'submission.csv')
        logger.info('Save best tour as {}'.format(filename))
        with open(filename, 'w') as submission:
            submission.write('Path\n')
            for city in self.best_solution.ids:
                submission.write('{}\n'.format(city))

    def save_stats(self):
        """Save the convergence stats to the disk"""
        stats_filename = os.path.join(self.submission_folder, 'stats.csv')
        with open(stats_filename, 'w') as submission:
            submission.write('Iter,Score,Temp\n')
            for stat in self.stats:
                submission.write('{},{},{}\n'.format(stat[0], stat[1], stat[2]))

    def save_all(self):
        self.save_solution()
        self.save_stats()

    def plot_stats(self, show=True):
        fig = plt.figure(figsize=(5, 5))
        plt.plot([x[0] for x in self.stats], [x[1] for x in self.stats], '.-', color='blue', alpha=0.8, label='Score')
        plt.plot([x[0] for x in self.stats], [x[2] for x in self.stats], '.-', color='green', alpha=0.8, label='Temp')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.legend()
        if show:
            plt.show()

    def plot_solution(self, show=True):
        self.best_solution.plot(show=show)



# --------------------------------------
# Main
# --------------------------------------


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simulated Annealing Solver')
    parser.add_argument('-n', '--name', help='name of experiment', action="store", default='debug'),
    parser.add_argument('-s', '--steps', help='number of steps', action="store", type=int)
    args = parser.parse_args()
    name = args.name
    nb_steps = args.steps

    print('*'*60 + '\nKaggle Sant 2018 - Simulated Annealing\n' + '*'*60)

    # load the grid
    tsp = TSP.load_from_file('aug_cities.csv')

    # run simulated annealing optimization
    sa = SA(name=name, problem=tsp, T_start=5000, alpha=0.995)
    sa.initialize_solution()
    try:
        sa.solve(nb_iterations=100)
    finally:
        sa.save_stats()
        sa.save_solution()
        sa.build_submission()

    # plot the results
    sa.plot_stats(show=False)
    sa.plot_solution()
