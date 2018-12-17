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
import numba
from numba import jit

from utils import load_cities

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
# Efficient numba functions
# -----------------------------------------------------------------------------------------

ID_COL = 0
X_COL = 1
Y_COL = 2
PRIME_COL = 3

@jit(nopython=True)
def roll(a, shift):
    n = a.size
    reshape = True
    if n == 0:
        return a
    shift %= n
    indexes = np.concatenate((np.arange(n - shift, n), np.arange(n - shift)))
    res = a.take(indexes)
    if reshape:
        res = res.reshape(a.shape)
    return res

@jit(nopython=True)
def distance(grid):
    distances = np.hypot(
        grid[X_COL,:]-roll(grid[X_COL,:], shift=-1),
        grid[Y_COL,:]-roll(grid[Y_COL,:], shift=-1)
    )
    penalties = 0.1*distances[9::10]*(1-grid[PRIME_COL,:][9::10])
    return np.sum(distances)+np.sum(penalties)

@jit(nopython=True)
def euc_dist(x1, y1, x2, y2):
    return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))

@jit(nopython=True)
def swap(grid, i, k, N):
    return np.concatenate((grid[:, 0:i], grid[:, k:-N+i-1:-1], grid[:, k+1:N]), axis=1)

@jit(nopython=True)
def two_opt(grid, fitness, N):
    """2-opt Algorithm"""
    best_fitness = fitness
    # Record the distance at the beginning of the loop
    distance_to_beat = best_fitness
    # From each city except the first and last,
    for i in range(1, N-2):
        # to each of the cities following,
        for k in range(i+1, N):
            # 2-opt swap
            new_grid = swap(grid, i, k, N)
            # check the total distance with this modification.
            # we have to recaculate the total distance as the order of cities change
            # and therefore the penalties too
            new_fitness = distance(new_grid)
            # if the new route is better save it
            if new_fitness < best_fitness:
                grid = new_grid
                best_fitness = new_fitness
    # Calculate how much the route has improved.
    improvement_factor = 1 - best_fitness/distance_to_beat
    return grid, best_fitness

# -----------------------------------------------------------------------------------------
# TSP Model
# -----------------------------------------------------------------------------------------



class TSP:
    """Representation of a traveling salesman optimization problem.  The goal
    is to find the shortest path that visits every city in a closed loop path.
    """

    def __init__(self, ids=None, X=None, Y=None, primes=None, grid=None):
        """Build a TSP instance by passing either a matrix or the 4 columns
        separately"""
        if grid is not None:
            self.grid = copy.deepcopy(grid)
        else:
            self.grid = np.array([
                copy.deepcopy(ids),
                copy.deepcopy(X),
                copy.deepcopy(Y),
                copy.deepcopy(primes)
            ])
        self.size = self.grid.shape[1]

    def copy(self):
        """Return a copy of the current state."""
        new_tsp = TSP(grid=self.grid)
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
        return distance(self.grid)

    def save(self, filename):
        """Save the solution to the disk"""
        with open(filename, 'w') as file:
            file.write("CityId,X,Y,is_prime\n")
            for i in range(0, self.size):
                file.write('{},{},{},{}\n'.format(
                    self.grid[ID_COL, i],
                    self.grid[X_COL, i],
                    self.grid[Y_COL, i],
                    self.grid[PRIME_COL, i]
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

    def getX(self):
        return self.grid[X_COL, :]

    def getY(self):
        return self.grid[Y_COL, :]

    def getCityId(self):
        return self.grid[ID_COL, :]

    def plot(self, show=True):
        """Plot a TSP path."""
        fig = plt.figure(figsize=(15, 15))
        X = self.getX()
        X = np.append(X, X[0])
        Y = self.getY()
        Y = np.append(Y, Y[0])
        plt.plot(X, Y, '.-', color='lightblue', alpha=0.6, label='Cities')
        plt.plot(X[0], Y[0], 'o', color='fuchsia', label='North Pole')
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
        solution = TSP(grid=problem.grid[:, new_path])
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

    def swap(self, grid, i, k):
        return np.concatenate((grid[:, 0:i], grid[:, k:-self.N+i-1:-1], grid[:, k+1:self.N]), axis=1)

    def two_opt(self, problem, fitness):
        """2-opt Algorithm"""
        grid = problem.grid
        new_grid, new_fitness = two_opt(grid, fitness, self.N)
        new_solution = TSP(grid=new_grid)
        return new_solution, new_fitness

    def random_swap(self, problem):
        """swap the order of two consecutive cities"""
        i = random.randint(1, self.N - 3)
        l = random.randint(1, 3)
        candidate = problem.copy()
        candidate.grid[:, i:i+l] = np.flip(candidate.grid[:, i:i+l], axis=1)
        return candidate

    def move_prime(self, problem):
        """Find prime cities that lead to penalties and move them"""
        # select all the "prime" cities that are not located at "10th"steps
        prime_indices = np.where(problem.grid[PRIME_COL, :]==1)[0]
        prime_indices = prime_indices[prime_indices % 10 !=0]
        # randomly pick one and move it to the next 10th position
        candidate = problem.copy()
        loc = np.random.choice(prime_indices)
        next_loc = loc+10-loc%10
        if next_loc < self.N:
            candidate.grid = swap(candidate.grid, loc, next_loc, self.N)
        return candidate

    def get_successors(self):
        """Build the list of potential successors from the current solution"""
        # random swap
        candidate = self.random_swap(self.current_solution)
        # move a prime city
        candidate = self.move_prime(candidate)
        # apply 2-opt
        fitness = candidate.fitness()
        candidate, fitness = self.two_opt(candidate, fitness)
        return [candidate]

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
        elapsed_time = time.time()-start
        logger.info('Total elapsed time: {:.2f} s'.format(elapsed_time))
        logger.info('Average time / iteration: {:.3f} s'.format(elapsed_time/nb_iterations))

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
            for city in self.best_solution.getCityId():
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
    parser.add_argument('-i', '--iterations', help='number of iterations', action="store", type=int, default=100)
    args = parser.parse_args()
    name = args.name
    nb_iterations = args.iterations

    print('*'*60 + '\nKaggle Sant 2018 - Simulated Annealing\n' + '*'*60)

    # load the grid
    tsp = TSP.load_from_file('aug_cities200.csv')

    # run simulated annealing optimization
    sa = SA(name=name, problem=tsp, T_start=5000, alpha=0.99998)
    sa.initialize_solution()
    try:
        sa.solve(nb_iterations=nb_iterations)
    finally:
        sa.save_stats()
        sa.save_solution()
        sa.build_submission()

    # plot the results
    sa.plot_stats(show=False)
    sa.plot_solution()
