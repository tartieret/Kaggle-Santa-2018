"""Simulated Annealing implementation"""

import argparse
import sys
import os
import random
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import logging
import time
import tqdm

from utils import load_cities, distance

INPUT_FILE = "aug_cities.csv"
OUTPUT_FOLDER = "output"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

class TSP:
    """Representation of a traveling salesman optimization problem.  The goal
    is to find the shortest path that visits every city in a closed loop path.

    Parameters
    ----------
    cities : list
        A list of cities specified by a tuple containing the id and the x, y
        location of the city on a grid. e.g., (153, (585.6, 376.8))

    Attributes
    ----------
    names
    coords
    path : list
        The current path between cities as specified by the order of the city
        tuples in the list.
    """
    def __init__(self, ids, X, Y, primes):
        self.ids = copy.deepcopy(ids)
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)
        self.primes = copy.deepcopy(primes)
        self.size = self.X.size

    def copy(self):
        """Return a copy of the current state."""
        new_tsp = TSP(self.ids, self.X, self.Y, self.primes)
        return new_tsp

    def fitness(self):
        """Calculate the total length of the closed-circuit path of the current
        state by summing the distance between every pair of adjacent cities.  Since
        the default simulated annealing algorithm seeks to maximize the objective
        function, return -1x the path length. (Multiplying by -1 makes the smallest
        path the smallest negative number, which is the maximum value.)

        Returns
        -------
        float
            A floating point value with the total cost of the path given by visiting
            the cities in the order according to the self.cities list

        """
        return distance(self.X, self.Y, self.primes)

    def plot(self, show=True):
        """Plot a TSP path."""
        fig = plt.figure(figsize=(15,15))
        plt.plot(np.append(self.X, self.X[0]), np.append(self.Y, self.Y[0]), '.-', color='lightblue', alpha=0.6, label='Cities')
        plt.plot(self.X[0], self.Y[0], 'o', color='fuchsia', label='North Pole')
        plt.axis('off')
        plt.legend()
        if show:
            plt.show()

    def nodes(self):
        return list(zip(self.ids, self.X, self.Y, self.primes))


class SA:
    """Simulated Annealing"""

    def __init__(self, name, problem, alpha=0.995, stopping_T=1e-8, startingT=-1):
        self.name = name
        self.problem = problem

        self.N = self.problem.size
        self.T = math.sqrt(problem.size) if startingT == -1 else startingT
        self.alpha = alpha
        self.current_solution = problem
        self.current_fitness = problem.fitness()
        self.best_fitness = self.current_fitness
        self.best_solution = self.current_solution.copy()
        logger.info('Problem has fitness {}'.format(self.best_fitness))

        self.iteration = 0
        self.stopping_T = stopping_T

        self.stats = []
        # create a submission folder for this experiment
        self.submission_folder = os.path.join(OUTPUT_FOLDER, self.name)
        if not os.path.isdir(self.submission_folder):
            os.mkdir(self.submission_folder)

    def euc_dist(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))

    def greedy_solution(self):
        """Generate a greedy solution using the nearest neighbour approach"""
        start_t = time.time()
        nodes = self.problem.nodes()
        cur_node = nodes[0]
        path = [cur_node]
        free_nodes = set(nodes)
        free_nodes.remove(cur_node)
        for i in tqdm.tqdm(range(1, self.N)):
        #while free_nodes:
            # find the nearest neighbour
            next_node = min(free_nodes, key=lambda x: self.euc_dist(cur_node[1], cur_node[2], x[1], x[2]))
            free_nodes.remove(next_node)
            path.append(next_node)
            cur_node = next_node
        end_t = time.time()

        # format the solution
        ids, X, Y, primes = zip(*path)
        ids = np.array(ids)
        X = np.array(X)
        Y = np.array(Y)
        primes = np.array(primes)
        solution = TSP(ids, X, Y, primes)
        fitness = solution.fitness()
        logger.info('Generated greedy solution in {:2f}s with fitness {}'.format(end_t-start_t, fitness))
        return solution, fitness

    def initialize_solution(self):
        logger.info('Build initial solution')
        self.current_solution, self.current_fitness = self.greedy_solution()
        if self.current_fitness < self.best_fitness:
            self.best_solution = self.current_solution.copy()
            self.best_fitness = self.current_fitness
            logger.info('Initialization led to a better solution with fitness {}'.format(self.best_fitness))

    def schedule(self, T):
        return self.alpha*T

    def swap(self, arr, i, k):
        return np.concatenate((arr[0:i],arr[k:-self.N+i-1:-1],arr[k+1:self.N]))

    def two_opt(self):
        """2-opt Algorithm"""
        improvement_threshold = 0.01
        ids = self.current_solution.ids
        X = self.current_solution.X
        Y = self.current_solution.Y
        primes = self.current_solution.primes
        best_distance = self.current_fitness
        improvement_factor = 1 # Initialize the improvement factor.
        while improvement_factor > improvement_threshold: # If the route is still improving, keep going!
            distance_to_beat = best_distance # Record the distance at the beginning of the loop.
            for swap_first in range(1,self.N-2): # From each city except the first and last,
                for swap_last in range(swap_first+1, self.N): # to each of the cities following,
                    new_X = self.swap(X, swap_first, swap_last)
                    new_Y = self.swap(Y, swap_first, swap_last)
                    new_primes = self.swap(primes, swap_first, swap_last)
                    new_distance = distance(new_X, new_Y, new_primes) # and check the total distance with this modification.
                    if new_distance < self.current_fitness: # If the path distance is an improvement,
                        ids = self.swap(ids, swap_first, swap_last)
                        X = new_X
                        Y = new_Y
                        primes = new_primes
                        best_distance = new_distance # and update the distance corresponding to this route.
            improvement_factor = 1 - best_distance/distance_to_beat # Calculate how much the route has improved.

        new_solution = TSP(ids, X, Y, primes)
        return new_solution # When the route is no longer improving substantially, stop searching and return the route.

    def get_successors(self):
        """Build the list of potential successors from the current solution"""
        # i = random.randint(1, self.N - 2)
        # l = 1#random.randint(1, self.N - l)
        # candidate = self.current_solution.copy()
        # candidate.ids[i : (i + l)] = np.flip(candidate.ids[i : (i + l)])
        # candidate.X[i : (i + l)] = np.flip(candidate.X[i : (i + l)])
        # candidate.Y[i : (i + l)] = np.flip(candidate.Y[i : (i + l)])
        # candidate.primes[i : (i + l)] = np.flip(candidate.primes[i : (i + l)])
        candidate =  self.two_opt()
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
        for self.iteration in tqdm.tqdm(range(1, nb_iterations+1)):
            if self.T < self.stopping_T:
                break
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

                    logger.info('{} - Best fitness: {}, gain: {:.2f}%'.format(
                        self.iteration, self.best_fitness, improvment*100
                    ))
            else:
                if random.random() < self.transition_prob(successor_fitness):
                    self.current_solution = successor
                    self.current_fitness = successor_fitness

            # cool down the temperature
            self.T = self.schedule(self.T)

            # record the stats
            self.stats.append((self.iteration, self.best_fitness, self.T))

        logger.info('End Simulated Annealing - Iter={it}, T={T} - current_fitness={fit}'.format(
            it=self.iteration, T=self.T, fit=self.current_fitness
        ))
        logger.info('Best fitness={fit}'.format(
            fit=self.best_fitness
        ))

    def save_solution(self):
        """save the best tour to the disk"""
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
        fig = plt.figure(figsize=(5,5))
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
    parser.add_argument('-n', '--name', help='name of experiment', action="store")
    parser.add_argument('-s', '--steps', help='number of steps', action="store", type=int)
    args = parser.parse_args()
    name = args.name
    nb_steps = args.steps

    print('*'*60 + '\nKaggle Sant 2018 - Simulated Annealing\n' + '*'*60)

    # load the grid
    ids, X, Y, primes = load_cities(INPUT_FILE)
    # for now reduces to 100 cities
    # ids = ids[0:50]
    # X = X[0:50]
    # Y = Y[0:50]
    # primes = primes[0:50]

    # define a TSP instance
    tsp = TSP(ids, X, Y, primes)

    # run simulated annealing optimization
    sa = SA(name=name, problem=tsp, startingT=5000, alpha=0.995)
    sa.initialize_solution()
    sa.solve(nb_iterations=1)

    # plot the results
    sa.plot_stats(show=False)
    sa.plot_solution()

