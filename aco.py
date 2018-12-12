"""Kaggle Sant 2018 - Ant Colony
"""

import numpy as np
import pandas as pd
import tqdm
import math
import pickle
import os

RUN_FOLDER = "runs"

# --------------------------------------
# ACO
# --------------------------------------


class Ant:

    def __init__(self, nb_cities, start_id, start_x, start_y, start_is_prime):
        # the total tour contains one more city as we go back to the start
        self.tour = np.zeros(nb_cities+1, dtype=np.int32)
        self.visited = np.zeros(nb_cities, dtype=np.int8)
        # initialize the ant
        self.total_dist = 0
        self.current_step = 0
        # record the location of the start as we want to get back to
        # it at the end of the tour
        self.start_id = start_id
        self.start_x = start_x
        self.start_y = start_y
        # set the current node as the start
        self.current_x = start_x
        self.current_y = start_y
        self.is_prime = start_is_prime
        self.visited[start_id] = 1
        self.tour[0] = start_id

    def get_current_city(self):
        return self.tour[self.current_step]

    def get_visited_cities(self):
        """Get the list of visited city indices"""
        return np.nonzero(self.visited)[0].astype(np.int32)

    def get_unvisited_cities(self):
        """Get the list of remaining city indices"""
        return np.where(self.visited == 0)[0].astype(np.int32)

    def move_to(self, city_id, next_x, next_y, next_is_prime):
        # calculate the distance between current city and the next one
        dist = math.sqrt((self.current_x-next_x)**2+(self.current_y-next_y)**2)
        if self.current_step % 10 == 0:
            if not self.is_prime:
                dist = 1.1*dist
        self.total_dist += dist
        # register the new city as the current one
        self.visited[city_id] = 1
        self.current_step += 1
        self.tour[self.current_step] = city_id
        self.current_x = next_x
        self.current_y = next_y
        self.is_prime = next_is_prime

    def score(self):
        return self.total_dist

    def get_tour(self):
        return self.tour

    def get_total_distance(self):
        return self.total_dist


class ACO:

    def __init__(self, X, Y, primes, alpha=1, beta=5, Q=20, rho=0.9, tau_0=0.000001):
        self.X = X
        self.Y = Y
        self.primes = primes
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.rho = rho
        self.tau_0 = tau_0
        self.nb_cities = len(X)

    def init_pheronomes(self):
        self.tau = self.tau_0*np.ones((self.nb_cities, self.nb_cities), dtype=np.float)

    def evaporate_pheronomes(self):
        self.tau = (1-self.rho)*self.tau

    def add_pheronomes(self, ants):
        for ant in ants:
            # calculate the weighted amount of pheronomes
            L = ant.get_total_distance()
            dep = self.Q/L
            tour = ant.get_tour()
            # update the pheronomes
            self.tau[tour[0:-1], tour[1:]] += dep
            self.tau[tour[1:], tour[0:-1]] += dep

    def select_next_city(self, current_city_id, unvisited, current_step):
        """Select the next city based on the distances and the amount
        of pheronomes on the paths between cities

        Args:
            current_city_id (int): where to start from
            unvisited (numpy.ndarray): list of non visited cities

        Returns:
            int: id of the next city
        """
        i = current_city_id
        x = self.X[i]
        y = self.Y[i]
        # check if the penalty should be applied
        coeff = 1
        if current_step % 10 == 0:
            if not self.primes[i]:
                coeff = 1.1
        # calculate the transition probabilities
        dist = coeff*np.hypot(self.X[unvisited]-x, self.Y[unvisited]-y)
        pheronomes = self.tau[i][unvisited]
        denominator = np.sum(np.power(pheronomes, self.alpha)/np.power(dist, self.beta))
        transition_prob = 1/denominator*np.power(pheronomes, self.alpha)/np.power(dist, self.beta)
        # select the next city by random roulette
        # and mark the city as visited
        next_city = np.random.choice(unvisited, p=transition_prob)
        return next_city

    def select_elites(self, ants, nb_elites=2):
        scores = np.array([ant.score() for ant in ants])
        elites = []
        for i in range(0, nb_elites):
            elite_id = np.argmin(scores)
            elites.append(ants[elite_id])
            scores[elite_id] = float('inf')
        return elites

    def solve(self, nb_ants=1, nb_generations=10, nb_elites=4):
        self.best_score = float('inf')
        self.best_tour = []
        # initialize the pheronomes
        self.init_pheronomes()
        # initialize a first generation of ants
        ants = [Ant(
                    nb_cities=self.nb_cities,
                    start_id=0,
                    start_x=self.X[0],
                    start_y=self.Y[0],
                    start_is_prime=self.primes[0]
                ) for k in range(0, nb_ants)]
        elites = []

        self.generation = 0
        self.stats = []
        while self.generation < nb_generations:
            print('Generation {}'.format(self.generation))
            # generate a new generation of ants
            # we also keep the elites from the previous generation
            ants = [Ant(
                nb_cities=self.nb_cities,
                start_id=0,
                start_x=self.X[0],
                start_y=self.Y[0],
                start_is_prime=self.primes[0]
            ) for k in range(0, nb_ants-len(elites))]

            for step in tqdm.tqdm(range(0, self.nb_cities-1)):
                # move each ant by one step
                for ant in ants:
                    current_city_id = ant.get_current_city()
                    # get the list of non visited cities
                    unvisited = ant.get_unvisited_cities()
                    # select the next city
                    next_city = self.select_next_city(
                        current_city_id=current_city_id,
                        unvisited=unvisited,
                        current_step=ant.current_step
                    )
                    # move the ant
                    ant.move_to(
                        city_id=next_city,
                        next_x=self.X[next_city],
                        next_y=self.Y[next_city],
                        next_is_prime=self.primes[next_city]
                    )

            # move the ants back to the start
            for ant in ants:
                ant.move_to(
                        city_id=ant.start_id,
                        next_x=ant.start_x,
                        next_y=ant.start_y,
                        next_is_prime=0
                    )
            # calculate the ant scores
            current_score = ant.score()
            if current_score < self.best_score:
                improvment = (self.best_score - current_score)/self.best_score
                self.best_score = current_score
                self.best_tour = ant.get_tour()
                print('GEN {} - Best score: {}, gain: {:.2f}%'.format(
                    self.generation, self.best_score, improvment*100
                ))

                # save the solution
                pickle.dump(self.best_tour, open(os.path.join(RUN_FOLDER, 'best_tour_{}.dat'.format(self.generation)), 'wb'))
                # save the pheronomes
                pickle.dump(self.tau, open(os.path.join(RUN_FOLDER, 'tau_{}.dat'.format(self.generation)), 'wb'))

            # record the stats
            self.stats.append((self.generation, self.best_score, improvment))

            # update pheronomes
            self.add_pheronomes(ants+elites)
            self.evaporate_pheronomes()

            # print some stats:
            print('Pheronomes - min: {}, max: {}, avg: {}'.format(
                np.min(self.tau), np.max(self.tau), np.mean(self.tau)
            ))

            # select the new elites from this generation
            elites = self.select_elites(ants=ants+elites, nb_elites=nb_elites)

            self.generation += 1

        return self.best_tour, self.best_score, self.stats
