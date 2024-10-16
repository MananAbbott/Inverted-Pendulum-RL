
import numpy as np
from numba import njit

class EvolutionStrategy:
    def __init__(self, initial_params, population_size, sigma, learning_rate):
        self.params = initial_params
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate

    @njit
    def get_population(self):
        return np.random.randn(self.population_size, len(self.params)) * self.sigma + self.params

    @njit
    def get_reward(self, solution):
        # Assuming a reward function is defined
        # Placeholder function, replace with actual environment simulation
        return -np.sum(solution ** 2)

    @njit
    def update_params(self, rewards, population):
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        gradient = np.dot(rewards, (population - self.params)) / (self.population_size * self.sigma)
        self.params += self.learning_rate * gradient

    def train(self, iterations):
        for _ in range(iterations):
            population = self.get_population()
            rewards = np.array([self.get_reward(p) for p in population])
            self.update_params(rewards, population)
