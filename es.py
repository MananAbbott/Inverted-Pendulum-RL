import numpy as np
from environment import InvertedPendulum
from utils import NeuralNetwork
from joblib import Parallel, delayed



def estimate_J(theta, N=10, num_layers=1, neurons_per_layer=None):
    policy = NeuralNetwork(num_layers=num_layers, neurons_per_layer=neurons_per_layer)
    policy.load_weights(theta)
    total_return = 0

    for _ in range(N):
        environment = InvertedPendulum()
        state = environment.reset()
        episode_reward = 0
        done = False
        while not done:
            action = policy.get_action(state)
            state, reward, done = environment.step(action)
            episode_reward += reward
        total_return += episode_reward
    return total_return / N

def evolution_strategy(environment, num_layers = 1, neurons_per_layer= None, sigma=0.1, alpha=0.1, nPerturbation=100, max_iters=100, exp_repeat=1):
    best_theta = None
    best_reward = -np.inf
    all_returns = []
    for _ in range(exp_repeat):
        # reset the environment and theta for each experiment
        policy = NeuralNetwork(num_layers=num_layers, neurons_per_layer=neurons_per_layer)
        theta = policy.get_weights()
        returns = []
        for i in range(max_iters):
            epsilon = np.random.randn(nPerturbation,theta.size)# J_curr = estimate_J(environment, policy, theta)
            J_curr = estimate_J(theta, num_layers=num_layers, neurons_per_layer=neurons_per_layer)
            returns.append(J_curr)
            J_i = []
            # create perturbations
            theta_perturbations = theta + sigma * epsilon
            # for j in range(nPerturbation):
            #     J_i.append(estimate_J(environment, policy, theta_perturbations[j]))
            J_i = Parallel(n_jobs=-1)(delayed(estimate_J)(theta_perturbations[j], num_layers=num_layers, neurons_per_layer=neurons_per_layer) for j in range(nPerturbation))
            J_i = np.array(J_i)
            J_diff = (J_i - J_curr).reshape(nPerturbation,1)
            print(epsilon.shape, J_diff.shape)
            theta = theta + alpha * (1/(nPerturbation*sigma)) * np.dot(epsilon.T, J_diff).flatten()
            print(f'The estimated return at iteration {i} is {J_curr}')
            if J_curr > best_reward:
                best_reward = J_curr
                best_theta = theta
        returns = np.array(returns).reshape(-1,1)
        all_returns.append(returns)
    policy.load_weights(best_theta)
    return policy, np.array(all_returns), best_reward

