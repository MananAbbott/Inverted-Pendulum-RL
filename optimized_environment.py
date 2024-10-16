
import numpy as np
from numba import njit

class InvertedPendulum:
    def __init__(self, max_torque, g=9.8, l=1.0):
        self.max_torque = max_torque
        self.g = g
        self.l = l
        self.state = [0.0, 0.0]  # [omega, omega_dot]
        self.steps = 0

    @njit
    def step(self, action):
        # Clip action to be within the torque limit
        action = np.clip(action, -self.max_torque, self.max_torque)
        omega, omega_dot = self.state
        omega_dot_dot = ((3 * self.g) / (2 * self.l) * np.sin(omega)) + (3.0 / (self.l ** 2) * action)

        # Update state
        omega_dot = omega_dot + 0.05 * omega_dot_dot  # Assuming a time delta of 0.05
        omega = omega + 0.05 * omega_dot

        self.state = [omega, omega_dot]
        self.steps += 1

        # Check if terminal state (omega goes too far)
        done = abs(omega) > np.pi

        # Reward function
        reward = 1.0 if not done else 0.0

        return np.array(self.state), reward, done
