import numpy as np

class InvertedPendulum:
    def __init__(self):
        self.terminal_episodes = 200
        self.gamma = 1.0
        self.g = 10
        self.m = 1
        self.l = 1
        self.dt = 0.05
        self.max_torque = 2
        self.max_speed = 8
        self.steps = 0
        self.state = None
    
    def reset(self):
        self.steps = 0
        self.omega = np.random.uniform(low=-5/6*np.pi, high=5/6*np.pi)
        self.omega_dot = np.random.uniform(low=-1, high=1)
        self.state = np.array([self.omega, self.omega_dot])
        return self.state

    def step(self, action):
        self.steps += 1
        omega, omega_dot = self.state
        action = min(max(action, -self.max_torque), self.max_torque)
        omega_dot_dot = ((3 * self.g) / (2 * self.l) * np.sin(omega)) + ((3 * action) / (self.m * self.l ** 2))
        new_omega_dot = min(max(omega_dot + (omega_dot_dot * self.dt), -self.max_speed), self.max_speed)
        new_omega = omega + (new_omega_dot * self.dt)
        self.state = np.array([new_omega, new_omega_dot])

        # Calculate reward
        omega_normalized = ((omega + np.pi) % (2 * np.pi)) - np.pi
        reward = -((omega_normalized)**2 + 0.1 * omega_dot**2 + 0.001 * action**2)

        done = self.steps >= self.terminal_episodes
        return self.state, reward, done