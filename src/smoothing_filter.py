import numpy as np

class SmoothingFilter:
    def __init__(self, config):
        self.config = config
        self.prev_value = None
        self.prev_derivative = 0

    def filter(self, value, dt):
        dt = max(float(dt), 1e-4)
        if self.prev_value is None:
            self.prev_value = value
            return value
        
        # One Euro Filter implementation
        alpha = self._alpha(self.config['smoothing']['min_cutoff'], dt)
        value_hat = alpha * value + (1 - alpha) * self.prev_value
        
        derivative = (value_hat - self.prev_value) / dt
        beta = self._alpha(self.config['smoothing']['d_cutoff'], dt)
        derivative_hat = beta * derivative + (1 - beta) * self.prev_derivative
        
        cutoff = self.config['smoothing']['min_cutoff'] + self.config['smoothing']['beta'] * abs(derivative_hat)
        alpha_final = self._alpha(cutoff, dt)
        
        smoothed = alpha_final * value_hat + (1 - alpha_final) * self.prev_value
        
        self.prev_value = smoothed
        self.prev_derivative = derivative_hat
        return smoothed

    def reset(self):
        self.prev_value = None
        self.prev_derivative = 0

    def _alpha(self, cutoff, dt):
        tau = 1 / (2 * np.pi * cutoff)
        return 1 / (1 + tau / dt)
