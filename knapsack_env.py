import numpy as np
import knapsack

class Knapsack:
    def __init__(self, K, max_weight, state_shape = 'flat', penalize_repeat = False):
        self.K = K
        self.max_weight = max_weight
        self.penalize_repeat = penalize_repeat # Not used for now, have to figure out details
        self.env_name = 'Knapsack'
        self.state_shape = state_shape
        if self.state_shape == 'flat':
            self.state_shape = [self.K * 3]
        else:
            self.state_shape = [self.K, 3]
        self.num_actions = self.K


    def reset(self):
        self.values = np.random.rand(self.K)
        self.weights = np.random.rand(self.K)
        self.xs = np.zeros(self.K)
        self.episode_rewards = []
        if self.state_shape == 'flat':
            return np.concatenate([self.values, self.weights, self.xs])
        else:
            return np.array([self.values, self.weights, self.xs]).T

    def optimal_solution(self):
        total_reward, choices = knapsack.knapsack(self.weights, self.values).solve(self.max_weight)
        xs = np.zeros(self.K)
        for i in choices:
            xs[i] = 1
        return total_reward, xs

    def at_random_solution(self):
        current_xs = np.zeros(self.K)
        next_xs = np.zeros(self.K)
        while np.sum(current_xs) < self.K:
            next_xs[np.random.randint(self.K)] = 1
            if np.sum(self.weights * next_xs) > self.max_weight:
                break
            current_xs = np.copy(next_xs)
        return np.sum(self.values * current_xs), current_xs, \
               np.sum(self.weights * current_xs)


    def accumulated_reward(self):
        return np.sum(self.values * self.xs)

    def max_reward_to_go(self):
        remaining_weight_capacity = self.max_weight - np.sum(self.weights[self.xs == 1])
        max_rtg, _ = knapsack.knapsack(self.weights[self.xs != 1],
                                            self.values[self.xs != 1]).solve(remaining_weight_capacity)
        return max_rtg

    def step(self, action):
        # Action is the index of the next object to add
        current_sacks_weight = np.sum(self.weights * self.xs)
        if self.xs[action] == 1 or current_sacks_weight + self.weights[action] > self.max_weight: # Do nothing
            if self.state_shape == 'flat':
                new_state = np.concatenate([self.values, self.weights, self.xs])
            else:
                new_state = np.array([self.values, self.weights, self.xs]).T
            self.episode_rewards.append(0)
            return new_state, 0, False
        else:
            self.xs[action] = 1
            current_sacks_weight = np.sum(self.weights * self.xs)
            if self.state_shape == 'flat':
                new_state = np.concatenate([self.values, self.weights, self.xs])
            else:
                new_state = np.array([self.values, self.weights, self.xs]).T
            reward = self.values[action]
            self.episode_rewards.append(reward)

            if np.sum(self.xs) == self.K:
                return new_state, reward, True

            next_lightest_weight = np.min(self.weights[self.xs != 1])
            if current_sacks_weight + next_lightest_weight > self.max_weight:
                done = True
            else:
                done = False

            return new_state, reward, done

