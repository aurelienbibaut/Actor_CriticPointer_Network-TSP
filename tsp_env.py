import numpy as np
import random
from tsp_utils import generate_n_random_point, compute_distance_for_list
import itertools
import math

def length(x, y):
    dx = x[0] - y[0]
    dy = x[1] - y[1]
    return math.sqrt(dx*dx + dy*dy)

class TSP_env(object):
    def __init__(self, n, no_move_penalty=0, use_alternative_state=False):
        """
        current_states : contains the indexes of the selected nodes eg [1,3,4]
        nodes : contains all the nodes and their 2 dimension coordinates [n,2] dimension array eg [[0.5,0.7],[0.2,0.3]]
        n : number of nodes
        """
        self.number_nodes = n
        self.no_move_penalty = no_move_penalty
        self.num_actions = self.number_nodes

        self.use_alternative_state = use_alternative_state
        if use_alternative_state:
            self.state_shape = [4 * n]
        else:
            self.state_shape = [n]
        self.env_name = 'TSP'


    def binary_vector_state(self):
        xs = np.zeros(self.number_nodes)
        for i in self.current_state:
            xs[i] = 1
        return xs

    def get_alternative_state(self):
        current_position = np.zeros(self.number_nodes)
        if len(self.current_state) > 0:
            current_position[self.current_state[-1]] = 1
        return np.concatenate([self.nodes[:, 0],
                               self.nodes[:, 1],
                               self.binary_vector_state(),
                               current_position])

    def compute_subtour_length(self):
        subtour_length = 0
        if len(self.current_state) > 1:
            for i in range(len(self.current_state) - 1):
                subtour_length += self.weight_matrix[self.current_state[i],\
                                                     self.current_state[i + 1]]
            subtour_length += self.weight_matrix[self.current_state[i + 1],\
                                                 self.current_state[0]]
        return subtour_length

    def reset(self):
        self.current_state = []
        self.nodes = generate_n_random_point(self.number_nodes)
        self.adjacency_matrix = np.ones((self.number_nodes, self.number_nodes))
        np.fill_diagonal(self.adjacency_matrix, 0)
        x1s, x2s = self.nodes[:, 0], self.nodes[:, 1]
        self.weight_matrix = np.sqrt(np.power(np.subtract.outer(x1s, x1s), 2) + \
                                     np.power(np.subtract.outer(x2s, x2s), 2))
        np.fill_diagonal(self.weight_matrix, self.no_move_penalty)

        if not self.use_alternative_state:
            return self.binary_vector_state()
        else:
            return self.get_alternative_state()

    def accumulated_reward(self):
        return -self.compute_subtour_length()

    def at_random_solution(self, visit_once=True):
        """Pick a city at random, allowing cities to be visited more than once"""
        tour = [np.random.randint(self.number_nodes)]
        while set(tour) != set(range(self.number_nodes)):
            if visit_once:
                proposed_city = np.random.randint(self.number_nodes)
                if proposed_city not in tour:
                    tour.append(proposed_city)
            else:
                tour.append(np.random.randint(self.number_nodes))
        tour.append(tour[0])
        tour_length = 0
        for i in range(len(tour) - 1):
            tour_length += self.weight_matrix[tour[i], tour[i+1]]

        return -tour_length, tour

    def best_solution_from_now(self):
        if len(self.current_state) == 0:
            return self.optimal_solution()

        remaining_cities = list(set(range(self.number_nodes)) - set(self.current_state))
        paths = list(itertools.permutations(remaining_cities))
        tour_lengths = []
        for path in paths:
            tour_length = 0
            candidate_tour = self.current_state + list(path) + [self.current_state[0]]
            for i in range(len(candidate_tour) - 1):
                tour_length += self.weight_matrix[candidate_tour[i], candidate_tour[i + 1]]
            tour_lengths.append(tour_length)
        return -np.min(tour_lengths), paths[np.argmin(tour_lengths)]

    def optimal_solution(self):
        # calc all lengths
        points = self.nodes
        all_distances = [[length(x, y) for y in points] for x in points]
        # initial value - just distance from 0 to every other point + keep the track of edges
        A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j], A[(S - {j}, k)][1] + [j]) for k in S if
                                     k != 0 and k != j])  # this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
            A = B
        res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        return -res[0], res[1]

    def step(self, action):
        done = False

        if self.current_state == []:
            self.current_state = [action]
            added_distance = 0
        else:
            if not(action in self.current_state):
                subtour_length = self.compute_subtour_length()
                self.current_state.append(action)
                new_subtour_length = self.compute_subtour_length()
                added_distance = new_subtour_length - subtour_length
            else:
                added_distance = self.no_move_penalty

        if len(self.current_state) == self.number_nodes:
            done = True
        if not self.use_alternative_state:
            return self.binary_vector_state(), -added_distance, done
        else:
            return self.get_alternative_state(), -added_distance, done
