
import numpy as np


def adjacency_matrix(list_nodes):
    """
    compute the adjacency matrix
    input : list of nodes example (1,2,3,4)
    output: adjacency matrix example [[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]]
    """

    # check min index is 0 or 1
    min_index = min(list_nodes)
    length_list = len(list_nodes)
    if min_index == 1:
        list_nodes = np.array(list_nodes) - 1

    adj_mat = np.zeros((length_list, length_list))
    for n in range(length_list - 1):
        i = list_nodes[n]
        j = list_nodes[n + 1]
        adj_mat[i][j] = 1
        adj_mat[j][i] = 1
    i = list_nodes[length_list - 1]
    j = list_nodes[0]
    adj_mat[i][j] = 1
    adj_mat[j][i] = 1
    return adj_mat


def flatten_adj_matrix(adj_matrix):
    """
    flatten a N x N into a 1 x N^2 vector
    input : N x N matrix
    output : 1 x N^2 matrix
    """
    return adj_matrix.reshape((1, -1))


def adj_matrix_from_vector(vector, number_nodes):
    """
    retrieve the adjacency matrix from a vector
    input : 1 x N^2 matrix
    output : N x N matrix
    """
    return vector.reshape((number_nodes, number_nodes))


def all_circular_permutations(solution):
    """
    compute all the circular permutation of a solution
    (list of nodes) which are also a solution.
    input : list of nodes (solution)
    output : list of circular permutations of solution
    """
    n = len(solution)
    return [[solution[i - j] for i in range(n)] for j in range(n)]


def check_optimality_supervised(solution, opt_solution):
    """
    check wether solution is cicular permutation of
    opt_solution
    input : solution and optimal solution
    output : boolean
    """
    return solution in all_circular_permutations(opt_solution)


def check_neighbor_constraint(solution_adj_matrix, node, min_index):
    """
    check whether a node in a solution
    has exactly 2 neighbors or not.
    input : solution adjacency matrix, node
    output : boolean
    """
    if min_index == 1:
        node -= 1
    return sum(solution_adj_matrix[node]) == 2


def check_subtour_constraints(list_nodes, solution_adj_matrix):
    """
    check whether the subtour constraints
    are violated or not.
    input : solution adjacency matrix, node
    output : boolean
    """
    import itertools
    # check min index is 0 or 1
    min_index = min(list_nodes)
    length_list = len(list_nodes)
    if min_index == 1:
        list_nodes = np.array(list_nodes) - 1

    subtour_sizes = range(1, length_list)
    subtours = []
    for k in subtour_sizes:
        l = list(itertools.combinations(list_nodes, k))
        subtours.append([list(ll) for ll in l])
    for k_subtours in subtours:
        for subtour in k_subtours:
            if np.sum(solution_adj_matrix[np.ix_(
                    subtour, subtour)]) - len(subtour) >= len(subtour):
                return False

    return True


def compute_cost(distance_matrix, solution_adj_matrix, tour_length):
    """
    compute the total distance of a tour
    input : distance matrix, solution adjacency matrix, tour length
    output : tour cost
    """
    cost = 0
    for i in range(tour_length):
        cost += np.sum(distance_matrix[i] * solution_adj_matrix[i])

    return cost / 2


def node_list_from_adj_matrix(adj_matrix, number_of_nodes,min_index):
    """
    return a permutation using the adjacency matrix
    input : adjacency matrix, number of nodes
    output : a permutation of a tour
    """

    visited = [0]
    for i in range(number_of_nodes - 1):
        l = list(np.where(adj_matrix[visited[i]])[0])

        next_neighbor = [x for x in l if x not in visited]
        visited.append(np.min(next_neighbor))

    if min_index:
        visited = [x + 1 for x in visited]

    return visited

def generate_n_random_point(n):
    import random
    node_list =[]
    for i in range(n):
        x_i = random.random()
        y_i = random.random()
        point=np.array([x_i,y_i])
        node_list.append(point)
    return np.array(node_list)

def compute_distance_for_list(node_list):
    d = 0
    for i in range(len(node_list)-1):
        d += np.linalg.norm(node_list[i]-node_list[i+1])
    d += np.linalg.norm(node_list[-1]-node_list[0])
    return d