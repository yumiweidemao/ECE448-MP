'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import numpy as np

epsilon = 1e-3

def compute_transition_matrix(model):
    '''
    Parameters:
    model - the MDP model returned by load_MDP()

    Output:
    P - An M x N x 4 x M x N numpy array. P[r, c, a, r', c'] is the probability that the agent will move from cell (r, c) to (r', c') if it takes action a, where a is 0 (left), 1 (up), 2 (right), or 3 (down).
    '''
    P = np.zeros(shape=(model.M, model.N, 4, model.M, model.N))
    for i in range(model.M):
        for j in range(model.N):
            for d in range(4):
                if model.T[i, j]:
                    P[i, j, d, :, :] = 0
                    continue
                # calculate current direction probability
                if d == 0:
                    # left (i, j-1) with prob = model.D[i, j, 0]
                    if j-1 < 0 or model.W[i, j-1]:
                        P[i, j, 0, i, j] += model.D[i, j, 0]
                    else:
                        P[i, j, 0, i, j-1] += model.D[i, j, 0]
                    # down (i+1, j) with prob = model.D[i, j, 1]
                    if i+1 >= model.M or model.W[i+1, j]:
                        P[i, j, 0, i, j] += model.D[i, j, 1]
                    else:
                        P[i, j, 0, i+1, j] += model.D[i, j, 1]
                    # up (i-1, j) with prob = model.D[i, j, 2]
                    if i-1 < 0 or model.W[i-1, j]:
                        P[i, j, 0, i, j] += model.D[i, j, 2]
                    else:
                        P[i, j, 0, i-1, j] += model.D[i, j, 2]
                elif d == 1:
                    # up (i-1, j) with prob = model.D[i, j, 0]
                    if i-1 < 0 or model.W[i-1, j]:
                        P[i, j, 1, i, j] += model.D[i, j, 0]
                    else:
                        P[i, j, 1, i-1, j] += model.D[i, j, 0]
                    # left (i, j-1) with prob = model.D[i, j, 1]
                    if j-1 < 0 or model.W[i, j-1]:
                        P[i, j, 1, i, j] += model.D[i, j, 1]
                    else:
                        P[i, j, 1, i, j-1] += model.D[i, j, 1]
                    # right (i, j+1) with prob = model.D[i, j, 2]
                    if j+1 >= model.N or model.W[i, j+1]:
                        P[i, j, 1, i, j] += model.D[i, j, 2]
                    else:
                        P[i, j, 1, i, j+1] += model.D[i, j, 2]
                elif d == 2:
                    # right (i, j+1) with prob = model.D[i, j, 0]
                    if j+1 >= model.N or model.W[i, j+1]:
                        P[i, j, 2, i, j] += model.D[i, j, 0]
                    else:
                        P[i, j, 2, i, j+1] += model.D[i, j, 0]
                    # up (i-1, j) with prob = model.D[i, j, 1]
                    if i-1 < 0 or model.W[i-1, j]:
                        P[i, j, 2, i, j] += model.D[i, j, 1]
                    else:
                        P[i, j, 2, i-1, j] += model.D[i, j, 1]
                    # down (i+1, j) with prob = model.D[i, j, 2]
                    if i+1 >= model.M or model.W[i+1, j]:
                        P[i, j, 2, i, j] += model.D[i, j, 2]
                    else:
                        P[i, j, 2, i+1, j] += model.D[i, j, 2]
                else:
                    # down (i+1, j) with prob = model.D[i, j, 0]
                    if i+1 >= model.M or model.W[i+1, j]:
                        P[i, j, 3, i, j] += model.D[i, j, 0]
                    else:
                        P[i, j, 3, i+1, j] += model.D[i, j, 0]
                    # right (i, j+1) with prob = model.D[i, j, 1]
                    if j+1 >= model.N or model.W[i, j+1]:
                        P[i, j, 3, i, j] += model.D[i, j, 1]
                    else:
                        P[i, j, 3, i, j+1] += model.D[i, j, 1]
                    # left (i, j-1) with prob = model.D[i, j, 2]
                    if j-1 < 0 or model.W[i, j-1]:
                        P[i, j, 3, i, j] += model.D[i, j, 2]
                    else:
                        P[i, j, 3, i, j-1] += model.D[i, j, 2]

    return P




def update_utility(model, P, U_current):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()
    P - The precomputed transition matrix returned by compute_transition_matrix()
    U_current - The current utility function, which is an M x N array

    Output:
    U_next - The updated utility function, which is an M x N array
    '''
    U_next = np.zeros(shape=U_current.shape)

    for i in range(model.M):
        for j in range(model.N):
            largest = -float("inf")
            for d in range(4):
                s = 0
                for k in range(model.M):
                    for h in range(model.N):
                        s += P[i, j, d, k, h] * U_current[k, h]
                if s > largest:
                    largest = s
            U_next[i, j] = model.R[i, j] + model.gamma * largest

    return U_next

def value_iteration(model):
    '''
    Parameters:
    model - The MDP model returned by load_MDP()

    Output:
    U - The utility function, which is an M x N array
    '''
    P = compute_transition_matrix(model)
    U_current = np.zeros(shape=(model.M, model.N))
    U_next = update_utility(model, P, U_current)
    while True:
        # check for termination
        if np.all(U_next - U_current < epsilon):
            return U_next
        U_current = U_next
        U_next = update_utility(model, P, U_current)

if __name__ == "__main__":
    import utils
    model = utils.load_MDP('models/small.json')
    model.visualize()
    U = value_iteration(model)
    model.visualize(U)
