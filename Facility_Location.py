import submodlib_config
import time 
import numpy as np
from submodlib.helper import create_kernel
from submodlib.functions.facilityLocation import FacilityLocationFunction

def facility_location_order(
    gradient_data, metric, budget, weights=None, mode="sparse", num_n=64
):
    # ------------------------------------------------------------------- #
    data = list(map(tuple, gradient_data))
    dataArray = np.array(data)
    N = len(data)
    # ------------------------------------------------------------------- #

    if mode == "dense":
        num_n = None

    start = time.time()

    # ----------------------------------------- #
    # Initilize FL function with filtered dataset
    K_dense = create_kernel(dataArray, mode=mode,metric='euclidean')
    obj = FacilityLocationFunction(n=len(gradient_data), mode=mode, sijs = K_dense, num_neighbors = num_n, separate_rep=False)
    # ----------------------------------------- #

    S_time = time.time() - start

    start = time.time()

    # ----------------------------------------------------------------------------------------------------- #
    # Optimization
    # Use greedy algorithm to select a subset of points that maximizes the total similarity within the subset
    greedyList = obj.maximize(
        budget=budget, # given budget
        optimizer="LazyGreedy", # greedy algorithm
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )

    # Indcies of the selected point
    order = list(map(lambda x: x[0], greedyList))

    # Gain in the objective function from adding the point
    sz = list(map(lambda x: x[1], greedyList))
    # ----------------------------------------------------------------------------------------------------- #

    greedy_time = time.time() - start

    S = obj.sijs
    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(budget, dtype=np.float64)


    # How many times each selected point was considered the best representative for points in X
    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]

    # If any selected point does not actually represent any other points
    sz[np.where(sz == 0)] = 1


    # Return
    # 1. Indices of the selected data points within the class
    # 2. How many times each selected point was considered the best representative
    # 3. Optimization time
    # 4. Initilization time
    return order, sz, greedy_time, S_time