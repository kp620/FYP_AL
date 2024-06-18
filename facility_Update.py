"""
Facility Location Function
Selects instances that maximize the facility location function, which is a submodular function that models the diversity of a subset of instances.
"""

import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel

# designed to operate on a single class of data points at a time.
def facility_location_order(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    """
    Input: class, data, labels, metric, number of instances to select, weights, mode, number of neighbors
    Output: selected indices, weights
    """
    class_indices = np.where(y == c)[0]
    X = X[class_indices]
    N = X.shape[0]
    if mode == "dense":
        num_n = None
    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )

    # Greedily select instances that maximize the objective function
    greedyList = obj.maximize(
        budget=num_per_class,  # given budget
        optimizer="LazyGreedy",  # greedy algorithm
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
        show_progress=False,
    )

    # Extract the selected instances and their weights
    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))
    S = obj.sijs
    order = np.asarray(order, dtype=np.int32)
    sz = np.zeros(num_per_class, dtype=np.float64)

    # Calculate the weight of each selected instance
    for i in range(N):
        if np.max(S[i, order]) <= 0:
            continue
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    sz[np.where(sz == 0)] = 1
    output = []
    for orders in order:
        output.append(class_indices[orders])

    # output: Indices of the selected instances in the original dataset X, ordered by their selection into the subset.
    # sz: A count of how many times each selected instance was considered the most representative (or similar) to other instances in the class.
    return output, sz

# Orchestrates the process across all classes in the dataset
# Balancing the selection process to ensure that a proportionate number of instances from each class are selected, depending on whether equal_num is set.
# Aggregate and balance the selections made by faciliy_location_order across all classes!!!
def get_orders_and_weights(
    B, # budget 
    X, # gradient data
    metric,
    y=None, # labels
    weights=None,
    equal_num=False,
    mode="sparse",
    num_n=128,
):
    """
    Input: budget, data, metric, labels, weights, equal_num, mode, number of neighbors
    Output: selected indices, weights
    """
   # It calculates the number of instances to select per class, either equally dividing the budget B across classes or proportionally based on class frequencies.
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    classes = classes.astype(np.int32).tolist()
    C = len(classes)  # number of classes

    # Calculate the number of instances to select per class
    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(
            np.ceil(np.divide([sum(y == i) for i in classes], N) * B)
        )

    # For each class, it calls facility_location_order to select instances that maximize the facility location function.
    order_mg_all, cluster_sizes_all = zip(
        *map(
            # Collecting the selections and associated sizes (sz values) for each class.
            lambda c: facility_location_order(
                c[1], X, y, metric, num_per_class[c[0]], weights, mode, num_n
            ),
            enumerate(classes),
        )
    )

    # Combines these per-class selections into a global ordering and weighting scheme that respects the overall budget B and desired balance across classes.
    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios))  # TODO
    order_mg_all = np.array(order_mg_all, dtype=object)
    cluster_sizes_all = np.array(cluster_sizes_all, dtype=object)
    for i in range(
        int(
            np.rint(
                np.max([len(order_mg_all[c]) / props[c] for c, _ in enumerate(classes)])
            )
        )
    ):
        for c, _ in enumerate(classes):
            ndx = slice(
                i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c]))
            )
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)
    weights_mg = np.array(
        weights_mg, dtype=np.float32
    )  
    return order_mg, weights_mg
