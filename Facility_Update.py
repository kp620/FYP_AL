import time

import numpy as np
from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel

# designed to operate on a single class of data points at a time.
def faciliy_location_order(
    c, X, y, metric, num_per_class, weights=None, mode="sparse", num_n=128
):
    class_indices = []
    for i in range(len(y)):
        if y[i] == c:
            class_indices.append(i)
    # Given a class c, a dataset X and labels y
    # Filters X to include instances of only class c
    X = X[class_indices]
    print("X: ", X.shape)
    N = X.shape[0]


    if mode == "dense":
        num_n = None


    obj = FacilityLocationFunction(
        n=len(X), mode=mode, data=X, metric=metric, num_neighbors=num_n
    )

    greedyList = obj.maximize(
        budget=num_per_class,  # given budget
        optimizer="LazyGreedy",  # greedy algorithm
        stopIfZeroGain=False,
        stopIfNegativeGain=False,
        verbose=False,
    )

    order = list(map(lambda x: x[0], greedyList))
    sz = list(map(lambda x: x[1], greedyList))

    S = obj.sijs
    order = np.asarray(order, dtype=np.int32)
    sz = np.zeros(num_per_class, dtype=np.float64)

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

    # 1. Indices of the selected instances in the original dataset X, ordered by their selection into the subset.
    # 2. A count of how many times each selected instance was considered the most representative (or similar) to other instances in the class.
    return output, sz

# Orchestrates the process across all classes in the dataset
# Balancing the selection process to ensure that a proportionate number of instances from each class are selected, depending on whether equal_num is set.
# !!!Aggregate and balance the selections made by faciliy_location_order across all classes
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
   # It calculates the number of instances to select per class, either equally dividing the budget B across classes or proportionally based on class frequencies.
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    unique_values = set(y)
    classes = list(unique_values)
    C = len(classes)  # number of classes

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

    print(f"Greedy: selecting {num_per_class} elements")

    order_mg_all, cluster_sizes_all = zip(
        *map(
            # Collecting the selections and associated sizes (sz values) for each class.
            lambda c: faciliy_location_order(
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


# def select_random_set(gradients):
#     total_number = len(gradients)
#     print("total_number: ", total_number)
#     indices = np.arange(total_number) 
#     np.random.shuffle(indices)
#     subset_size = int(np.ceil(total_number / 4000))
#     print("subset_size: ", subset_size)
#     subsets = [indices[i * subset_size:(i + 1) * subset_size] for i in range(4000)]
#     return subsets

# def get_coreset_weight(gradients, pseudo_labels):
#     subsets = select_random_set(gradients)
#     print("Greedy FL Start!")
#     subset_count = 0
#     for subset in subsets: 
#         if subset_count % 500 == 0:
#             print("Handling subset #", subset_count, " out of #", len(subsets))
#         gradient_data = gradients[subset].squeeze()
#         if gradient_data.size > 0:
#             gradient_data = gradient_data.reshape(gradient_data.shape[0], -1)
#         else:
#             continue
#         # Facility location function

#         # length = len(gradients)
#         # pseudo_labels = np.random.randint(0, 2, length)
#         print("length of pseudo_labels", len(pseudo_labels))
#         fl_labels = pseudo_labels[subset] - np.min(pseudo_labels[subset])
#         print("fl_labels completed")
        
#         print("get order and weights start")
#         subset, weights= get_orders_and_weights(
#         15,
#         gradient_data,
#         "euclidean",
#         y=fl_labels,
#         equal_num=False,
#         mode="sparse",
#         num_n=128,
#     )
#         print(subset)
#         print("len coreset", len(subset))
#         # print(weights)
#         print("len weights", len(weights))

#     print("Greedy FL End!")
#     return subset, weights



