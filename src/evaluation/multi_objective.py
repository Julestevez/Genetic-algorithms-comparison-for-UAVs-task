"""Multi-objective evaluation utilities."""

import numpy as np
from typing import List, Dict, Tuple, Union


def pareto_dominates(obj1: Union[np.ndarray, Dict[str, float]],
                     obj2: Union[np.ndarray, Dict[str, float]]) -> bool:
    """
    Check if obj1 Pareto-dominates obj2.

    Dominates if:
    - Better or equal on ALL objectives
    - Strictly better on AT LEAST ONE objective

    Args:
        obj1: First objectives (np.ndarray or dict, lower is better)
        obj2: Second objectives (np.ndarray or dict, lower is better)

    Returns:
        True if obj1 dominates obj2
    """
    # Convert to numpy arrays if dicts
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if set(obj1.keys()) != set(obj2.keys()):
            raise ValueError("Objective dicts must have same keys")
        keys = sorted(obj1.keys())
        arr1 = np.array([obj1[k] for k in keys])
        arr2 = np.array([obj2[k] for k in keys])
    elif isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        arr1 = obj1
        arr2 = obj2
    else:
        raise ValueError("obj1 and obj2 must both be dicts or both be arrays")

    better_or_equal = np.all(arr1 <= arr2)
    strictly_better = np.any(arr1 < arr2)

    return better_or_equal and strictly_better


def is_pareto_efficient(objectives: Union[List[Dict[str, float]], List[np.ndarray]]) -> List[bool]:
    """
    Find Pareto-efficient solutions.

    Args:
        objectives: List of objective dicts or np.ndarrays

    Returns:
        Boolean list indicating which solutions are Pareto-efficient
    """
    n = len(objectives)
    is_efficient = [True] * n

    for i in range(n):
        if not is_efficient[i]:
            continue
        for j in range(n):
            if i == j or not is_efficient[j]:
                continue
            # Check if j dominates i
            if pareto_dominates(objectives[j], objectives[i]):
                is_efficient[i] = False
                break

    return is_efficient


def crowding_distance(objectives: Union[List[Dict[str, float]], List[np.ndarray]]) -> np.ndarray:
    """
    Calculate crowding distance for diversity maintenance.

    Higher distance = more isolated = more valuable for diversity

    Args:
        objectives: List of objective dicts or np.ndarrays (on Pareto front)

    Returns:
        Array of crowding distances
    """
    n = len(objectives)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)

    # Convert to array format if needed
    if isinstance(objectives[0], dict):
        obj_keys = list(objectives[0].keys())
        n_objectives = len(obj_keys)
        obj_matrix = np.array([[obj[k] for k in obj_keys] for obj in objectives])
    else:
        obj_matrix = np.array(objectives)
        n_objectives = obj_matrix.shape[1]

    # For each objective dimension
    for obj_idx in range(n_objectives):
        # Sort by this objective
        values = obj_matrix[:, obj_idx]
        sorted_indices = np.argsort(values)

        # Boundary points get infinite distance
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        # Calculate distance for middle points
        obj_range = values[sorted_indices[-1]] - values[sorted_indices[0]]
        if obj_range == 0:
            continue

        for i in range(1, n - 1):
            idx = sorted_indices[i]
            distances[idx] += (
                values[sorted_indices[i + 1]] - values[sorted_indices[i - 1]]
            ) / obj_range

    return distances


def hypervolume_2d(pareto_front: Union[List[Dict[str, float]], List[np.ndarray]],
                   reference: Union[Dict[str, float], np.ndarray]) -> float:
    """
    Calculate 2D hypervolume indicator.

    Measures volume of objective space dominated by Pareto front.
    Higher is better.

    Args:
        pareto_front: List of Pareto-optimal objectives (dicts or arrays)
        reference: Reference point (nadir point, dict or array)

    Returns:
        Hypervolume value
    """
    if len(pareto_front) == 0:
        return 0.0

    # Convert to array format if needed
    if isinstance(pareto_front[0], dict):
        obj_keys = list(pareto_front[0].keys())
        if len(obj_keys) != 2:
            raise ValueError("2D hypervolume requires exactly 2 objectives")
        front_array = np.array([[obj[k] for k in obj_keys] for obj in pareto_front])
        ref_array = np.array([reference[k] for k in obj_keys])
    else:
        front_array = np.array(pareto_front)
        ref_array = np.array(reference) if isinstance(reference, (list, tuple)) else reference
        if front_array.shape[1] != 2:
            raise ValueError("2D hypervolume requires exactly 2 objectives")

    # Sort by first objective
    sorted_indices = np.argsort(front_array[:, 0])
    sorted_front = front_array[sorted_indices]

    # Calculate hypervolume
    hv = 0.0
    prev_val1 = ref_array[0]

    for obj in sorted_front:
        width = prev_val1 - obj[0]
        height = ref_array[1] - obj[1]
        if width > 0 and height > 0:
            hv += width * height
        prev_val1 = obj[0]

    return hv


def inverted_generational_distance(
    pareto_front: Union[List[Dict[str, float]], List[np.ndarray]],
    reference_front: Union[List[Dict[str, float]], List[np.ndarray]]
) -> float:
    """
    Calculate Inverted Generational Distance (IGD).

    Measures average distance from reference front to obtained front.
    Lower is better.

    Args:
        pareto_front: Obtained Pareto front (dicts or arrays)
        reference_front: True/reference Pareto front (dicts or arrays)

    Returns:
        IGD value
    """
    if len(pareto_front) == 0 or len(reference_front) == 0:
        return np.inf

    # Convert to array format if needed
    if isinstance(pareto_front[0], dict):
        obj_keys = list(pareto_front[0].keys())
        front_array = np.array([[obj[k] for k in obj_keys] for obj in pareto_front])
    else:
        front_array = np.array(pareto_front)

    if isinstance(reference_front[0], dict):
        obj_keys = list(reference_front[0].keys())
        ref_array = np.array([[obj[k] for k in obj_keys] for obj in reference_front])
    else:
        ref_array = np.array(reference_front)

    total_distance = 0.0

    for ref_obj in ref_array:
        # Find minimum distance to obtained front
        distances = np.linalg.norm(front_array - ref_obj, axis=1)
        min_dist = np.min(distances)
        total_distance += min_dist

    return total_distance / len(reference_front)


def spacing(pareto_front: Union[List[Dict[str, float]], List[np.ndarray]]) -> float:
    """
    Calculate spacing metric for distribution uniformity.

    Lower is better (more uniform distribution).

    Args:
        pareto_front: Pareto front objectives (dicts or arrays)

    Returns:
        Spacing value
    """
    if len(pareto_front) <= 1:
        return 0.0

    # Convert to array format if needed
    if isinstance(pareto_front[0], dict):
        obj_keys = list(pareto_front[0].keys())
        front_array = np.array([[obj[k] for k in obj_keys] for obj in pareto_front])
    else:
        front_array = np.array(pareto_front)

    # Calculate distances to nearest neighbor
    n = len(front_array)
    distances = []

    for i in range(n):
        min_dist = float('inf')
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(front_array[i] - front_array[j])
            min_dist = min(min_dist, dist)
        distances.append(min_dist)

    # Calculate spacing
    mean_dist = np.mean(distances)
    spacing_metric = np.sqrt(
        np.sum((np.array(distances) - mean_dist)**2) / len(distances)
    )

    return spacing_metric


def calculate_mo_metrics(
    pareto_front: Union[List[Dict[str, float]], List[np.ndarray]],
    reference_point: Union[Dict[str, float], np.ndarray] = None,
    reference_front: Union[List[Dict[str, float]], List[np.ndarray]] = None
) -> Dict[str, float]:
    """
    Calculate all multi-objective metrics.

    Args:
        pareto_front: Obtained Pareto front (dicts or arrays)
        reference_point: Reference point for hypervolume (dict or array)
        reference_front: True Pareto front for IGD (dicts or arrays)

    Returns:
        Dict of metric values
    """
    metrics = {
        'pareto_front_size': len(pareto_front),
        'spacing': spacing(pareto_front),
    }

    # Determine number of objectives
    if isinstance(pareto_front[0], dict):
        n_objectives = len(pareto_front[0].keys())
    else:
        n_objectives = len(pareto_front[0])

    if reference_point is not None and n_objectives == 2:
        metrics['hypervolume'] = hypervolume_2d(pareto_front, reference_point)

    if reference_front is not None:
        metrics['igd'] = inverted_generational_distance(pareto_front, reference_front)

    return metrics


def objectives_to_fitness(objectives: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Convert objective vector to single fitness (for single-objective algorithms).

    Args:
        objectives: Dict of objective values
        weights: Dict of weights for each objective

    Returns:
        Weighted sum fitness value
    """
    return sum(objectives[k] * weights.get(k, 1.0) for k in objectives.keys())
