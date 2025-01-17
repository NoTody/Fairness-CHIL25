import numpy as np

# Bootstrap resampling function
def bootstrap_resample(data, n_iterations=1000):
    """
    Perform bootstrap resampling to compute sample means.

    Parameters:
        data (array-like): Input data to resample.
        n_iterations (int): Number of bootstrap iterations.

    Returns:
        np.ndarray: Array of resampled means.
    """
    n = len(data)
    bootstrap_samples = np.random.choice(data, (n_iterations, n), replace=True)
    return np.mean(bootstrap_samples, axis=1)

def compute_fairness(listA1, listA2, listB1, listB2, listC1=None, listC2=None, n_iterations=1000):
    """
    Evaluate fairness of model outputs for two or three sub-cohorts.

    Parameters:
        listA1, listA2 (list of arrays): Model 1 and Model 2 scores for Subgroup A.
        listB1, listB2 (list of arrays): Model 1 and Model 2 scores for Subgroup B.
        listC1, listC2 (list of arrays, optional): Model 1 and Model 2 scores for Subgroup C.
        n_iterations (int): Number of bootstrap iterations.

    Returns:
        None: Prints aggregated differences and confidence intervals.
    """
    count = 0
    has_three_groups = listC1 is not None and listC2 is not None

    for groups in zip(listA1, listA2, listB1, listB2, *(listC1, listC2) if has_three_groups else []):
        # Convert inputs to numpy arrays
        groups = [np.array(g) for g in groups]

        # Bootstrap resampling
        bs_groups_model1 = [bootstrap_resample(groups[i], n_iterations) for i in range(0, len(groups), 2)]
        bs_groups_model2 = [bootstrap_resample(groups[i], n_iterations) for i in range(1, len(groups), 2)]

        # Calculate average pairwise differences
        def calculate_pairwise_differences(bs_groups):
            diffs = 0
            num_groups = len(bs_groups)
            for i in range(num_groups):
                for j in range(i + 1, num_groups):
                    diffs += bs_groups[i] - bs_groups[j]
            return diffs / (num_groups * (num_groups - 1) / 2)

        diffs_model1 = calculate_pairwise_differences(bs_groups_model1)
        diffs_model2 = calculate_pairwise_differences(bs_groups_model2)

        # Mean of differences
        aggregate_diff_model1 = np.mean(diffs_model1)
        aggregate_diff_model2 = np.mean(diffs_model2)

        # Confidence intervals
        ci_model1 = np.percentile(diffs_model1, [2.5, 97.5])
        ci_model2 = np.percentile(diffs_model2, [2.5, 97.5])

        print(f"Class {count + 1}:")
        print(f"  Model 1: Mean = {aggregate_diff_model1:.2f}, 95% CI = [{ci_model1[0] * 100:.2f}, {ci_model1[1] * 100:.2f}]")
        print(f"  Model 2: Mean = {aggregate_diff_model2:.2f}, 95% CI = [{ci_model2[0] * 100:.2f}, {ci_model2[1] * 100:.2f}]")

        count += 1

# Example usage
# compute_fairness(listA1, listA2, listB1, listB2, n_iterations=1000) for two samples
# compute_fairness(listA1, listA2, listB1, listB2, listC1, listC2, n_iterations=1000) for three samples
