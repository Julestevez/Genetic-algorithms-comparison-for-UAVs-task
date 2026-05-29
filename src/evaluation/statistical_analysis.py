"""Statistical analysis for comparing algorithm performance."""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any


class StatisticalAnalyzer:
    """Perform statistical analysis on benchmark results."""

    def __init__(self, confidence_level: float = 0.95):
        """Initialize statistical analyzer."""
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def compare_algorithms(self,
                          results_by_algorithm: Dict[str, List[float]],
                          test_type: str = 'wilcoxon') -> Dict[str, Any]:
        """
        Compare algorithms using statistical tests.

        Args:
            results_by_algorithm: Dict mapping algorithm names to fitness values
            test_type: Type of test ('t_test', 'wilcoxon', 'friedman', 'anova')

        Returns:
            Dictionary with statistical test results
        """
        algorithms = self._extract_algorithm_names(results_by_algorithm)
        results = {}

        if test_type == 't_test':
            results = self._paired_t_tests(results_by_algorithm)

        elif test_type == 'wilcoxon':
            results = self._wilcoxon_tests(results_by_algorithm)

        elif test_type == 'friedman':
            results = self._friedman_test(results_by_algorithm)

        elif test_type == 'anova':
            results = self._anova_test(results_by_algorithm)

        results['test_type'] = test_type
        results['confidence_level'] = self.confidence_level
        results['algorithms'] = algorithms

        if self._is_grouped_by_scenario(results_by_algorithm):
            results['scenarios'] = list(results_by_algorithm.keys())

        return results

    def _paired_t_tests(self, results_by_algorithm: Dict[str, List[float]]) -> Dict:
        """Perform pairwise t-tests."""
        algorithms = list(results_by_algorithm.keys())
        n = len(algorithms)

        p_values = np.zeros((n, n))
        significant = np.zeros((n, n), dtype=bool)

        for i in range(n):
            for j in range(i + 1, n):
                algo_i = algorithms[i]
                algo_j = algorithms[j]

                data_i = results_by_algorithm[algo_i]
                data_j = results_by_algorithm[algo_j]

                # Paired t-test
                statistic, p_value = stats.ttest_rel(data_i, data_j)

                p_values[i, j] = p_value
                p_values[j, i] = p_value

                significant[i, j] = p_value < self.alpha
                significant[j, i] = p_value < self.alpha

        return {
            'p_values': p_values.tolist(),
            'significant': significant.tolist(),
            'method': 'Paired t-test'
        }

    def _wilcoxon_tests(self, results_by_algorithm: Dict[str, List[float]]) -> Dict:
        """Perform pairwise Wilcoxon signed-rank tests."""
        if self._is_grouped_by_scenario(results_by_algorithm):
            scenarios = list(results_by_algorithm.keys())
            algorithms = self._extract_algorithm_names(results_by_algorithm)
            n = len(algorithms)

            raw_p_values = {}
            corrected_p_values = {}
            significant_raw = {}
            significant_corrected = {}
            comparisons = []

            for scenario in scenarios:
                raw_p_values[scenario] = np.zeros((n, n))
                corrected_p_values[scenario] = np.zeros((n, n))
                significant_raw[scenario] = np.zeros((n, n), dtype=bool)
                significant_corrected[scenario] = np.zeros((n, n), dtype=bool)

                scenario_results = results_by_algorithm[scenario]

                for i in range(n):
                    for j in range(i + 1, n):
                        algo_i = algorithms[i]
                        algo_j = algorithms[j]

                        data_i = self._coerce_float_list(scenario_results[algo_i])
                        data_j = self._coerce_float_list(scenario_results[algo_j])

                        try:
                            statistic, p_value = stats.wilcoxon(data_i, data_j)
                            statistic = float(statistic)
                            p_value = float(p_value)
                        except Exception:
                            statistic = float('nan')
                            p_value = 1.0

                        raw_p_values[scenario][i, j] = p_value
                        raw_p_values[scenario][j, i] = p_value
                        significant_raw[scenario][i, j] = p_value < self.alpha
                        significant_raw[scenario][j, i] = p_value < self.alpha

                        comparisons.append({
                            'scenario': scenario,
                            'algorithm_a': algo_i,
                            'algorithm_b': algo_j,
                            'statistic': statistic,
                            'raw_p_value': p_value,
                        })

            corrected_values = self._holm_bonferroni_correction(
                [comparison['raw_p_value'] for comparison in comparisons]
            )

            for comparison, corrected_p_value in zip(comparisons, corrected_values):
                scenario = comparison['scenario']
                i = algorithms.index(comparison['algorithm_a'])
                j = algorithms.index(comparison['algorithm_b'])

                corrected_p_value = float(corrected_p_value)
                comparison['corrected_p_value'] = corrected_p_value
                comparison['significant_raw'] = comparison['raw_p_value'] < self.alpha
                comparison['significant_corrected'] = corrected_p_value < self.alpha

                corrected_p_values[scenario][i, j] = corrected_p_value
                corrected_p_values[scenario][j, i] = corrected_p_value
                significant_corrected[scenario][i, j] = corrected_p_value < self.alpha
                significant_corrected[scenario][j, i] = corrected_p_value < self.alpha

            return {
                'p_values': {
                    scenario: matrix.tolist()
                    for scenario, matrix in raw_p_values.items()
                },
                'corrected_p_values': {
                    scenario: matrix.tolist()
                    for scenario, matrix in corrected_p_values.items()
                },
                'significant_raw': {
                    scenario: matrix.tolist()
                    for scenario, matrix in significant_raw.items()
                },
                'significant': {
                    scenario: matrix.tolist()
                    for scenario, matrix in significant_corrected.items()
                },
                'comparisons': comparisons,
                'method': 'Wilcoxon signed-rank test',
                'correction': 'Holm-Bonferroni',
                'significant_after_correction': [
                    comparison for comparison in comparisons
                    if comparison['significant_corrected']
                ],
            }

        algorithms = list(results_by_algorithm.keys())
        n = len(algorithms)

        p_values = np.zeros((n, n))
        corrected_p_values = np.zeros((n, n))
        significant_raw = np.zeros((n, n), dtype=bool)
        significant = np.zeros((n, n), dtype=bool)
        comparisons = []

        for i in range(n):
            for j in range(i + 1, n):
                algo_i = algorithms[i]
                algo_j = algorithms[j]

                data_i = results_by_algorithm[algo_i]
                data_j = results_by_algorithm[algo_j]

                # Wilcoxon signed-rank test
                try:
                    statistic, p_value = stats.wilcoxon(data_i, data_j)
                    statistic = float(statistic)
                    p_value = float(p_value)
                except Exception:
                    statistic = float('nan')
                    p_value = 1.0

                p_values[i, j] = p_value
                p_values[j, i] = p_value

                significant_raw[i, j] = p_value < self.alpha
                significant_raw[j, i] = p_value < self.alpha

                comparisons.append({
                    'scenario': 'pooled',
                    'algorithm_a': algo_i,
                    'algorithm_b': algo_j,
                    'statistic': statistic,
                    'raw_p_value': p_value,
                })

        corrected_values = self._holm_bonferroni_correction(
            [comparison['raw_p_value'] for comparison in comparisons]
        )

        for comparison, corrected_p_value in zip(comparisons, corrected_values):
            i = algorithms.index(comparison['algorithm_a'])
            j = algorithms.index(comparison['algorithm_b'])

            corrected_p_value = float(corrected_p_value)
            comparison['corrected_p_value'] = corrected_p_value
            comparison['significant_raw'] = comparison['raw_p_value'] < self.alpha
            comparison['significant_corrected'] = corrected_p_value < self.alpha

            corrected_p_values[i, j] = corrected_p_value
            corrected_p_values[j, i] = corrected_p_value
            significant[i, j] = corrected_p_value < self.alpha
            significant[j, i] = corrected_p_value < self.alpha

        return {
            'p_values': p_values.tolist(),
            'corrected_p_values': corrected_p_values.tolist(),
            'significant_raw': significant_raw.tolist(),
            'significant': significant.tolist(),
            'comparisons': comparisons,
            'method': 'Wilcoxon signed-rank test',
            'correction': 'Holm-Bonferroni',
        }

    def _friedman_test(self, results_by_algorithm: Dict[str, List[float]]) -> Dict:
        """Perform Friedman test."""
        algorithms = self._extract_algorithm_names(results_by_algorithm)

        if self._is_grouped_by_scenario(results_by_algorithm):
            scenarios = list(results_by_algorithm.keys())
            scenario_ranks = {}
            rank_history = {algorithm: [] for algorithm in algorithms}
            data = []

            for algorithm in algorithms:
                algorithm_means = []
                for scenario in scenarios:
                    scenario_values = results_by_algorithm[scenario][algorithm]
                    algorithm_means.append(float(np.mean(self._coerce_float_list(scenario_values))))
                data.append(algorithm_means)

            for scenario in scenarios:
                scenario_means = [
                    float(np.mean(self._coerce_float_list(results_by_algorithm[scenario][algorithm])))
                    for algorithm in algorithms
                ]
                ranks = stats.rankdata(scenario_means, method='average')
                scenario_ranks[scenario] = {
                    algorithm: float(rank)
                    for algorithm, rank in zip(algorithms, ranks)
                }
                for algorithm, rank in zip(algorithms, ranks):
                    rank_history[algorithm].append(float(rank))

            statistic, p_value = stats.friedmanchisquare(*data)

            mean_ranks = {
                algorithm: float(np.mean(rank_history[algorithm]))
                for algorithm in algorithms
            }

            return {
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'method': 'Friedman test',
                'mean_ranks': mean_ranks,
                'scenario_ranks': scenario_ranks,
            }

        trimmed_data = [self._coerce_float_list(results_by_algorithm[algo]) for algo in algorithms]
        min_length = min(len(values) for values in trimmed_data)
        data = [values[:min_length] for values in trimmed_data]

        statistic, p_value = stats.friedmanchisquare(*data)

        rank_history = {algorithm: [] for algorithm in algorithms}
        for block_idx in range(min_length):
            block_values = [data[algo_idx][block_idx] for algo_idx in range(len(algorithms))]
            ranks = stats.rankdata(block_values, method='average')
            for algorithm, rank in zip(algorithms, ranks):
                rank_history[algorithm].append(float(rank))

        mean_ranks = {
            algorithm: float(np.mean(rank_history[algorithm]))
            for algorithm in algorithms
        }

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'method': 'Friedman test',
            'mean_ranks': mean_ranks,
        }

    def _anova_test(self, results_by_algorithm: Dict[str, List[float]]) -> Dict:
        """Perform one-way ANOVA."""
        algorithms = list(results_by_algorithm.keys())
        data = [results_by_algorithm[algo] for algo in algorithms]

        statistic, p_value = stats.f_oneway(*data)

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'method': 'One-way ANOVA'
        }

    def calculate_effect_size(self,
                             data1: List[float],
                             data2: List[float]) -> Dict[str, float]:
        """Calculate effect size (Cohen's d)."""
        mean1, mean2 = np.mean(data1), np.mean(data2)
        std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
        n1, n2 = len(data1), len(data2)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

        return {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(abs(cohens_d))
        }

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def generate_rankings(self,
                         results_by_algorithm: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Generate algorithm rankings based on performance.

        Args:
            results_by_algorithm: Dict mapping algorithm names to fitness values

        Returns:
            Rankings and statistics
        """
        rankings = []

        for algo_name, values in results_by_algorithm.items():
            # Skip if no values for this algorithm
            if not values or len(values) == 0:
                print(f"Warning: No results for algorithm '{algo_name}', skipping...")
                continue

            rankings.append({
                'algorithm': algo_name,
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
            })

        # Sort by mean (lower is better for minimization)
        rankings.sort(key=lambda x: x['mean'])

        # Add ranks
        for i, r in enumerate(rankings):
            r['rank'] = i + 1

        return {
            'rankings': rankings,
            'best': rankings[0]['algorithm'] if rankings else None,
            'worst': rankings[-1]['algorithm'] if rankings else None,
        }

    def _is_grouped_by_scenario(self, results: Dict[str, Any]) -> bool:
        """Check whether results are grouped as scenario -> algorithm -> values."""
        if not results:
            return False
        first_value = next(iter(results.values()))
        return isinstance(first_value, dict)

    def _extract_algorithm_names(self, results: Dict[str, Any]) -> List[str]:
        """Extract algorithm names from flat or scenario-grouped results."""
        if not self._is_grouped_by_scenario(results):
            return list(results.keys())

        algorithms = []
        for scenario_results in results.values():
            for algorithm in scenario_results.keys():
                if algorithm not in algorithms:
                    algorithms.append(algorithm)
        return algorithms

    def _coerce_float_list(self, values: Any) -> List[float]:
        """Convert a scalar, tuple, list, or ndarray to a flat float list."""
        if isinstance(values, np.ndarray):
            return values.astype(float).flatten().tolist()
        if isinstance(values, (list, tuple)):
            return [float(value) for value in values]
        return [float(values)]

    def _holm_bonferroni_correction(self, raw_p_values: List[float]) -> List[float]:
        """Apply Holm-Bonferroni correction and return adjusted p-values."""
        if not raw_p_values:
            return []

        raw = np.asarray(raw_p_values, dtype=float)
        order = np.argsort(raw)
        adjusted_sorted = np.zeros(len(raw), dtype=float)
        running_max = 0.0

        for rank_index, raw_index in enumerate(order):
            corrected = raw[raw_index] * (len(raw) - rank_index)
            running_max = max(running_max, corrected)
            adjusted_sorted[rank_index] = min(running_max, 1.0)

        adjusted = np.zeros(len(raw), dtype=float)
        for rank_index, raw_index in enumerate(order):
            adjusted[raw_index] = adjusted_sorted[rank_index]

        return adjusted.tolist()
