from .equity import (
    pairwise_smds, ks_tests, bootstrap_group_stats,
    flagging_disparity, sensitivity_curve, equalized_cutoffs,
)

__version__ = '0.2.0'

__all__ = [
    '__version__',
    'pairwise_smds', 'ks_tests', 'bootstrap_group_stats',
    'flagging_disparity', 'sensitivity_curve', 'equalized_cutoffs',
]
