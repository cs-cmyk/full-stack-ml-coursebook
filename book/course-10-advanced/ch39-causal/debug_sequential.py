"""Debug sequential testing to see actual FPR values"""
import numpy as np
from scipy import stats

np.random.seed(42)

n_days = 30
users_per_day = 1000
baseline_rate = 0.05
n_simulations = 100  # Smaller for faster testing

false_positives_fixed = 0
false_positives_peeking = 0

for sim in range(n_simulations):
    cumulative_control = []
    cumulative_treatment = []

    peeking_stopped = False
    for day in range(n_days):
        control_today = np.random.binomial(1, baseline_rate, users_per_day // 2)
        treatment_today = np.random.binomial(1, baseline_rate, users_per_day // 2)

        cumulative_control.extend(control_today)
        cumulative_treatment.extend(treatment_today)

        if day >= 2 and not peeking_stopped:
            t_stat, p_val = stats.ttest_ind(cumulative_treatment, cumulative_control)
            if p_val < 0.05:
                false_positives_peeking += 1
                peeking_stopped = True
                # Don't break - let it continue to end for fixed horizon test

    # Fixed horizon: test only at day 30
    t_stat, p_val = stats.ttest_ind(cumulative_treatment, cumulative_control)
    if p_val < 0.05:
        false_positives_fixed += 1

fpr_fixed = false_positives_fixed / n_simulations
fpr_peeking = false_positives_peeking / n_simulations

print(f"Fixed horizon FPR: {fpr_fixed:.1%}")
print(f"Peeking FPR: {fpr_peeking:.1%}")
print(f"Peeking > Fixed: {fpr_peeking > fpr_fixed}")
