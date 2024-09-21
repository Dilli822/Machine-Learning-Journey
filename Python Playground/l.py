from scipy.stats import chi2
from tabulate import tabulate

# Observed frequencies
# observed = [3033, 4932, 1096, 780, 99, 59, 1]
observed = [525, 419, 47, 9, 7]
# Expected frequencies
# expected = [3024, 5040, 1080, 720, 90, 45, 1]
expected = [504, 432, 27, 36, 1]

# Calculate the chi-square statistic
chi_square_stat = sum([(obs - exp)**2 / exp for obs, exp in zip(observed, expected)])

# Degrees of freedom (number of categories - 1)
df = len(observed) - 1

# Critical value for alpha = 0.05 and df = 6
alpha = 0.05
critical_value = chi2.ppf(1 - alpha, df)

# Determine whether to reject the null hypothesis
reject_null = chi_square_stat > critical_value

# Prepare data for tabulated output
table = [
    ["Chi-Square Statistic", chi_square_stat],
    ["Critical Value", critical_value],
    ["Reject Null Hypothesis", reject_null]
]

# Print results in a table format
print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))