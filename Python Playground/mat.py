def sum_of_sub(s, k, r, w, m, x, solutions):
    # Generate left child (include w[k])
    x[k] = 1
    if s + w[k] == m:
        # If a subset that sums to m is found, add it to the solutions list
        solutions.append([w[i] for i in range(k + 1) if x[i] == 1])
    elif s + w[k] + w[k + 1] <= m:
        # Recur with the current element included
        sum_of_sub(s + w[k], k + 1, r - w[k], w, m, x, solutions)
    
    # Generate right child (exclude w[k])
    if (s + r - w[k] >= m) and (s + w[k + 1] <= m):
        x[k] = 0
        sum_of_sub(s, k + 1, r - w[k], w, m, x, solutions)

def find_subsets(w, m):
    n = len(w)
    x = [0] * n  # To track the subset inclusion
    solutions = []
    total_sum = sum(w)  # r is the sum of all remaining elements initially
    sum_of_sub(0, 0, total_sum, w, m, x, solutions)
    return solutions

# Example usage
w = [5, 10, 12, 13, 15, 18]
m = 30
solutions = find_subsets(w, m)
print("Subsets that sum to", m, ":")
for subset in solutions:
    print(subset)
