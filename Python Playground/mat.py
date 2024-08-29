# # # # # import matplotlib.pyplot as plt
# # # # # import matplotlib.patches as patches

# # # # # def draw_card(ax, x, y, rank, suit):
# # # # #     # Draw card rectangle
# # # # #     ax.add_patch(patches.Rectangle((x, y), 1, 1.5, edgecolor='black', facecolor='white'))
# # # # #     ax.text(x + 0.1, y + 1.4, f'{rank}', fontsize=12, verticalalignment='top')
# # # # #     ax.text(x + 0.1, y + 0.1, f'{suit}', fontsize=12, verticalalignment='bottom')

# # # # # def draw_hand(ax, hand, positions):
# # # # #     for pos, (rank, suit) in zip(positions, hand):
# # # # #         draw_card(ax, pos[0], pos[1], rank, suit)

# # # # # def setup_plot():
# # # # #     fig, ax = plt.subplots(figsize=(12, 8))
# # # # #     ax.set_xlim(0, 10)
# # # # #     ax.set_ylim(0, 5)
# # # # #     ax.axis('off')
# # # # #     return ax

# # # # # def create_poker_hand_images():
# # # # #     hands = {
# # # # #         "Royal Flush": [('10', '♠'), ('J', '♠'), ('Q', '♠'), ('K', '♠'), ('A', '♠')],
# # # # #         "Straight Flush": [('6', '♦'), ('7', '♦'), ('8', '♦'), ('9', '♦'), ('10', '♦')],
# # # # #         "Four of a Kind": [('7', '♣')] * 4 + [('2', '♦')],
# # # # #         "Full House": [('3', '♥')] * 3 + [('4', '♠')] * 2,
# # # # #         "Flush": [('2', '♠'), ('4', '♠'), ('6', '♠'), ('8', '♠'), ('10', '♠')],
# # # # #         "Straight": [('4', '♣'), ('5', '♦'), ('6', '♠'), ('7', '♥'), ('8', '♠')],
# # # # #         "Three of a Kind": [('8', '♠')] * 3 + [('2', '♦'), ('3', '♣')],
# # # # #         "Two Pair": [('5', '♠')] * 2 + [('9', '♦')] * 2 + [('2', '♥')],
# # # # #         "One Pair": [('6', '♣')] * 2 + [('2', '♠'), ('3', '♦'), ('4', '♥')]
# # # # #     }

# # # # #     for hand_name, hand in hands.items():
# # # # #         ax = setup_plot()
# # # # #         draw_hand(ax, hand, [(i, 0) for i in range(len(hand))])
# # # # #         plt.title(hand_name)
# # # # #         plt.savefig(f'{hand_name.replace(" ", "_")}.png')
# # # # #         plt.show()

# # # # # create_poker_hand_images()


# # # # # from scipy.stats import chi2
# # # # # from tabulate import tabulate

# # # # # # Observed frequencies
# # # # # observed = [3033, 4932, 1096, 780, 99, 59, 1]

# # # # # # Expected frequencies
# # # # # expected = [3024, 5040, 1080, 720, 90, 45, 1]

# # # # # # Calculate the chi-square statistic
# # # # # chi_square_stat = sum([(obs - exp)**2 / exp for obs, exp in zip(observed, expected)])

# # # # # # Degrees of freedom (number of categories - 1)
# # # # # df = len(observed) - 1

# # # # # # Critical value for alpha = 0.05 and df = 6
# # # # # alpha = 0.05
# # # # # critical_value = chi2.ppf(1 - alpha, df)

# # # # # # Determine whether to reject the null hypothesis
# # # # # reject_null = chi_square_stat > critical_value

# # # # # # Prepare data for tabulated output
# # # # # table = [
# # # # #     ["Chi-Square Statistic", chi_square_stat],
# # # # #     ["Critical Value", critical_value],
# # # # #     ["Reject Null Hypothesis", reject_null]
# # # # # ]

# # # # # # Print results in a table format
# # # # # print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


# # # # import random
# # # # from scipy.stats import chi2
# # # # from tabulate import tabulate

# # # # # Generate random three-digit observed frequencies
# # # # observed = [random.randint(100, 999) for _ in range(7)]

# # # # # Generate random three-digit expected frequencies
# # # # expected = [random.randint(100, 999) for _ in range(7)]

# # # # # Calculate the chi-square statistic
# # # # chi_square_stat = sum([(obs - exp)**2 / exp for obs, exp in zip(observed, expected)])

# # # # # Degrees of freedom (number of categories - 1)
# # # # df = len(observed) - 1

# # # # # Critical value for alpha = 0.05 and df = 6
# # # # alpha = 0.05
# # # # critical_value = chi2.ppf(1 - alpha, df)

# # # # # Determine whether to reject the null hypothesis
# # # # reject_null = chi_square_stat > critical_value

# # # # # Prepare data for tabulated output
# # # # table = [
# # # #     ["Chi-Square Statistic", chi_square_stat],
# # # #     ["Critical Value", critical_value],
# # # #     ["Reject Null Hypothesis", reject_null],
# # # #     ["Observed Frequencies", observed],
# # # #     ["Expected Frequencies", expected]
# # # # ]

# # # # # Print results in a table format
# # # # print(tabulate(table, headers=["Metric", "Value"], tablefmt="grid"))


# # # import sys

# # # def matrix_chain_order(p):
# # #     n = len(p) - 1
# # #     m = [[0 for _ in range(n)] for _ in range(n)]
# # #     s = [[0 for _ in range(n)] for _ in range(n)]

# # #     print("Initial m table:")
# # #     print_table(m)
# # #     print("\nInitial s table:")
# # #     print_table(s)

# # #     for chain_length in range(2, n + 1):
# # #         print(f"\nProcessing chain length: {chain_length}")
# # #         for i in range(n - chain_length + 1):
# # #             j = i + chain_length - 1
# # #             m[i][j] = sys.maxsize
# # #             for k in range(i, j):
# # #                 cost = m[i][k] + m[k+1][j] + p[i]*p[k+1]*p[j+1]
# # #                 if cost < m[i][j]:
# # #                     m[i][j] = cost
# # #                     s[i][j] = k
# # #             print(f"\nUpdated m[{i}][{j}] = {m[i][j]}")
# # #             print(f"Updated s[{i}][{j}] = {s[i][j]}")
# # #             print("\nCurrent m table:")
# # #             print_table(m)
# # #             print("\nCurrent s table:")
# # #             print_table(s)

# # #     return m, s

# # # def print_table(table):
# # #     for row in table:
# # #         print(' '.join(f'{cell:6}' for cell in row))

# # # def print_optimal_parens(s, i, j):
# # #     if i == j:
# # #         print(f"A{i+1}", end="")
# # #     else:
# # #         print("(", end="")
# # #         print_optimal_parens(s, i, s[i][j])
# # #         print_optimal_parens(s, s[i][j] + 1, j)
# # #         print(")", end="")

# # # # Example usage
# # # dimensions = [1,2,3,4]
# # # m, s = matrix_chain_order(dimensions)

# # # print("\nFinal m table (minimum costs):")
# # # print_table(m)

# # # print("\nFinal s table (split positions):")
# # # print_table(s)

# # # print("\nMinimum number of multiplications:", m[0][len(dimensions)-2])
# # # print("Optimal parenthesization: ", end="")
# # # print_optimal_parens(s, 0, len(dimensions)-2)
# # # print()

# # # import math

# # # def autocorrelation_test(sequence, i, m, N, M, alpha):
# # #     # Calculate ρ_35
# # #     rho_35 = 0
# # #     for k in range(M + 1):
# # #         rho_35 += sequence[i + k*m - 1] * sequence[i + (k+1)*m - 1]
# # #     rho_35 = (1 / (M + 1)) * rho_35 - 0.25

# # #     # Calculate σ_ρ35
# # #     sig_rhoDom = math.sqrt((13*M + 7))
# # #     print(sig_rhoDom)
# # #     sigma_rhoNum = 12*(M + 1);
    
# # #     sigma_rho35 = sig_rhoDom / sigma_rhoNum;
# # #     print(sigma_rho35)
    

# # #     # Calculate Z_0
# # #     Z_0 = rho_35 / sigma_rho35

# # #     # Critical value (given in the image)
# # #     z_critical = 1.96

# # #     # Test the hypothesis
# # #     if -z_critical <= Z_0 <= z_critical:
# # #         result = "Result: We Cannot reject the Null hypothesis of independence. This means No Auto-Correlation. Ztab >= Z calculated"
# # #     else:
# # #         result = "Result: Reject the Null hypothesis of independence and acccept the H1 Hypothesis. There is auto-correlation. Ztab < Z calculated"

# # #     return rho_35, sigma_rho35, Z_0, result

# # # # Given sequence
# # # sequence = [
# # #     0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
# # #     0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
# # #     0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87
# # # ]

# # # # Parameters from Example 7.8
# # # i = 3  # starting with the third number
# # # m = 4  # every five numbers
# # # N = 30  # 30 numbers in the sequence
# # # M = 5  # largest integer such that 3 + (M + 1)5 ≤ 30
# # # alpha = 0.05
# # # z_critical = 1.96
# # # # Perform the test
# # # rho_35, sigma_rho35, Z_0, result = autocorrelation_test(sequence, i, m, N, M, alpha)


# # # # Print results
# # # print(f"ρ_35 = {rho_35:.4f}")
# # # print(f"σ_ρ35 = {sigma_rho35:.4f}")
# # # print(f"Z_0 = {Z_0:.4f}")
# # # print(f"Critical Value (z_critical) = {z_critical:.2f}")
# # # print(result)


# # import math
# # import matplotlib.pyplot as plt

# # def autocorrelation_test(sequence, i, m, N, M, alpha):
# #     # Calculate ρ_35
# #     rho_35 = 0
# #     for k in range(M + 1):
# #         rho_35 += sequence[i + k*m - 1] * sequence[i + (k+1)*m - 1]
# #     rho_35 = (1 / (M + 1)) * rho_35 - 0.25

# #     # Calculate σ_ρ35
# #     sig_rhoDom = math.sqrt(12 * (M + 1))  # Inside square root
# #     sigma_rhoNum = 13 * M + 7
# #     sigma_rho35 = sig_rhoDom / sigma_rhoNum

# #     # Calculate Z_0
# #     Z_0 = rho_35 / sigma_rho35

# #     # Critical value (for alpha = 0.05 two-tailed test)
# #     z_critical = 1.96

# #     # Test the hypothesis
# #     if abs(Z_0) > z_critical:
# #         result = ("Result: Reject the null hypothesis of independence and accept the alternative hypothesis. "
# #                   "This suggests that there is significant autocorrelation in the sequence. "
# #                   "The test statistic falls outside the critical range, indicating that the observed autocorrelation is unlikely to be due to random chance.")
# #     else:
# #         result = ("Result: We cannot reject the null hypothesis of independence. "
# #                   "This means no autocorrelation is detected in the sequence. "
# #                   "The test statistic is within the critical range, indicating that any observed autocorrelation could be due to random chance.")

# #     # Plotting
# #     plt.figure(figsize=(14, 6))

# #     # Plot the original sequence
# #     plt.subplot(1, 2, 1)
# #     plt.plot(sequence, marker='o', linestyle='-', color='blue')
# #     plt.title('Original Data Sequence')
# #     plt.xlabel('Index')
# #     plt.ylabel('Value')
# #     plt.grid(True)

# #     # Plot the autocorrelation test result
# #     plt.subplot(1, 2, 2)
# #     plt.axvline(x=Z_0, color='blue', linestyle='--', label='Test Statistic (Z_0)')
# #     plt.axvline(x=-z_critical, color='red', linestyle='--', label='Critical Value (-Z_critical)')
# #     plt.axvline(x=z_critical, color='red', linestyle='--', label='Critical Value (Z_critical)')
# #     plt.fill_betweenx([-3, 3], -z_critical, z_critical, color='grey', alpha=0.2, label='Acceptance Region')
# #     plt.fill_betweenx([-3, 3], -3, -z_critical, color='orange', alpha=0.5, label='Rejection Region')
# #     plt.fill_betweenx([-3, 3], z_critical, 3, color='orange', alpha=0.5)
# #     plt.xlabel('Z Value')
# #     plt.ylabel('Density')
# #     plt.title('Autocorrelation Test Result')
# #     plt.legend()
# #     plt.grid(True)

# #     # Show the plots
# #     plt.tight_layout()
# #     plt.show()

# #     return rho_35, sigma_rho35, Z_0, result, z_critical

# # # Given sequence
# # sequence = [
# #     0.12, 0.01, 0.23, 0.28, 0.89, 0.31, 0.64, 0.28, 0.83, 0.93,
# #     0.99, 0.15, 0.33, 0.35, 0.91, 0.41, 0.60, 0.27, 0.75, 0.88,
# #     0.68, 0.49, 0.05, 0.43, 0.95, 0.58, 0.19, 0.36, 0.69, 0.87
# # ]

# # # Parameters from Example 7.8
# # i = 3  # starting with the third number
# # m = 4  # every five numbers
# # N = 30  # 30 numbers in the sequence
# # M = 5  # largest integer such that 3 + (M + 1)5 ≤ 30
# # alpha = 0.05

# # # Perform the test and plot results
# # rho_35, sigma_rho35, Z_0, result, z_critical = autocorrelation_test(sequence, i, m, N, M, alpha)

# # # Print results
# # print(f"ρ_35 = {rho_35:.4f}")
# # print(f"σ_ρ35 = {sigma_rho35:.4f}")
# # print(f"Z_0 = {Z_0:.4f}")
# # print(f"Critical Value (z_critical) = ±{z_critical:.2f}")
# # print(result)


# def extended_euclidean_algorithm(a, b):
#     """ Returns the GCD of a and b, and the coefficients x and y such that a*x + b*y = gcd(a, b). """
#     steps = []  # To record each step of the Euclidean Algorithm

#     def euclidean(a, b):
#         if b == 0:
#             steps.append((a, b, 1, 0))  # Base case: gcd(a, b) = a, coefficients (1, 0)
#             return a, 1, 0
#         else:
#             gcd, x1, y1 = euclidean(b, a % b)
#             x = y1
#             y = x1 - (a // b) * y1
#             steps.append((a, b, x, y))
#             return gcd, x, y

#     # Start the process
#     gcd, x, y = euclidean(a, b)
    
#     # Print the steps of the Euclidean Algorithm
#     print("Euclidean Algorithm Steps:")
#     for step in steps:
#         a_i, b_i, x_i, y_i = step
#         # Avoid division by zero
#         quotient = a_i // b_i if b_i != 0 else 'undefined'
#         remainder = a_i % b_i if b_i != 0 else a_i
#         print(f"{a_i} = {b_i} * {quotient} + {remainder}")
    
#     # Print the backward steps
#     print("\nBackward Substitution Steps:")
#     for step in reversed(steps):
#         a_i, b_i, x_i, y_i = step
#         print(f"Back substitution: gcd({a_i}, {b_i}) = {gcd}, coefficients (x, y) = ({x_i}, {y_i})")
    
#     return gcd, x, y

# def modular_inverse(a, m):
#     """ Returns the modular inverse of a under modulo m, or None if it does not exist. """
#     print(f"\nCalculating the modular inverse of {a} modulo {m}.")
#     gcd, x, _ = extended_euclidean_algorithm(a, m)
    
#     if gcd != 1:
#         print(f"No modular inverse exists since gcd({a}, {m}) = {gcd} != 1.")
#         return None  # Modular inverse does not exist if gcd != 1
#     else:
#         inverse = x % m
#         print(f"Modular inverse of {a} modulo {m} is {inverse}.")
#         return inverse

# # Example usage
# a = 7
# m = 19

# inverse = modular_inverse(a, m)
# if inverse is not None:
#     print(f"\nThe modular inverse of {a} modulo {m} is {inverse}.")
# else:
#     print(f"\nThe modular inverse of {a} modulo {m} does not exist.")


# from sympy import isprime, totient
# from math import gcd
# import random

# # Function to randomly select a prime number from a given range
# def random_prime(min_num, max_num):
#     primes = [num for num in range(min_num, max_num + 1) if isprime(num)]
#     if not primes:
#         raise ValueError(f"No prime numbers found in the range {min_num}-{max_num}")
#     return random.choice(primes)

# # Function to randomly select any number from a given range
# def random_any_number(min_num, max_num):
#     return random.randint(min_num, max_num)

# # Example usage

# # Random prime number selection from the range 10 to 50
# prime_number = random_prime(10, 50)
# # print(f"Randomly selected prime number: {prime_number}")

# # Random number selection from the range 10 to 50
# any_number = random_any_number(10, 50)
# # print(f"Randomly selected any number: {any_number}")

# roots = []

# # Function for modular exponentiation
# def mod_exp(base, exp, mod):
#     return pow(base, exp, mod)

# # Function to check if a number is prime
# def check_prime(n):
#     return isprime(n)

# # Function to calculate all primitive roots modulo p
# def find_primitive_roots(p):
#     if not check_prime(p):
#         raise ValueError(f"{p} is not a prime number. Cannot find primitive roots.")
#     required_set = {num for num in range(1, p)}  # Set of elements 1 to p-1
    
#     # Loop over all possible candidates for primitive roots
#     for g in range(2, p):
#         actual_set = {mod_exp(g, powers, p) for powers in range(1, p)}
#         if required_set == actual_set:  # Check if the set generated by g covers all numbers from 1 to p-1
#             roots.append(g)
    
#     return roots

# # Main function to check primality and find primitive roots
# def check_prime_and_find_roots(n):
#     if check_prime(n):
#         # print(f"{n} is a prime number.")
#         roots = find_primitive_roots(n)
#         # print(f"Primitive roots of {n} are: {roots}")
#     else:
#         print(f"{n} is a composite number.")

# # Example usage with GF(19)
# number = 17
# check_prime_and_find_roots(number)


# def text_to_numbers(text):
#     # Convert text to uppercase and ignore non-alphabet characters
#     text = text.upper()
#     result = []
    
#     for char in text:
#         if 'A' <= char <= 'Z':  # Ensure the character is a letter
#             result.append(ord(char) - ord('A') + 1)
    
#     return result

# # Example usage
# input_text = "Hi"
# converted = text_to_numbers(input_text)
# # print(f"The text '{input_text}' converted to numbers is: {converted}")


# # alpha = random.choice(list(roots))
# # q = prime_number
# # X_A = any_number
# # uncomment to use random prime numbers and private key for robustness

# q = 29
# alpha = 8
# X_A = 9

# def generateY_A():
#     return pow(alpha, X_A, q) 
#     # return alpha^X_A mod q

# Y_A = generateY_A()
# public_key = [q,  alpha, Y_A]
# print("[q, alpha, Y_A] :", public_key)

# for public_keyItems in public_key:
#     # print(public_keyItems)
#     pass


# # USER B Now use USER A public key and encrypts the Message HI
# # userB_K = random.randint(2, 100)
# userB_K = 3
# # print(userB_K)

# C1 = pow(alpha, userB_K, q)
# print("Cipher_1 is ", C1)

# # CIPHER 2 IS DYNAMIC
# userB_K_new = pow(Y_A, userB_K, q)
# print("userB_K_new", userB_K_new)

# # Message h = 8 so 
# M = 13
# C2 = (userB_K_new * M) % q
# print("C1, C2 :", C1, C2)

# # USER A RECEIVED MESSAGE AS C1 AND C2
# # DECRYPTION USING PRIVATE KEY

# K_A = C1 ** X_A % q
# print(K_A)

# K_inverse = pow(K_A,-1, q)
# print(K_inverse)

# final_DecryptMsg = C2 * K_inverse % q
# print("Final DecryptMsg: ", final_DecryptMsg)



import numpy as np
def letter_to_number(letter):
    return ord(letter) - ord('A')

def number_to_letter(number):
    return chr(number + ord('A'))

def text_to_matrix(text):
    numbers = [letter_to_number(letter) for letter in text]
    return [
        numbers[0:2],
        numbers[2:4]
    ]

def matrix_to_letters(matrix):
    return [[number_to_letter(num) for num in row] for row in matrix]

# Define the plaintext and key matrices
plain_text = "ATTACK"
P_matrix = text_to_matrix(plain_text)

matrix = np.array([
    [2, 3],
    [3, 6]
])
print(matrix[0][0])

# Define the key matrix values
P1, P2, P3, P4 = P_matrix[0][0], P_matrix[0][1], P_matrix[1][0], P_matrix[1][1]
K11, K12, K21, K22 = matrix[0, 0], matrix[0,1], matrix[1,0], matrix[1,1]

# Compute C1, C2, C3, and C4
C1 = (P1 * K11 + P2 * K12) % 26
C2 = (P1 * K21 + P2 * K22) % 26

C3 = (P3 * K11 + P4 * K12) % 26
C4 = (P3 * K21 + P4 * K22) % 26

# Form the ciphertext matrix
ciphertext_matrix = [[C1, C2], [C3, C4]]

# Convert ciphertext matrix to letters
letter_matrix = matrix_to_letters(ciphertext_matrix)

# Print the results
print("Plaintext Matrix:")
for row in P_matrix:
    print(row)

print("\nKey Matrix:")
print([[K11, K12], [K21, K22]])

print("\nCiphertext Matrix:")
for row in ciphertext_matrix:
    print(row)

print("\nCiphertext Matrix in Letters:")
for row in letter_matrix:
    print(row)

# DECRYPTION

def matrix_determinant(matrix):
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    return a * d - b * c

def matrix_adjoint(matrix):
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    return np.array([[d, -b], [-c, a]])

# Define the 2x2 matrix

# Calculate the determinant
determinant = matrix_determinant(matrix)

# Calculate the adjoint
adjoint = matrix_adjoint(matrix)

# Print the results
print("Key:")
print(matrix)

print("\nDeterminant:")
print(determinant)

print("\nAdjoint Matrix:")
print(adjoint)

def modular_inverse(a, m):
    # Using Extended Euclidean Algorithm
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"No modular inverse exists for {a} modulo {m}")
    return x % m

# Find the modular inverse of deteminant mod 26
inverse = modular_inverse(determinant, 26)
print(f"The modular inverse of 11 mod 26 is {inverse}")

final_kinverse = inverse * adjoint %  26
print(final_kinverse)

finalP1 = C1 * final_kinverse[0][0] + C2 * final_kinverse[0][1]
finalP2 = C1 * final_kinverse[1][0] + C2 * final_kinverse[1][1] 

finalP3 = C3 * final_kinverse[0][0] + C4 * final_kinverse[0][1]
finalP4 = C3 * final_kinverse[1][0] + C4 * final_kinverse[1][1] 

finalP1_ = finalP1 % 26
finalP2_ = finalP2 % 26
finalP3_ = finalP3 % 26
finalP4_ = finalP4 % 26

print("final decrypted matrix is ", finalP1_,finalP2_ ,finalP3_, finalP4_)

# Convert the decrypted numbers into a 2x2 matrix
finaldecrypted_matrix = [
    [finalP1_, finalP2_],
    [finalP3_, finalP4_]
]

# Convert the decrypted matrix to letters
decrypted_letters = matrix_to_letters(finaldecrypted_matrix)

# Print the decrypted matrix and letters
print("Decrypted Matrix:")
for row in finaldecrypted_matrix:
    print(row)

print("\nDecrypted Matrix in Letters:")
for row in decrypted_letters:
    print(row)