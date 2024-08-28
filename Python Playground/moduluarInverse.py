def extended_euclidean_algorithm(a, b):
    """ Returns the GCD of a and b, and the coefficients x and y such that a*x + b*y = gcd(a, b). """
    print(f"Applying Extended Euclidean Algorithm on ({a}, {b})")
    
    # Base case
    if b == 0:
        print(f"Base case reached: gcd({a}, {b}) = {a}, coefficients (x, y) = (1, 0)")
        return a, 1, 0
    
    # Recursive step
    gcd, x1, y1 = extended_euclidean_algorithm(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    
    print(f"Back substitution: gcd({a}, {b}) = {gcd}, coefficients (x, y) = ({x}, {y})")
    return gcd, x, y

def modular_inverse(a, m):
    """ Returns the modular inverse of a under modulo m, or None if it does not exist. """
    print(f"Calculating the modular inverse of {a} modulo {m}.")
    gcd, x, _ = extended_euclidean_algorithm(a, m)
    
    if gcd != 1:
        print(f"No modular inverse exists since gcd({a}, {m}) = {gcd} != 1.")
        return None  # Modular inverse does not exist if gcd != 1
    else:
        inverse = x % m
        print(f"Modular inverse of {a} modulo {m} is {inverse}.")
        return inverse

# Example usage
a = 14
m = 19

inverse = modular_inverse(a, m)
if inverse is not None:
    print(f"The modular inverse of {a} modulo {m} is {inverse}.")
else:
    print(f"The modular inverse of {a} modulo {m} does not exist."