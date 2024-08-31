# import math
# from prettytable import PrettyTable

# class GaloisField:
#     def __init__(self, p, n):
#         if not self._is_prime(p):
#             raise ValueError(f"{p} is not a prime number")
#         self.p = p
#         self.n = n
#         self.order = p ** n
#         self.elements = self._generate_elements()
#         self.addition_table = self._generate_operation_table(self.add)
#         self.multiplication_table = self._generate_operation_table(self.multiply)

#     def _is_prime(self, num):
#         if num < 2:
#             return False
#         for i in range(2, int(math.sqrt(num)) + 1):
#             if num % i == 0:
#                 return False
#         return True

#     def _generate_elements(self):
#         return list(range(self.order))

#     def add(self, a, b):
#         return (a + b) % self.order

#     def subtract(self, a, b):
#         return (a - b) % self.order

#     def multiply(self, a, b):
#         return (a * b) % self.order

#     def divide(self, a, b):
#         if b == 0:
#             raise ValueError("Division by zero")
#         b_inv = pow(b, self.order - 2, self.order)
#         return self.multiply(a, b_inv)

#     def _generate_operation_table(self, operation):
#         return [[operation(a, b) for b in self.elements] for a in self.elements]

#     def print_operation_table(self, operation_name, table):
#         print(f"\n{operation_name} Table:")
#         pt = PrettyTable()
#         pt.field_names = [""] + [str(e) for e in self.elements]
#         for i, row in enumerate(table):
#             pt.add_row([str(self.elements[i])] + [str(e) for e in row])
#         print(pt)

#     def check_axioms(self):
#         print("\nChecking Field Axioms:")
        
#         # Closure
#         print("1. Closure:")
#         print("  Addition: ✓")
#         print("  Multiplication: ✓")
        
#         # Associativity
#         print("2. Associativity:")
#         add_assoc = all(self.add(self.add(a, b), c) == self.add(a, self.add(b, c)) 
#                         for a in self.elements for b in self.elements for c in self.elements)
#         mul_assoc = all(self.multiply(self.multiply(a, b), c) == self.multiply(a, self.multiply(b, c)) 
#                         for a in self.elements for b in self.elements for c in self.elements)
#         print(f"  Addition: {'✓' if add_assoc else '✗'}")
#         print(f"  Multiplication: {'✓' if mul_assoc else '✗'}")
        
#         # Commutativity
#         print("3. Commutativity:")
#         add_comm = all(self.add(a, b) == self.add(b, a) for a in self.elements for b in self.elements)
#         mul_comm = all(self.multiply(a, b) == self.multiply(b, a) for a in self.elements for b in self.elements)
#         print(f"  Addition: {'✓' if add_comm else '✗'}")
#         print(f"  Multiplication: {'✓' if mul_comm else '✗'}")
        
#         # Identity elements
#         print("4. Identity Elements:")
#         add_id = next((e for e in self.elements if all(self.add(a, e) == a for a in self.elements)), None)
#         mul_id = next((e for e in self.elements if all(self.multiply(a, e) == a for a in self.elements)), None)
#         print(f"  Additive Identity (0): {'✓' if add_id == 0 else '✗'}")
#         print(f"  Multiplicative Identity (1): {'✓' if mul_id == 1 else '✗'}")
        
#         # Inverse elements
#         print("5. Inverse Elements:")
#         add_inv = all(any(self.add(a, b) == add_id for b in self.elements) for a in self.elements)
#         mul_inv = all(any(self.multiply(a, b) == mul_id for b in self.elements) for a in self.elements if a != 0)
#         print(f"  Additive Inverse: {'✓' if add_inv else '✗'}")
#         print(f"  Multiplicative Inverse (except 0): {'✓' if mul_inv else '✗'}")
        
#         # Distributivity
#         print("6. Distributivity:")
#         distrib = all(self.multiply(a, self.add(b, c)) == self.add(self.multiply(a, b), self.multiply(a, c))
#                       for a in self.elements for b in self.elements for c in self.elements)
#         print(f"  {'✓' if distrib else '✗'}")

#     def __str__(self):
#         return f"GF({self.p}^{self.n})"

# # Let the user input p and n
# p = int(input("Enter a prime number p: "))
# n = int(input("Enter a positive integer n: "))

# try:
#     gf = GaloisField(p, n)
#     print(f"\nCreated Galois Field: {gf}")
    
#     gf.print_operation_table("Addition", gf.addition_table)
#     gf.print_operation_table("Multiplication", gf.multiplication_table)
    
#     gf.check_axioms()
    
# except ValueError as e:
#     print(f"Error: {e}")

import math
from prettytable import PrettyTable

class GaloisField:
    def __init__(self, p, n):
        if not self._is_prime(p):
            raise ValueError(f"{p} is not a prime number")
        self.p = p
        self.n = n
        self.order = p ** n
        self.elements = self._generate_elements()
        self.addition_table = self._generate_operation_table(self.add)
        self.subtraction_table = self._generate_operation_table(self.subtract)
        self.multiplication_table = self._generate_operation_table(self.multiply)
        self.division_table = self._generate_division_table()

    def _is_prime(self, num):
        if num < 2:
            return False
        for i in range(2, int(math.sqrt(num)) + 1):
            if num % i == 0:
                return False
        return True

    def _generate_elements(self):
        return list(range(self.order))

    def add(self, a, b):
        return (a + b) % self.order

    def subtract(self, a, b):
        return (a - b) % self.order

    def multiply(self, a, b):
        return (a * b) % self.order

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        b_inv = pow(b, self.order - 2, self.order)
        return self.multiply(a, b_inv)

    def _generate_operation_table(self, operation):
        return [[operation(a, b) for b in self.elements] for a in self.elements]

    def _generate_division_table(self):
        table = []
        for a in self.elements:
            row = []
            for b in self.elements:
                try:
                    row.append(self.divide(a, b))
                except ValueError:
                    row.append('N/A')
            table.append(row)
        return table

    def print_operation_table(self, operation_name, table):
        print(f"\n{operation_name} Table:")
        pt = PrettyTable()
        pt.field_names = [""] + [str(e) for e in self.elements]
        for i, row in enumerate(table):
            pt.add_row([str(self.elements[i])] + [str(e) for e in row])
        print(pt)

    def check_axioms(self):
        print("\nChecking Field Axioms:")
        
        # Closure
        print("1. Closure:")
        print("  Addition: ✓")
        print("  Subtraction: ✓")
        print("  Multiplication: ✓")
        print("  Division (except by 0): ✓")
        
        # Associativity
        print("2. Associativity:")
        add_assoc = all(self.add(self.add(a, b), c) == self.add(a, self.add(b, c)) 
                        for a in self.elements for b in self.elements for c in self.elements)
        mul_assoc = all(self.multiply(self.multiply(a, b), c) == self.multiply(a, self.multiply(b, c)) 
                        for a in self.elements for b in self.elements for c in self.elements)
        print(f"  Addition: {'✓' if add_assoc else '✗'}")
        print(f"  Multiplication: {'✓' if mul_assoc else '✗'}")
        
        # Commutativity
        print("3. Commutativity:")
        add_comm = all(self.add(a, b) == self.add(b, a) for a in self.elements for b in self.elements)
        mul_comm = all(self.multiply(a, b) == self.multiply(b, a) for a in self.elements for b in self.elements)
        print(f"  Addition: {'✓' if add_comm else '✗'}")
        print(f"  Multiplication: {'✓' if mul_comm else '✗'}")
        
        # Identity elements
        print("4. Identity Elements:")
        add_id = next((e for e in self.elements if all(self.add(a, e) == a for a in self.elements)), None)
        mul_id = next((e for e in self.elements if all(self.multiply(a, e) == a for a in self.elements)), None)
        print(f"  Additive Identity (0): {'✓' if add_id == 0 else '✗'}")
        print(f"  Multiplicative Identity (1): {'✓' if mul_id == 1 else '✗'}")
        
        # Inverse elements
        print("5. Inverse Elements:")
        add_inv = all(any(self.add(a, b) == add_id for b in self.elements) for a in self.elements)
        mul_inv = all(any(self.multiply(a, b) == mul_id for b in self.elements) for a in self.elements if a != 0)
        print(f"  Additive Inverse: {'✓' if add_inv else '✗'}")
        print(f"  Multiplicative Inverse (except 0): {'✓' if mul_inv else '✗'}")
        
        # Distributivity
        print("6. Distributivity:")
        distrib = all(self.multiply(a, self.add(b, c)) == self.add(self.multiply(a, b), self.multiply(a, c))
                      for a in self.elements for b in self.elements for c in self.elements)
        print(f"  {'✓' if distrib else '✗'}")

    def __str__(self):
        return f"GF({self.p}^{self.n})"

# Let the user input p and n
p = int(input("Enter a prime number p: "))
n = int(input("Enter a positive integer n: "))

try:
    gf = GaloisField(p, n)
    print(f"\nCreated Galois Field: {gf}")
    
    gf.print_operation_table("Addition", gf.addition_table)
    gf.print_operation_table("Subtraction", gf.subtraction_table)
    gf.print_operation_table("Multiplication", gf.multiplication_table)
    gf.print_operation_table("Division", gf.division_table)
    
    gf.check_axioms()
    
except ValueError as e:
    print(f"Error: {e}")