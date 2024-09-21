# import itertools
# from prettytable import PrettyTable

# class GF2N:
#     def __init__(self, n, irreducible_poly):
#         self.n = n
#         self.order = 2**n
#         self.irreducible_poly = irreducible_poly
#         self.elements = self._generate_elements()

#     def _generate_elements(self):
#         return list(itertools.product([0, 1], repeat=self.n))

#     def _poly_to_str(self, poly):
#         terms = []
#         for i, coef in enumerate(reversed(poly)):
#             if coef:
#                 if i == 0:
#                     terms.append('1')
#                 elif i == 1:
#                     terms.append('x')
#                 else:
#                     terms.append(f'x^{i}')
#         return ' + '.join(reversed(terms)) or '0'

#     def add(self, a, b):
#         return tuple((x + y) % 2 for x, y in zip(a, b))

#     def multiply(self, a, b):
#         result = [0] * (2 * self.n - 1)
#         for i, x in enumerate(reversed(a)):
#             for j, y in enumerate(reversed(b)):
#                 result[i + j] ^= x & y
#         return self._reduce(result)

#     def _reduce(self, poly):
#         while len(poly) > self.n:
#             if poly[-1] == 1:
#                 for i, coef in enumerate(self.irreducible_poly):
#                     poly[i + len(poly) - len(self.irreducible_poly)] ^= coef
#             poly.pop()
#         return tuple(poly[:self.n])

#     def inverse(self, a):
#         if a == tuple([0] * self.n):
#             raise ValueError("Zero has no inverse")
#         for b in self.elements[1:]:
#             if self.multiply(a, b) == tuple([1] + [0] * (self.n - 1)):
#                 return b
#         raise ValueError("Inverse not found")

#     def divide(self, a, b):
#         if b == tuple([0] * self.n):
#             raise ValueError("Division by zero")
#         return self.multiply(a, self.inverse(b))

#     def print_operation_table(self, operation_name, operation):
#         print(f"\n{operation_name} Table:")
#         pt = PrettyTable()
#         pt.field_names = [""] + [self._poly_to_str(e) for e in self.elements]
#         for a in self.elements:
#             row = [self._poly_to_str(a)]
#             for b in self.elements:
#                 try:
#                     result = operation(a, b)
#                     row.append(self._poly_to_str(result))
#                 except ValueError:
#                     row.append("N/A")
#             pt.add_row(row)
#         print(pt)

# # Create GF(2^3) with irreducible polynomial x^3 + x + 1
# gf = GF2N(3, (1, 1, 0, 1))

# # Print addition table
# gf.print_operation_table("Addition", gf.add)

# # Print multiplication table
# gf.print_operation_table("Multiplication", gf.multiply)

# # Print division table
# gf.print_operation_table("Division", gf.divide)


class GF23:
    def __init__(self):
        self.order = 8
        self.irreducible_poly = 0b1011  # x^3 + x + 1
        self.elements = list(range(self.order))
        self.poly_repr = ['0', '1', 'x', 'x+1', 'x^2', 'x^2+1', 'x^2+x', 'x^2+x+1']

    def add(self, a, b):
        return a ^ b  # XOR operation

    def multiply(self, a, b):
        result = 0
        while b:
            if b & 1:
                result ^= a
            a <<= 1
            if a & 8:
                a ^= self.irreducible_poly
            b >>= 1
        return result

    def inverse(self, a):
        if a == 0:
            raise ValueError("Zero has no multiplicative inverse")
        for i in range(1, self.order):
            if self.multiply(a, i) == 1:
                return i
        raise ValueError("Inverse not found")

    def divide(self, a, b):
        if b == 0:
            raise ValueError("Division by zero")
        return self.multiply(a, self.inverse(b))

    def print_operation_table(self, operation_name, operation):
        print(f"\n{operation_name} Table:")
        print("    | " + " | ".join(f"{self.poly_repr[i]:^7}" for i in range(self.order)))
        print("-" * (8 * 9))
        for i in range(self.order):
            row = [f"{self.poly_repr[i]:^4}"]
            for j in range(self.order):
                try:
                    result = operation(i, j)
                    row.append(f"{self.poly_repr[result]:^7}")
                except ValueError:
                    row.append("  N/A  ")
            print(" | ".join(row))

# Create GF(2^3) instance
gf23 = GF23()

# Print operation tables
gf23.print_operation_table("Addition", gf23.add)
gf23.print_operation_table("Multiplication", gf23.multiply)
gf23.print_operation_table("Division", gf23.divide)