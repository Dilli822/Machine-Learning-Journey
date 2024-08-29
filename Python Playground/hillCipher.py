import numpy as np

def text_to_numbers(text):
    return [ord(char) - ord('A') for char in text.upper()]

def numbers_to_text(numbers):
    return ''.join(chr(num + ord('A')) for num in numbers)

def hill_cipher_encrypt(plain_text, key_matrix):
    # Ensure the key matrix is a NumPy array
    key_matrix = np.array(key_matrix)
    
    # Convert plaintext to numbers
    plain_numbers = text_to_numbers(plain_text)
    
    # Pad plaintext to fit the matrix size
    padding_length = (len(plain_numbers) % key_matrix.shape[0])
    if padding_length != 0:
        plain_numbers.extend([0] * (key_matrix.shape[0] - padding_length))
    
    # Encrypt in blocks
    cipher_numbers = []
    for i in range(0, len(plain_numbers), key_matrix.shape[0]):
        block = plain_numbers[i:i + key_matrix.shape[0]]
        block_matrix = np.array(block).reshape((key_matrix.shape[0], 1))
        encrypted_block = (key_matrix @ block_matrix) % 26
        cipher_numbers.extend(encrypted_block.flatten())
    
    # Convert numbers to text
    return numbers_to_text(cipher_numbers)

# Example usage
key_matrix = [
    [17, 17, 5],
        [21, 18, 21],
            [2, 2, 19],
    
]

plain_text = "PAY MORE MONEY"
cipher_text = hill_cipher_encrypt(plain_text, key_matrix)
print(f"Encrypted Text: {cipher_text}")
