import heapq
from collections import defaultdict, Counter

class Node:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [Node(char, freq) for char, freq in freq_dict.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)

        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right

        heapq.heappush(heap, merged)

    return heap[0]

def generate_codes(root, current_code="", codes={}):
    if root is None:
        return

    if root.char is not None:
        codes[root.char] = current_code
        return

    generate_codes(root.left, current_code + "0", codes)
    generate_codes(root.right, current_code + "1", codes)

    return codes

def encode_message(message, codes):
    return ''.join(codes[char] for char in message)

def decode_message(encoded_message, root):
    decoded_message = []
    current_node = root

    for bit in encoded_message:
        if bit == '0':
            current_node = current_node.left
        else:
            current_node = current_node.right

        if current_node.char is not None:
            decoded_message.append(current_node.char)
            current_node = root

    return ''.join(decoded_message)

def huffman_encoding(message):
    freq_dict = Counter(message)
    huffman_tree_root = build_huffman_tree(freq_dict)
    huffman_codes = generate_codes(huffman_tree_root)
    encoded_message = encode_message(message, huffman_codes)
    return encoded_message, huffman_tree_root, huffman_codes, freq_dict

# Function to calculate total bits before and after Huffman encoding
def calculate_total_bits(freq_dict, huffman_codes, message_length):
    # Before encoding: 8 bits per character
    total_bits_before = message_length * 8

    # After encoding: sum of frequency of char * length of its code
    total_bits_after = sum(freq_dict[char] * len(huffman_codes[char]) for char in freq_dict)

    return total_bits_before, total_bits_after

# Example usage
message = "BCCABBDDAECCBBAEDDCC"

# Perform Huffman encoding
encoded_message, huffman_tree_root, huffman_codes, freq_dict = huffman_encoding(message)

# Print Huffman codes for each character
print("Huffman Codes:", huffman_codes)

# Print the encoded message
print("Encoded Message:", encoded_message)

# Calculate total bits before and after encoding
total_bits_before, total_bits_after = calculate_total_bits(freq_dict, huffman_codes, len(message))

# Output total bits before and after
print(f"Total bits before encoding: {total_bits_before} bits")
print(f"Total bits after encoding: {total_bits_after} bits")

# Decode the encoded message to verify correctness
decoded_message = decode_message(encoded_message, huffman_tree_root)
print("Decoded Message:", decoded_message)

# Verify the encoding-decoding process
assert message == decoded_message
