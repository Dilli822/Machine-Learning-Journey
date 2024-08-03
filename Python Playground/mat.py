import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [4, -5, 16, 7, 28]

# Create a plot
plt.plot(x, y, marker="o", linestyle=":", color='blue')
# Add title and labels
plt.title("Basic Text Example")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")

# Add text to the plot in data coordinates
# plt.text(3, 6, 'Data Text', fontsize=12, color='green')

# # Add text to the plot in axes coordinates
# plt.text(0.5, 0.1, 'Axes Text', fontsize=12, color='red', ha='center', va='bottom', transform=plt.gca().transAxes)

# Display the plot
plt.grid(False)
plt.show()
