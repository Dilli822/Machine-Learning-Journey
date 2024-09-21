import matplotlib.pyplot as plt
import numpy as np

# Number of points
n_points = 10

# Generate spherical coordinates
phi = np.linspace(0, 2 * np.pi, n_points)
theta = np.linspace(0, np.pi, n_points)
phi, theta = np.meshgrid(phi, theta)
phi = phi.flatten()
theta = theta.flatten()

# Convert spherical coordinates to Cartesian coordinates
x = 15 * np.sin(theta) * np.cos(phi)  # Increase the radius
y = 15 * np.sin(theta) * np.sin(phi)  # Increase the radius
z = 15 * np.cos(theta)  # Increase the radius

# Generate a color map
colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(x)))

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter the points with larger size and color map
scatter = ax.scatter(x, y, z, c=colors, s=100, alpha=0.8)

# Annotate each point with its x-coordinate (or y or z, as needed)
for i in range(len(x)):
    ax.text(x[i], y[i], z[i], f'{x[i]:.1f}', fontsize=14, color='black')

# Show the grid lines
ax.grid(True)

# Show the x, y, and z axis lines
ax.xaxis.pane.set_visible(True)
ax.yaxis.pane.set_visible(True)
ax.zaxis.pane.set_visible(True)

# Show the ticks and labels
ax.set_xticks(np.linspace(-20, 20, 5))
ax.set_yticks(np.linspace(-20, 20, 5))
ax.set_zticks(np.linspace(-20, 20, 5))

# Optionally, you can keep or set the axis labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set the limits to extend the length of the axes
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-20, 20])

plt.show()


# import matplotlib.pyplot as plt
# import numpy as np

# def is_within_octagon(x, y, radius):
#     """Check if the point (x, y) is within an octagon."""
#     octagon_vertices = np.array([
#         [radius, 0],
#         [radius/np.sqrt(2), radius/np.sqrt(2)],
#         [0, radius],
#         [-radius/np.sqrt(2), radius/np.sqrt(2)],
#         [-radius, 0],
#         [-radius/np.sqrt(2), -radius/np.sqrt(2)],
#         [0, -radius],
#         [radius/np.sqrt(2), -radius/np.sqrt(2)]
#     ])
    
#     def distance_to_point(p1, p2):
#         return np.linalg.norm(p1 - p2)

#     angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
#     octagon_edges = np.array([
#         [radius * np.cos(angles[i]), radius * np.sin(angles[i])] for i in range(8)
#     ])

#     return all(distance_to_point(np.array([x, y]), edge) <= radius / np.sqrt(2) for edge in octagon_edges)

# # Number of points
# n_points = 70

# # Generate random spherical coordinates
# phi = np.linspace(0, 2 * np.pi, n_points)
# theta = np.linspace(0, np.pi, n_points)
# phi, theta = np.meshgrid(phi, theta)
# phi = phi.flatten()
# theta = theta.flatten()

# # Convert spherical coordinates to Cartesian coordinates
# x = 15 * np.sin(theta) * np.cos(phi)
# y = 15 * np.sin(theta) * np.sin(phi)
# z = 15 * np.cos(theta)

# # Filter points within the octagonal bounds
# radius = 15
# filtered_indices = [i for i in range(len(x)) if is_within_octagon(x[i], y[i], radius)]
# x_filtered = x[filtered_indices]
# y_filtered = y[filtered_indices]
# z_filtered = z[filtered_indices]

# # Generate a color map
# colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(x_filtered)))

# # Create a 3D scatter plot
# fig = plt.figure(figsize=(16, 12))
# ax = fig.add_subplot(111, projection='3d')

# # Scatter the points with larger size and color map
# scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c=colors, s=100, alpha=0.8)

# # Annotate each point with its x-coordinate (or y or z, as needed)
# for i in range(len(x_filtered)):
#     ax.text(x_filtered[i], y_filtered[i], z_filtered[i], f'{x_filtered[i]:.1f}', fontsize=14, color='black')

# # Hide the grid lines
# ax.grid(False)

# # Hide the x, y, and z axis lines
# ax.xaxis.pane.set_visible(False)
# ax.yaxis.pane.set_visible(False)
# ax.zaxis.pane.set_visible(False)

# # Hide the ticks and labels
# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_zticks([])

# # Set the limits to extend the length of the axes
# ax.set_xlim([-20, 20])
# ax.set_ylim([-20, 20])
# ax.set_zlim([-20, 20])

# # Optionally, remove the axis labels
# ax.set_xlabel('')
# ax.set_ylabel('')
# ax.set_zlabel('')

# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Number of points per layer
n_points_per_layer = 10

# Number of layers
n_layers = 10

# Create an empty list for coordinates
x = []
y = []
z = []

# Generate triangular lattice coordinates
for layer in range(n_layers):
    # Create a triangular grid in the xy-plane for each z layer
    z_layer = np.full(n_points_per_layer, layer * 2)  # Set z-coordinate
    for i in range(n_points_per_layer):
        angle = 2 * np.pi * i / n_points_per_layer
        radius = layer * 2  # Increase radius with layer
        x.append(radius * np.cos(angle))
        y.append(radius * np.sin(angle))
        z.append(z_layer[i])

# Convert lists to numpy arrays
x = np.array(x)
y = np.array(y)
z = np.array(z)

# Generate a color map
colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(x)))

# Create a 3D scatter plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Create a colorful background surface
x_bg, y_bg = np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100))
z_bg = np.zeros_like(x_bg)
colors_bg = plt.get_cmap('plasma')(np.linspace(0, 1, len(x_bg.flatten())))

# Plot the background surface
ax.plot_surface(x_bg, y_bg, z_bg, facecolors=colors_bg.reshape(x_bg.shape[0], x_bg.shape[1], -1), alpha=0.5, zorder=-1)

# Scatter the points with larger size and color map
scatter = ax.scatter(x, y, z, c=colors, s=100, alpha=0.8)

# Annotate each point with its coordinates
for i in range(len(x)):
    ax.text(x[i], y[i], z[i], f'({x[i]:.1f}, {y[i]:.1f}, {z[i]:.1f})', fontsize=8, color='black')

# Show the grid lines
ax.grid(True)

# Show the x, y, and z axis lines
ax.xaxis.pane.set_visible(True)
ax.yaxis.pane.set_visible(True)
ax.zaxis.pane.set_visible(True)

# Show the ticks and labels
ax.set_xticks(np.linspace(-20, 20, 5))
ax.set_yticks(np.linspace(-20, 20, 5))
ax.set_zticks(np.linspace(-20, 20, 5))

# Set axis labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set the limits to extend the length of the axes
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([0, 20])

plt.show()
