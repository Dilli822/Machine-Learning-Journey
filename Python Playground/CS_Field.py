import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 10))

# Draw the main big circle for Computer Science
main_circle = patches.Circle((0.5, 0.5), 0.4, edgecolor='black', facecolor='none', lw=2)
ax.add_patch(main_circle)

# Core fields with their positions (x, y), sizes, and labels
fields = [
    {"label": "Artificial Intelligence", "pos": (0.65, 0.7), "size": 0.1},
    {"label": "Applied Mathematics\n- Linear Algebra\n- Calculus\n- Number Theory", "pos": (0.3, 0.7), "size": 0.1},
    {"label": "Software and\nSystem Architecture", "pos": (0.7, 0.5), "size": 0.08},
    {"label": "Web Development", "pos": (0.5, 0.5), "size": 0.07},
    {"label": "Networking", "pos": (0.5, 0.3), "size": 0.05},
    {"label": "Database", "pos": (0.4, 0.4), "size": 0.06},
    {"label": "Cyber Security", "pos": (0.3, 0.3), "size": 0.07},
    {"label": "Cloud Computing", "pos": (0.2, 0.5), "size": 0.07},
]

# Draw each field as a circle with the label inside it
for field in fields:
    circle = patches.Circle(field["pos"], field["size"], edgecolor='black', facecolor='none')
    ax.add_patch(circle)
    ax.text(field["pos"][0], field["pos"][1], field["label"], 
            ha='center', va='center', fontsize=6, wrap=True)

# Add the main title inside the big circle
plt.text(0.5, 0.85, "Computer Science", ha='center', va='center', fontsize=14, fontweight='bold')

# Set up the plot limits, grid, and hide axes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')

# Display the plot
plt.show()
