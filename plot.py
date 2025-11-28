import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Parse JSON
performance_data = json.load(open("tournament_result.json", "r"))

agents = list(performance_data.keys())

# Calculate average success rate for each agent (excluding self-play)
agent_success = {}
for agent in agents:
    wins = [performance_data[agent][opponent] for opponent in performance_data[agent] if opponent != agent]
    agent_success[agent] = np.mean(wins)

# Sort agents by success rate (descending)
sorted_agents = sorted(agents, key=lambda x: -agent_success[x])

# Create matrix with sorted agents
num_agents = len(sorted_agents)
heatmap_data = np.zeros((num_agents, num_agents))

for i, agent1 in enumerate(sorted_agents):
    for j, agent2 in enumerate(sorted_agents):
        if agent1 == agent2:
            heatmap_data[i, j] = np.nan
        else:
            heatmap_data[i, j] = performance_data[agent1].get(agent2, np.nan)

# Create custom colormap (red to green)
colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red -> Yellow -> Green
cmap = LinearSegmentedColormap.from_list("rg", colors, N=256)

# Create figure
plt.figure(figsize=(10, 10))

# Create heatmap
plt.imshow(heatmap_data, cmap=cmap, vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar()
cbar.set_label('Win Rate', rotation=270, labelpad=20)

# Add agent names as ticks
plt.xticks(np.arange(num_agents), sorted_agents, rotation=45, ha='right')
plt.yticks(np.arange(num_agents), sorted_agents)

# Add grid lines
plt.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
plt.tick_params(which='minor', length=0)

# Add values in cells
for i in range(num_agents):
    for j in range(num_agents):
        if not np.isnan(heatmap_data[i, j]):
            plt.text(j, i, f"{heatmap_data[i, j]:.1f}", 
                    ha="center", va="center", color="black")

# Title and labels
plt.title('Agent Performance Heatmap (Sorted by Success Rate)', pad=20)
plt.xlabel('Opponent Agent')
plt.ylabel('Main Agent (Sorted)')

# Add success rates to y-axis labels
yticklabels = []
for agent in sorted_agents:
    yticklabels.append(f"{agent}\n({agent_success[agent]:.1%})")
plt.yticks(np.arange(num_agents), yticklabels)

# Adjust layout
plt.tight_layout()

# Save as PNG
plt.savefig('sorted_agent_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("Sorted heatmap saved as 'sorted_agent_performance_heatmap.png'")