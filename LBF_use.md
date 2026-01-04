# Level-Based Foraging (LBF) Environment Documentation

This document explains the naming conventions and configuration parameters for the Level-Based Foraging (LBF) environment.

## 1. Environment Naming Convention

The environment ID follows a structured format:

`Foraging{Sight}-{Size}x{Size}-{Players}p-{Foods}f{Coop}{Ind}{Pen}-v3`

### Components Breakdown

| Component | Description | Example |
| :--- | :--- | :--- |
| **Prefix** | `Foraging`: Standard environment.<br>`Foraging-2s`: Partial observation (sight range = 2).<br>`Foraging-grid`: Returns grid-based observations. | `Foraging` |
| **Grid Size** | `{Size}x{Size}`: The dimensions of the grid map (Rows x Cols). | `8x8` (8x8 grid) |
| **Players** | `{Players}p`: Number of agents participating. | `2p` (2 agents) |
| **Foods** | `{Foods}f`: Initial number of food items spawned. | `2f` (2 foods) |
| **Suffixes** | `-coop`: **Forced Cooperation**. Foods require multiple agents to collect.<br>`-ind`: **Independent/Index**. Custom max food level settings.<br>`-pen`: **Penalty**. Negative reward for failed load actions. | `-coop-pen` |
| **Version** | `-v3`: Current environment version. | `-v3` |

### Example IDs
- `Foraging-8x8-2p-2f-v3`: Standard 8x8 map, 2 players, 2 foods.
- `Foraging-2s-10x10-3p-3f-coop-v3`: 10x10 map, partial observation (sight=2), 3 players, 3 foods, forced cooperation.

---

## 2. `gym.make` Parameters

When initializing the environment using `gym.make`, you can override specific behaviors using the following parameters. Note that **`gym.make` parameters take precedence over the environment ID**.

### Core Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `players` | `int` | ID-derived | Number of agents. |
| `field_size` | `tuple` | ID-derived | Map dimensions `(rows, cols)`. |
| `max_num_food` | `int` | ID-derived | Maximum number of concurrent food items. |
| `sight` | `int` | Map Size | Agent sight radius. If ID has `-2s`, this is `2`. |
| `max_episode_steps` | `int` | `50` | Maximum steps before truncation. |
| `force_coop` | `bool` | `False` | If `True`, spawns food with levels requiring cooperation. |

### Level & Spawning Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `min_player_level` | `int` | `1` | Minimum spawn level for players. |
| `max_player_level` | `int` | `2` | Maximum spawn level for players. |
| `min_food_level` | `int` | `1` | Minimum spawn level for food. |
| `max_food_level` | `int` | `None` | Maximum spawn level for food. If `None`, usually sum of player levels. |

### Observation & Reward Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `grid_observation` | `bool` | `False` | **`True`**: Returns 3D tensor `(3, H, W)` (Agent, Food, Access layers).<br>**`False`**: Returns feature vector. |
| `normalize_reward` | `bool` | `True` | If `True`, total reward for collecting all food sums to 1.0. |
| `penalty` | `float` | `0.0` | Penalty for failed load actions (e.g., `0.1` if `-pen` is used). |
| `observe_agent_levels` | `bool` | `True` | Whether observations include other agents' levels. |

---

## 3. Usage Example

```python
import gymnasium as gym
import lbforaging

# Create an environment with specific overrides
env = gym.make(
    "Foraging-8x8-2p-2f-v3",
    max_episode_steps=100,
    grid_observation=True,   # Force grid observation
    sight=2                  # Force partial observation
)

obs, info = env.reset()
print("Observation Shape:", obs.shape)
# Output: (2, 3, 8, 8) -> (Agents, Channels, Height, Width)
```
