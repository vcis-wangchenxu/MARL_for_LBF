# env_name
Foraging{Sight}-{Size}x{Size}-{Players}p-{Foods}f{Coop}{Ind}{Pen}-v3
## 1. Basic Prefix
1. Foraging: The base name of the standard environment.
2. Foraging-2s: If partial observation is enabled, the prefix becomes Foraging-2s. This means agents have a limited field of view (typically a sight range of 2 tiles).
3. Foraging-grid: Indicates the environment returns grid-based observations (Grid Observation) instead of a feature vector.

## 2. Grid Size
1. {Size}x{Size}: The map size.
2. Example: 8x8 means the map is an 8-row by 8-column grid world.

## 3. Number of Players
1. {Players}p: The number of agents (players) participating in the game.
2. Example: 2p means there are 2 agents.

## 4. Number of Foods
1. {Foods}f: The number of foods initially spawned on the field.
2. Example: 2f means there are initially 2 foods.

## 5. Optional Suffixes
1. -coop (Cooperative): Forced cooperation mode. Meaning: If this suffix is included, the environment will force higher-level foods when generating foods, so that a single agent cannot collect them alone and must cooperate with other agents. Code logic: force_coop=True.
2. -ind (Independent/Index): Custom/independent food level setting. Meaning: Corresponds to mfl (max_food_level) in the code. Typically used when specifying a non-default upper bound for food level or using special level settings. Code logic: if max_food_level is not None, this suffix is added.
3. -pen (Penalty): Penalty mechanism. Meaning: If this suffix is included, when an agent fails to load/collect food (e.g., insufficient level), it receives a negative reward (penalty). Code logic: penalty=0.1 (default is 0.0 without this suffix).

## 6. Version
1. -v3: The current environment version.

# gym.make parameters

## 1. Core Parameters
These parameters are usually determined directly by the environment ID, but can also be manually overridden.\
1. players (int): Meaning: Number of agents (players). ID mapping: N in Foraging-...-{N}p-.... Default: specified by the ID, typically 2 to 9.
2. field_size (tuple): Meaning: Grid map size in the format (rows, cols). ID mapping: S in Foraging-{S}x{S}-... (creates (S, S)). Default: specified by the ID, typically 5 to 20.
3. max_num_food (int): Meaning: Maximum number of foods present on the field at the same time. ID mapping: N in Foraging-...-{N}f.... Note: Even if foods are eaten, the environment may respawn foods to keep the count (exact logic depends on the code; in LBF it is often “spawn one after one is eaten” or “end after all are eaten”, depending on when spawn_food is called, but reset will spawn this many initially).
4. sight (int): Meaning: Agent sight range (radius). ID mapping: If the ID is Foraging-2s-..., then sight=2 (partial observation). If it is plain Foraging-..., sight equals the map side length (full observation). Effect: Determines the size of the observation space.
5. max_episode_steps (int): Meaning: Maximum steps per episode. Default: 50. Effect: After exceeding this limit, the environment returns truncated=True (or done=True depending on gym version compatibility).
6. force_coop (bool): Meaning: Whether to force cooperation. ID mapping: Foraging-...-coop-... suffix. Effect: If True, spawned foods usually have levels higher than a single player's max level, forcing cooperation to carry.

## 2. Level & Spawning Parameters
These parameters control how player and food levels are generated.\
1. min_player_level (int or list): Meaning: Minimum player level at spawn. Default: 1.
2. max_player_level (int or list): Meaning: Maximum player level at spawn. Default: 2. Note: Actual player levels are sampled randomly between min and max.
3. min_food_level (int or list): Meaning: Minimum food level at spawn. Default: 1.
4. max_food_level (int, list, or None): Meaning: Maximum food level at spawn. ID mapping: Foraging-...-ind-... sets this value. Note: If None, the maximum level is usually set dynamically as the sum of player levels (depending on the exact spawning logic). In the default registration, this is None when -ind is not present.

## 3. Observation & Reward Parameters
These parameters affect what information and feedback the agents receive.\
1. grid_observation (bool): Meaning: Whether to return grid-image-style observations. ID mapping: The Foraging-grid-... prefix sets this to True; otherwise it defaults to False. Effect: False: returns a feature vector (including player positions, levels, food positions, etc.). True: returns a 3D tensor (image-like) containing an Agent layer (channel 0): 0 (empty) or player level (1, 2, 3...), a Food layer (channel 1): 0 (empty) or food level (1, 2, 3...), and an Access layer (channel 2): 0 (blocked/out of bounds) or 1 (passable).
2. normalize_reward (bool): Meaning: Whether to normalize rewards. Default: True. Effect: If True, the total reward summed over collecting all foods equals 1.0. If not normalized, the reward equals the food level.
3. penalty (float): Meaning: Penalty value for failed actions. ID mapping: The Foraging-...-pen-... suffix sets it to 0.1; otherwise it defaults to 0.0. Effect: Deducted when an agent attempts LOAD but fails (insufficient level or no adjacent food).
4. observe_agent_levels (bool): Meaning: Whether observations include other agents' level information. Default: True.

# Note: gym.make parameters take precedence over environment naming
