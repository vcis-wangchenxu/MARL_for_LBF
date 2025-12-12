# env_name
Foraging{Sight}-{Size}x{Size}-{Players}p-{Foods}f{Coop}{Ind}{Pen}-v3
## 1. 基础前缀
1. Foraging: 标准环境的基础名称。
2. Foraging-2s: 如果开启了部分观测 (Partial Observation)，前缀会变成 Foraging-2s。这表示智能体的视野受限（通常视野范围为 2 格）。
3. Foraging-grid: 表示返回基于网格的观测（Grid Observation），而不是特征向量。

## 2. 网格尺寸
1. {Size}x{Size}: 地图的大小。
2. 例如：8x8 表示地图是一个 8 行 8 列的方格世界。

## 3. 玩家数量
1. {Players}p: 参与游戏的智能体（Agent）数量。
2. 例如：2p 表示有 2 个智能体。

## 4. 食物数量
1. {Foods}f: 场上初始生成的食物数量。
2. 例如：2f 表示初始有 2 个食物。

## 5. 可选后缀
1. -coop (Cooperative): 强制合作模式。含义：如果包含此后缀，环境在生成食物时，会强制食物的等级较高，使得单个智能体无法独自收集，必须与其他智能体合作才能收集。代码逻辑：force_coop=True。
2. -ind (Independent/Index): 自定义/独立食物等级。含义：对应代码中的 mfl (max_food_level)。通常用于指定食物等级的具体上限不是默认值，或者有特殊的等级设定。代码逻辑：如果 max_food_level 不为 None，则添加此后缀。
3. -pen (Penalty): 惩罚机制。含义：如果包含此后缀，当智能体尝试收集食物失败（例如等级不足）时，会收到负奖励（惩罚）。代码逻辑：penalty=0.1 (默认无后缀时为 0.0)。

## 6. 版本号
1. -v3: 当前环境的版本号。

# gym.make 参数定义

## 1. 核心参数
这些参数通常由环境 ID 直接确定，但也可以手动修改。\
1. players (int): 含义: 智能体（玩家）的数量。ID对应: Foraging-...-{N}p-... 中的 N。默认值: ID 中指定，通常为 2 到 9。
2. field_size (tuple): 含义: 网格地图的尺寸，格式为 (rows, cols)。ID对应: Foraging-{S}x{S}-... 中的 S (生成 (S, S))。默认值: ID 中指定，通常为 5 到 20。
3. max_num_food (int): 含义: 场上同时存在的最大食物数量。ID对应: Foraging-...-{N}f... 中的 N。说明: 即使食物被吃掉，环境也可能重新生成食物以保持数量（具体生成逻辑视代码而定，LBF通常是吃完一个生成一个或者吃完所有才结束，具体取决于 spawn_food 的调用时机，但在 reset 时会生成这么多）。
4. sight (int): 含义: 智能体的视野范围（半径）。ID对应: 如果是 Foraging-2s-...，则 sight=2（部分观测）。如果是普通 Foraging-...，则 sight 等于地图边长（完全观测）。作用: 决定了观测空间 (Observation Space) 的大小。
5. max_episode_steps (int): 含义: 每个 Episode 的最大步数。默认值: 50。作用: 超过此步数后，环境会返回 truncated=True（或 done=True 取决于 gym 版本兼容性）。
6. force_coop (bool): 含义: 是否强制合作。ID对应: Foraging-...-coop-... 后缀。作用: 如果为 True，生成的食物等级通常会高于单个玩家的最大等级，迫使玩家必须合作才能搬运。

## 2. 等级与生成参数
控制玩家和食物等级的生成逻辑。\
1. min_player_level (int 或 list): 含义: 玩家生成的最低等级。默认值: 1。
2. max_player_level (int 或 list): 含义: 玩家生成的最高等级。默认值: 2。说明: 玩家的实际等级会在 min 和 max 之间随机生成。
3. min_food_level (int 或 list): 含义: 食物生成的最低等级。默认值: 1。
4. max_food_level (int, list 或 None): 含义: 食物生成的最高等级。ID对应: Foraging-...-ind-... 会设置此值。说明: 如果为 None，则最高等级通常动态设定为玩家等级之和（取决于具体生成逻辑）。在默认注册中，如果不带 -ind，此项为 None。

## 3. 观测与奖励参数
这些参数影响智能体接收到的信息和反馈。\
1. grid_observation (bool): 含义: 是否返回网格图像形式的观测。ID对应: Foraging-grid-... 前缀会将此设为 True，否则默认为 False。作用: False: 返回特征向量（包含玩家位置、等级、食物位置等）。True: 返回 3D 张量（类似图像，包含 Agent 层(通道 0): 值为 0（空）或 玩家等级（如 1, 2, 3...）、Food 层(通道 1): 值为 0（空）或 食物等级（如 1, 2, 3...）、Access 层(通道 2): 值为 0（不可通行/越界）或 1（可通行））。
2. normalize_reward (bool): 含义: 是否归一化奖励。默认值: True。 作用: 如果为 True，所有食物被收集完的总奖励和为 1.0。如果不归一化，奖励等于食物的等级。
3. penalty (float): 含义: 失败动作的惩罚值。ID对应: Foraging-...-pen-... 后缀会将其设为 0.1，否则默认为 0.0。作用: 当智能体尝试 LOAD 但失败（等级不够或旁边没食物）时扣除的分数。
4. observe_agent_levels (bool): 含义: 观测中是否包含其他智能体的等级信息。默认值: True。

# Note: make参数优先级高于环境命名
