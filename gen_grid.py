import os
import numpy as np
from tqdm import tqdm

np.random.seed(1234)

class Grid:
	def __init__(self, size):
		self.size = size
		self.grid = [[False for _ in range(self.size)] for _ in range(self.size)]
		for i in range(self.size):
			self.grid[i][0] = self.grid[0][i] = self.grid[i][self.size-1] = self.grid[self.size-1][i] = True

	def check(self, i, j):
		count = 0
		count += self.grid[i][j] == self.grid[i+1][j]
		count += self.grid[i+1][j] == self.grid[i+1][j+1]
		count += self.grid[i+1][j+1] == self.grid[i][j+1]
		count += self.grid[i][j+1] == self.grid[i][j]
		return count == 2

	def check_connected(self):
		flag = [[False for _ in range(self.size)] for _ in range(self.size)]
		# for i in range(self.size):
		# 	flag[i][0] = flag[0][i] = flag[i][self.size-1] = flag[self.size-1][i] = True
		i = 1 
		j = 1 
		while i < self.size-1 and j < self.size-1 and self.grid[i][j]:
			i += 1 
			if i == self.size-1:
				i = 1
				j += 1
		# found empty tile
		flag[i][j] = True
		tiles = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]
		while tiles:
			i, j = tiles[0]
			tiles = tiles[1:]
			if self.grid[i][j] or flag[i][j]:
				continue
			else:
				flag[i][j] = True
				tiles.extend([(i+1, j), (i, j+1), (i-1, j), (i, j-1)])
		# for x in range(self.size):
		# 	for y in range(self.size):
		# 		if flag[x][y]:
		# 			print('*\t', end='')
		# 		else:
		# 			print(' \t', end='')
		# 	print()
		# print()
		# check for second component
		i = 1 
		j = 1
		while i < self.size-1 and j < self.size-1 and (self.grid[i][j] or flag[i][j]):
			i += 1 
			if i == self.size-1:
				i = 1
				j += 1
		# print(i, j)
		# if i != self.size-1:
		if i < self.size-1 and j < self.size-1 and not self.grid[i][j] and not flag[i][j]:
			return False
		return True

	def done(self):
		for i in range(self.size-1):
			for j in range(self.size-1):
				if not self.check(i, j):
					return False
		return True

	def place(self):
		# get a free tile
		x = np.random.choice(range(1, self.size-1))
		y = np.random.choice(range(1, self.size-1))
		while self.grid[x][y]:
			x = np.random.choice(range(1, self.size-1))
			y = np.random.choice(range(1, self.size-1))

		# try to place
		self.grid[x][y] = True
		if not all([self.check(x, y), self.check(x-1,y), self.check(x,y-1), self.check(x-1,y-1), self.check_connected()]):
		# if not all(self.check(x, y), self.check(x+1,y), self.check(x,y+1), self.check(x+1,y+1)):
			self.grid[x][y] = False
			return self.place()
		else:
			return x, y

	def generate(self):
		count = 0
		while not self.done() or count < 8:
			try:
				if count == 8:
					# import pdb; pdb.set_trace()
					break
				x, y = self.place()
				count += 1
			except RecursionError:
				return False
		# self.grid[x][y] = False
		return True


	def print(self):
		for i in range(self.size):
			for j in range(self.size):
				if self.grid[i][j]:
					print('*\t', end='')
				else:
					print(' \t', end='')
			print()
		print()

	def build_tree(self):
		flag = [[False for _ in range(self.size)] for _ in range(self.size)]
		deg = [[0 for _ in range(self.size)] for _ in range(self.size)]
		neighbors = [[[] for _ in range(self.size)] for _ in range(self.size)]
		i = 1 
		j = 1 
		while i < self.size-1 and j < self.size-1 and self.grid[i][j]:
			i += 1 
			if i == self.size-1:
				i = 1
				j += 1
		# found empty tile
		flag[i][j] = True
		tiles = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]
		prev = [(i, j), (i, j), (i, j), (i, j)]
		while tiles:
			i, j = tiles[0]
			iprev, jprev = prev[0]
			tiles = tiles[1:]
			prev = prev[1:]
			if self.grid[i][j] or flag[i][j]:
				continue
			else:
				# add an edge between (i, j) and (iprev, jprev)
				neighbors[i][j].append((iprev, jprev))
				neighbors[iprev][jprev].append((i, j))
				deg[i][j] += 1
				deg[iprev][jprev] += 1
				flag[i][j] = True
				tiles.extend([(i+1, j), (i, j+1), (i-1, j), (i, j-1)])
				prev.extend([(i, j), (i, j), (i, j), (i, j)])

		# for x in range(self.size):
		# 	for y in range(self.size):
		# 		if flag[x][y]:
		# 			print(f'{deg[x][y]}\t', end='')
		# 		else:
		# 			print(' \t', end='')
		# 	print()
		# print()

		self.flag = flag
		self.deg = deg 
		self.neighbors = neighbors

	def place_objects(self):
		flag = self.flag 
		deg = self.deg 
		# neighbors = self.neighbors

		max_deg = 0
		locs = []
		end_locs = []
		for x in range(self.size):
			for y in range(self.size):
				if flag[x][y]:
					if deg[x][y] > max_deg:
						locs = [(x, y)]
						max_deg = deg[x][y]
					elif deg[x][y] == max_deg:
						locs.append((x, y))
					if deg[x][y] == 1:
						end_locs.append((x, y))

		# print(locs)
		ball_loc = locs[np.random.choice(np.arange(len(locs)))]

		self.grid[ball_loc[0]][ball_loc[1]] = True

		flag2 = [[False for _ in range(self.size)] for _ in range(self.size)]
		i = 1 
		j = 1 
		while i < self.size-1 and j < self.size-1 and self.grid[i][j]:
			i += 1 
			if i == self.size-1:
				i = 1
				j += 1
		# found empty tile
		flag2[i][j] = True
		tiles = [(i+1, j), (i, j+1), (i-1, j), (i, j-1)]
		while tiles:
			i, j = tiles[0]
			tiles = tiles[1:]
			if self.grid[i][j] or flag2[i][j]:
				continue
			else:
				flag2[i][j] = True
				tiles.extend([(i+1, j), (i, j+1), (i-1, j), (i, j-1)])


		self.grid[ball_loc[0]][ball_loc[1]] = False

		same_component = np.random.random() < 0.5
		poss_locs = []
		goal_locs = []
		for loc in end_locs:
			x, y = loc 
			if same_component and flag2[x][y]:
				poss_locs.append((x, y))
			elif not same_component and not flag2[x][y]:
				poss_locs.append((x, y))
			if same_component and not flag2[x][y]:
				goal_locs.append((x, y))
			elif not same_component and flag2[x][y]:
				goal_locs.append((x, y))

		caregiver_loc = poss_locs[np.random.choice(np.arange(len(poss_locs)))]
		goal_loc = goal_locs[np.random.choice(np.arange(len(goal_locs)))]

		locs = []
		for loc in goal_locs:
			if loc != caregiver_loc and loc != goal_loc:
				locs.append(loc)

		agent_loc = locs[np.random.choice(np.arange(len(locs)))]

		self.full_grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]

		for i in range(self.size):
			for j in range(self.size):
				if self.grid[i][j]:
					self.full_grid[i][j] = '*'
				if (i, j) == agent_loc:
					self.full_grid[i][j] = 'A'
				if (i, j) == ball_loc:
					self.full_grid[i][j] = 'B'
				if (i, j) == caregiver_loc:
					self.full_grid[i][j] = 'C'
				if (i, j) == goal_loc:
					self.full_grid[i][j] = 'X'

		# for x in range(self.size):
		# 	for y in range(self.size):
		# 		print(f'{self.full_grid[x][y]}\t', end='')
		# 	print()
		# print()

		self.agent_loc = agent_loc
		self.ball_loc = ball_loc
		self.caregiver_loc = caregiver_loc
		self.goal_loc = goal_loc
		agent_start_dir = np.random.choice(4)
		caregiver_start_dir = np.random.choice(4)
		self.agent_start_dir = agent_start_dir
		self.caregiver_start_dir = caregiver_start_dir

	def solve(self):
		DIR_TO_VEC = [
			# Pointing right (positive X)
			np.array((1, 0)),
			# Down (positive Y)
			np.array((0, 1)),
			# Pointing left (negative X)
			np.array((-1, 0)),
			# Up (negative Y)
			np.array((0, -1)),
		]

		VEC_TO_DIR = {(x[0], x[1]): i for i, x in enumerate(DIR_TO_VEC)}
		
		neighbors = self.neighbors

		start_loc = self.caregiver_loc
		end_loc = self.goal_loc

		# actions = []

		# path = []

		# flag = [[False for _ in range(self.size)] for _ in range(self.size)]

		# DFS
		# cur_loc = start_loc 
		def dfs(cur_loc, path):
			if cur_loc == end_loc:
				return path
			for neighbor in neighbors[cur_loc[0]][cur_loc[1]]:
				if neighbor in path: continue
				res = dfs(neighbor, path + [neighbor])
				if res:
					return res
			return []

		path = dfs(start_loc, [])

		# print(path)

		actions = []

		# turns
		prev_loc = path[0]
		prev_dir = self.caregiver_start_dir
		for cur_loc in path[1:]:
			vec = (cur_loc[0]-prev_loc[0], cur_loc[1]-prev_loc[1])
			cur_dir = VEC_TO_DIR[vec]
			if cur_dir != prev_dir:
				# add turns 
				actions.append(1) # TODO
				actions.append(2)
			else:
				actions.append(2)
			prev_dir = cur_dir
			prev_loc = cur_loc

		# print(actions)

	def to_string(self, newline=True):
		s = ''
		for i in range(self.size):
			for j in range(self.size):
				s += f'{self.full_grid[i][j]}'
			if newline:
				s += '\n'
		if newline:
			s += '\n'
		return s

	def write(self, f):
		f.write(self.to_string())

	def write_all(self, f):
		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.write(f)

		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.write(f)

		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.write(f)

		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.full_grid = np.flip(self.full_grid, axis=1)
		self.write(f)

		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.write(f)

		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.write(f)

		self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
		self.write(f)

	def gen_all(self):
		strings = set()
		for i in range(10):
			try:
				self.place_objects()

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				strings.add(self.to_string())

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				strings.add(self.to_string())

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				strings.add(self.to_string())

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				self.full_grid = np.flip(self.full_grid, axis=1)
				strings.add(self.to_string())

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				strings.add(self.to_string())

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				strings.add(self.to_string())

				self.full_grid = np.rot90(self.full_grid, k=1, axes=(1,0))
				strings.add(self.to_string())
			except:
				pass
		return strings

	def read(self, f):
		lines = []
		self.full_grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
		for i in range(self.size):
			line = f.readline()
			lines.append(line)
			for j, char in enumerate(line[:-1]):
				self.full_grid[i][j] = char

				if char == 'A':
					self.agent_loc = (i, j)
				elif char == 'B':
					self.ball_loc = (i, j)
				elif char == 'C':
					self.caregiver_loc = (i, j)
				elif char == 'X':
					self.goal_loc = (i, j)

				if char == '*':
					self.grid[i][j] = True
				else:
					self.grid[i][j] = False

		# print(self.to_string())



    # partial_grid = np.rot90(partial_grid, k=1, axes=(1,0))
    # partial_grid = np.flip(partial_grid, axis=1).copy()


# class SimpleEnv(MiniGridEnv):
#     def __init__(
#         self,
#         size=7,
#         agent_view_size=13,
#         agent_start_pos=(1, 1),
#         agent_start_dir=0,
#         max_steps: int | None = None,
#         reward_model=None,
#         goal_pos=None, # (3, 3),
#         n_walls=0,
#         **kwargs,
#     ):
#         self.agent_start_pos = agent_start_pos
#         self.agent_start_dir = agent_start_dir
#         self.goal_pos = goal_pos
#         self.n_walls = n_walls

#         mission_space = MissionSpace(mission_func=self._gen_mission)

#         if max_steps is None:
#             max_steps = 4 * (size-2)**2

#         super().__init__(
#             mission_space=mission_space,
#             grid_size=size,
#             # agent_view_size=agent_view_size,
#             # Set this to True for maximum speed
#             see_through_walls=True,
#             max_steps=max_steps,
#             **kwargs,
#         )
#         # Allow only 3 actions permitted: left, right, forward
#         self.action_space = spaces.Discrete(self.actions.forward + 1)

#         self.reward_mode = 'default' # 'reward_model' 'none'

#         # self.set_reward_model()

#     # def set_reward_model(self):
#         if reward_model is not None:
#             self.reward_model = reward_model
#         # else:
#         #     self.reward_model = RewardModel()
#         # self.goals_collected = 0
#         # self.should_terminate = True

#     @staticmethod
#     def _gen_mission():
#         return "Coin game"

#     def _gen_grid(self, width, height):
#         # Create an empty grid
#         self.grid = Grid(width, height)

#         # Generate the surrounding walls
#         self.grid.wall_rect(0, 0, width, height)

#         # # Generate verical separation wall
#         # for i in range(0, height):
#         #     self.grid.set(5, i, Wall())
		
#         # # Place the door and key
#         # self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
#         # self.grid.set(3, 6, Key(COLOR_NAMES[0]))

#         for i in range(self.n_walls):
#             # Place a wall square in the bottom-right corner
#             x = np.random.choice(range(1, width-1))
#             y = np.random.choice(range(1, height-1))
#             while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
#                 x = np.random.choice(range(1, width-1))
#                 y = np.random.choice(range(1, height-1))
#             # for j in range(0, 2):
#             #     for k in range(0, 2):
#             #         self.put_obj(Lava(), x+j, y+k)
#             self.put_obj(Wall(), x, y)

#         for i in range(0):
#             # Place a lava square in the bottom-right corner
#             x = np.random.choice(range(1, width-1))
#             y = np.random.choice(range(1, height-1))
#             while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
#                 x = np.random.choice(range(1, width-1))
#                 y = np.random.choice(range(1, height-1))
#             # for j in range(0, 2):
#             #     for k in range(0, 2):
#             #         self.put_obj(Lava(), x+j, y+k)
#             self.put_obj(Lava(), x, y)

#         if self.goal_pos is None:
#             for i in range(1):
#                 # Place a goal square in the bottom-right corner
#                 x = np.random.choice(range(1, width-1))
#                 y = np.random.choice(range(1, height-1))
#                 # y = np.random.choice(range(1, height//2+1))
#                 while self.grid.get(x, y) or (x, y) == self.agent_start_pos:
#                     x = np.random.choice(range(1, width-1))
#                     y = np.random.choice(range(1, height-1))
#                 self.put_obj(Goal(), x, y)
#         else:
#             # self.put_obj(Goal(), width - 2, height - 2)
#             self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])
#         # self.target = np.array([x, y])

#         # r = np.random.random()
#         # if r < 1./3:
#         # self.put_obj(Goal(), width - 2, height - 2)
#         # elif r < 2./3:
#         #     self.put_obj(Goal(), width // 2 - 1, height // 2 - 1)
#         # else:
#         #     self.put_obj(Goal(), 1, height // 2 - 1)

#         # x = np.random.choice(range(1, width-1))
#         # y = np.random.choice(range(1, height-1))
#         # while self.grid.get(x, y):
#         #     x = np.random.choice(range(1, width-1))
#         #     y = np.random.choice(range(1, height-1))

#         # Place the agent
#         # if self.agent_start_pos is not None:
#         # self.agent_pos = (x, y)
#         self.agent_pos = self.agent_start_pos
#         self.agent_dir = self.agent_start_dir
#         # self.agent_dir = np.random.choice(4)
#         # else:
#         #     self.place_agent()

#         # self.mission = f"Coin game {self.target[0]} {self.target[1]}"
#         self.mission = "Coin game"

#     def _reward(self) -> float:
#         """
#         Compute the reward to be given upon success
#         """
#         return 1 - 0.9 * (self.step_count / self.max_steps)

#     def step(
#         self, action: ActType
#     ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
#         self.step_count += 1
#         self.action = action
#         prev_dir = self.agent_dir
#         self.prev_dir = prev_dir

#         reward = 0
#         terminated = False
#         truncated = False

#         # Get the position in front of the agent
#         fwd_pos = self.front_pos

#         # Get the contents of the cell in front of the agent
#         fwd_cell = self.grid.get(*fwd_pos)
#         self.fwd_cell = fwd_cell

#         # Rotate left
#         if action == self.actions.left:
#             self.agent_dir -= 1
#             if self.agent_dir < 0:
#                 self.agent_dir += 4

#         # Rotate right
#         elif action == self.actions.right:
#             self.agent_dir = (self.agent_dir + 1) % 4

#         # Move forward
#         elif action == self.actions.forward:
#             # reward = -0.9 * (1. / self.max_steps) # 1. / (self.width + self.height) # 
#             if fwd_cell is None or fwd_cell.can_overlap():
#                 self.agent_pos = tuple(fwd_pos)
#             if fwd_cell is not None and fwd_cell.type == "goal":
#                 # self.grid.set(fwd_pos[0], fwd_pos[1], None)

#                 # # Place new goal
#                 # x = np.random.choice(range(1, self.width-1))
#                 # y = np.random.choice(range(1, self.height-1))
#                 # while self.grid.get(x, y):
#                 #     x = np.random.choice(range(1, self.width-1))
#                 #     y = np.random.choice(range(1, self.height-1))
#                 # self.put_obj(Goal(), x, y)

#                 # self.goals_collected += 1
#                 # if self.goals_remaining == 0:
#                 # terminated = self.should_terminate
#                 terminated = True
#                 # terminated = False
#                 # if fwd_pos[0] == self.target[0] \
#                 # and fwd_pos[1] == self.target[1]:
#                 # reward += self._reward() #/ self.num_goals
#                 # else:
#                 #     reward = -self._reward()
#                 reward = self._reward()
#             if fwd_cell is not None and fwd_cell.type == "lava":
#                 terminated = True

#         # # Pick up an object
#         # elif action == self.actions.pickup:
#         #     if fwd_cell and fwd_cell.can_pickup():
#         #         if self.carrying is None:
#         #             self.carrying = fwd_cell
#         #             self.carrying.cur_pos = np.array([-1, -1])
#         #             self.grid.set(fwd_pos[0], fwd_pos[1], None)

#         # # Drop an object
#         # elif action == self.actions.drop:
#         #     if not fwd_cell and self.carrying:
#         #         self.grid.set(fwd_pos[0], fwd_pos[1], self.carrying)
#         #         self.carrying.cur_pos = fwd_pos
#         #         self.carrying = None

#         # # Toggle/activate an object
#         # elif action == self.actions.toggle:
#         #     if fwd_cell:
#         #         fwd_cell.toggle(self, fwd_pos)

#         # Done action (not used by default)
#         elif action == self.actions.done:
#             pass

#         else:
#             raise ValueError(f"Unknown action: {action}")

#         if self.step_count >= self.max_steps:
#             truncated = True

#         if self.render_mode == "human":
#             self.render()

#         # if self.reward_mode == 'reward_model':
#         #     reward = self._reward()

#         if hasattr(self, 'reward_model') and self.reward_model.train:
#             self.reward_model.observe(self.obs, action, reward) 

#         obs = self.gen_obs()
#         return obs, reward, terminated, truncated, {}

def main():
	counter = 0
	# for i in tqdm(range(1000)):
	# 	np.random.seed(i)
	# 	grid = Grid(7)
	# 	success = grid.generate()
	# 	if success: counter += 1 #grid.print()
	# print(counter)
	grids = set()
	for i in tqdm(range(5000)):
		grid = Grid(7)
		success = grid.generate()
		# if success: grid.print()
		if success: grid.build_tree()
		# if success: grid.place_objects()
		# if success: grid.write(f)
		if success:
			strings = grid.gen_all()
			grids = grids.union(strings)
		if success: counter += 1
	print(counter) # 2836
	# print(len(grids)) # 1000: 12sec, 25451
	print(len(grids)) # 5000: 69sec, 62358
	# 5000: 66sec, 42016
	with open("file.txt", "w") as f:
		for s in grids:
			f.write(s)
		f.write('\n')
	# if success: grid.solve()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# if success: grid.build_tree()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()

	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()
	# grid = Grid(7)
	# success = grid.generate()
	# if success: grid.print()

def read():
	f = open("file.txt", "r")
		# lines = f.readlines()
	lenbytes = os.path.getsize("file.txt")
	num_grids = (lenbytes-1) // 57
	grid_idx = np.random.choice(num_grids)
	print(grid_idx)
	f.seek(grid_idx*((7+1)*7+1))
	grid = Grid(7)
	grid.read(f)
	# lines = []
	# for i in range(7):
	# 	line = f.readline()
	# 	lines.append(line)
	f.close()
	# for line in lines:
	# 	print(line)


if __name__ == '__main__':
	# main()
	read()