import gym
import numpy as np
from gym import spaces
from tianshou.env import DummyVectorEnv
from tianshou.env import DummyVectorEnv
import matplotlib.pyplot as plt

class RouteEnv(gym.Env):
    """Environment with a number line containing multiple Wille nodes and complex channel selection."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, number_line_length=20):
        super().__init__()
        self.number_line_length = number_line_length
        self.num_willes = 3
        self.num_channels = 9
        self.sigma = 1
        self._flag = 0
        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self._steps_per_episode = 1

        # Available channel structure: N * N * 9 (three-dimensional array)
        self.available_channel = [1, 2, 3]

        # Channel parameters structure
        self.channel_param = np.zeros((number_line_length, self.num_channels),
                                      dtype=[('energy_level', 'f4'), ('covert_rate', 'f4')])
        self._initialize_channels()

        # Simplified action and observation spaces
        self.action_space = spaces.Discrete(self.num_channels)
        self.observation_space = spaces.Box(
            low=0,
            high=1000,
            shape=(11,),  # position, channel_type, energy_level, covert_rate + 7 additional features
            dtype=np.float32
        )

        # Initialize the number line with Wille nodes
        self.wille_position = self._initialize_wille_position()

        self.current_position = 0
        self.previous_channel_type = 0


    def _initialize_channels(self):
        # Initialize channel parameters with random values
        for i in range(self.number_line_length):
            for j in range(self.num_channels):
                self.channel_param[i, j] = (np.random.uniform(5, 10),  # energy_level
                                            0  # covert_rate
                                            )
            for j in range(0, self.num_channels, self.num_willes):
                self.channel_param[i, j:j + 3][1] = np.random.uniform(100, 200)  # covert_rate

    def _initialize_wille_position(self):
        # Initialize the number line with Wille nodes at random positions
        wille_position = []
        for _ in range(self.num_willes):
            wille = np.random.uniform(0, self.number_line_length - 1),

            wille_position.append(wille)
        return wille_position

    def _get_current_observation(self):
        # Get the current observation as a dictionary
        position = np.array([self.current_position / self.number_line_length], dtype=np.float32)  # Normalize position
        channel_type = np.array([self.previous_channel_type / self.num_channels], dtype=np.float32)  # Normalize channel_type
        energy_level = np.array([self.channel_param[self.current_position, self.previous_channel_type]['energy_level']], dtype=np.float32)
        covert_rate = np.array([self.channel_param[self.current_position, self.previous_channel_type]['covert_rate']], dtype=np.float32)

        # Return the observation as a dictionary
        observation = np.concatenate((position, channel_type, energy_level, covert_rate), axis=None)
        return observation

    def step(self, action):
        # assert not self._terminated, "Episode has terminated"
         # Decode the action into target node and channel
        target_node = action % 3+self.current_position
        channel = action // 3

        # Ensure action is within bounds
        if target_node >= self.number_line_length or channel >= self.num_channels:
            observation = self._get_current_observation()
            reward = -1
            return observation, reward, False, False, {}
            # raise ValueError("Action out of bounds")

        # Update the current position to the target node
        previous_position = self.current_position

        # Calculate reward
        reward = self._calculate_reward(previous_position, target_node, action)

        # Define termination conditions
        terminated = bool(self.current_position == self.number_line_length - 1)  # Ensure terminated is boolean
        truncated = False  # No truncation logic for now

        # Get the new observation
        observation = self._get_current_observation()

        # Update previous channel type
        self.previous_channel_type = channel  # Update based on channel_type (denormalized)
        self.current_position = target_node
        return observation, reward, terminated, truncated, {}

    def _calculate_reward(self, previous_position, target_node, channel_type):
        # Calculate the reward based on the movement and channel parameters
        d_AB = abs(target_node - previous_position) + self.sigma
        observation = self._get_current_observation()
        tr = self.channel_param[previous_position][channel_type][0]  # energy_level
        cr = self.channel_param[previous_position][channel_type][1]  # covert_rate

        reward = self._calculate_transition_rate(d_AB, tr) - self._calculate_covertness(target_node, previous_position, channel_type//3, cr)
        reward -= 3 * (self.previous_channel_type // 3 != channel_type)
        return reward - 1

    # trans rate
    def _calculate_transition_rate(self, distance, energy):
        return energy / distance * np.random.uniform(0.9, 1)

    # covertness
    def _calculate_covertness(self, target_node, previous_position, channel_type, covert_rate):
        wille_location = self.wille_position[channel_type][0]
        d_cW = abs(wille_location - previous_position)
        d_tW = abs(target_node - wille_location)
        return covert_rate / (d_cW + d_tW + self.sigma)

    def reset(self, seed=None):
        self._num_steps = 0
        self._terminated = False
        self.current_position = 0
        self.previous_channel_type = 0
        return self._get_current_observation(), {}

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        """Render the current state of the number line."""
        line_representation = ['-' for _ in range(self.number_line_length)]
        line_representation[self.current_position] = 'A'
        for wille in self.wille_position:
            position = int(round(wille[0]))
            if 0 <= position < self.number_line_length:
                line_representation[position] = 'W'
        print(''.join(line_representation))

    def close(self):
        pass
    
    # 新增函数：渲染数轴和Wille节点的图形表示
    def render_number_line_plot(self):
        plt.figure(figsize=(15, 6))
        plt.title("Three Number Lines Representing Different Channel Types with Connections")
        plt.xlabel("Position on Number Line")
        plt.ylabel("Channel Types")

        # 定义三条数轴，每条数轴代表一个信道类型
        channel_lines = [1, 2, 3]
        colors = ['red', 'blue', 'green']
        channel_labels = ['Channel Type 1', 'Channel Type 2', 'Channel Type 3']

        # 绘制三条数轴
        for i, channel_line in enumerate(channel_lines):
            plt.hlines(channel_line, 0, self.number_line_length - 1, colors='black', linewidth=1)

        # 绘制Wille节点并连线
        for i in range(len(self.wille_position)):
            wille_position_x = self.wille_position[i][0]  # x 坐标
            y_coordinate = channel_lines[i]  # 获取数轴的 y 坐标
            plt.plot(wille_position_x, y_coordinate, 'o', markersize=10)

        # 添加标签和图例
        plt.xticks(range(0, self.number_line_length, 2))
        plt.yticks(channel_lines, channel_labels)
        plt.legend()
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        # 显示图形
        plt.show()

    @staticmethod
    def generate_env(number_line_length, available_channel=None, channel_param=None, wille_position=None):
        return RouteEnv(number_line_length, available_channel, channel_param, wille_position)


def make_route_env(training_num=0, test_num=0):
    """Wrapper function for Route env.
    :return: a tuple of (single env, training envs, test envs).
    """
    line_length = 10
    env = RouteEnv(number_line_length=line_length)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: RouteEnv(number_line_length=line_length) for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: RouteEnv(number_line_length=line_length) for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs


if __name__ == "__main__":
    # Example input data to fully customize the environment
    number_line_length = 20
    num_willes = 5

    # Generate environment using the new generation function with custom values
    env = RouteEnv(number_line_length=number_line_length)
    # check_env(env)
    observation, info = env.reset()
    env.render()
    done = False

    while not done:
        action = env.action_space.sample()  # Randomly select target node and channel
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            done = True

    env.close()
