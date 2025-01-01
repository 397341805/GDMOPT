import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Discrete
from tianshou.env import DummyVectorEnv
import matplotlib.pyplot as plt
from torch.nn.functional import channel_shuffle
import math
from .utility import CovertUtility, CalculateReward


class RouteEnv(gym.Env):
    """Environment with a number line containing multiple Wille nodes and complex channel selection."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, number_node=20, fixed_seed=False):
        super().__init__()
        self.fixed_seed = fixed_seed
        self.number_node = number_node
        self.num_willes = 3
        self.num_channels = 9
        self.sigma = 1
        self._flag = 0
        self._num_steps = 0
        self._terminated = False
        self._laststate = None
        self._steps_per_episode = 1
        self.current_position = 0
        self.previous_channel_type = 0
        # Available channel structure: N * N * 9 (three-dimensional array)
        self.available_channel = [1, 2, 3]
        self._laststate = np.zeros(self.number_node + 1)
        # Channel parameters structure
        # self.channel_param = self._initialize_channels()
        # self.wille_position, self.wille_penalty = self._initialize_wille_wille_param()
        self.channel_param = np.array([
            [0.74908024, 0.77735458, 1.61488031, 3.22370579, 3.72620685, 3.92489459, 4.12203823, 4.03142919,
             4.36778313],
            [1.90142861, 0.54269806, 1.7921826, 2.27898772, 3.24659625, 2.50356459, 4.49517691, 4.63641041, 4.63230583],
            [1.46398788, 1.65747502, 0.63600695, 2.5842893, 2.66179605, 2.99449701, 4.03438852, 4.31435598, 4.63352971],
            [1.19731697, 0.71350665, 0.22010385, 2.73272369, 2.1271167, 2.60175662, 4.9093204, 4.50857069, 4.53577468],
            [0.31203728, 0.56186902, 0.45587033, 2.91213997, 2.62196464, 2.56968099, 4.25877998, 4.90756647,
             4.09028977],
            [0.31198904, 1.08539217, 0.85421558, 3.57035192, 2.65036664, 2.07377389, 4.66252228, 4.24929223, 4.8353025],
            [0.11616722, 0.28184845, 1.63602953, 2.39934756, 3.45921236, 3.21912867, 4.31171108, 4.41038292,
             4.32078006],
            [1.73235229, 1.60439396, 1.72146117, 3.02846888, 3.27511494, 3.00535805, 4.52006802, 4.75555114,
             4.18651851],
            [1.20223002, 0.14910129, 0.01390426, 3.18482914, 3.77442549, 2.1029575, 4.54671028, 4.22879817, 4.04077514],
            [1.41614516, 1.97377387, 1.02149461, 2.09290083, 2.94442985, 2.55729293, 4.18485446, 4.07697991,
             4.59089294],
            [0.04116899, 1.54448954, 0.83482201, 3.2150897, 2.23918849, 3.81653177, 4.96958463, 4.28975145, 4.67756436],
            [1.9398197, 0.39743136, 0.44421562, 2.34104825, 3.42648957, 2.47912378, 4.77513282, 4.16122129, 4.01658783],
            [1.66488528, 0.01104423, 0.23973073, 2.13010319, 3.5215701, 2.28978974, 4.93949894, 4.92969765, 4.51209306],
            [0.42467822, 1.63092286, 0.67523034, 3.89777107, 3.1225544, 2.97890552, 4.89482735, 4.80812038, 4.22649578],
            [0.36364993, 1.41371469, 1.88581941, 3.93126407, 3.54193436, 3.97130091, 4.59789998, 4.63340376,
             4.64517279],
            [0.36680902, 1.45801434, 0.64640586, 3.6167947, 2.98759119, 2.48411054, 4.92187424, 4.87146059, 4.17436643],
            [0.60848449, 1.54254069, 1.03758124, 2.60922754, 3.04546566, 3.34427109, 4.0884925, 4.80367208, 4.69093774],
            [1.04951286, 0.1480893, 1.40603792, 2.19534423, 2.85508204, 3.52323923, 4.19598286, 4.18657006, 4.38673535],
            [0.86389004, 0.71693146, 0.7272592, 3.36846605, 2.05083825, 2.47527509, 4.04522729, 4.892559, 4.93672999],
            [0.58245828, 0.23173812, 1.94356417, 2.88030499, 2.21578285, 3.4564327, 4.32533033, 4.53934224, 4.13752094]
        ])
        self.wille_position = np.array([2.632018060142943, 3.773824883145467, 7.979952138102537])
        self.wille_penalty = np.array([6.48026067, 2.1559969, 17.56917875])
        # Simplified action and observation spaces
        self._action_space = spaces.Discrete(self.number_node)
        # self._action_space = spaces.Box(shape=self.state.shape, low=-1, high=1)
        self._num_steps = 0
        self._observation_space = Box(shape=self.state.shape, low=0, high=1000)
        self._terminated = False

        self.last_expert_action = None
        # Initialize the number line with Wille nodes






    def _initialize_channels(self):
        # Initialize channel parameters with random values
        if self.fixed_seed:
            np.random.seed(42)  # Fixed seed for reproducibility
        channel_param= np.zeros((self.number_node, self.num_channels))
        for i in range(self.num_channels // 3):
            # 前1/3在一个区间随机赋值，中间1/3在一个区间随机赋值，后1/3在一个区间随机赋值
            channel_param[:, i] = np.random.uniform(0, 2, size=self.number_node)
            channel_param[:, i + 3] = np.random.uniform(2, 4, size=self.number_node)
            channel_param[:, i + 6] = np.random.uniform(4, 5, size=self.number_node)
        return channel_param

    def _initialize_wille_wille_param(self):
        # Initialize the number line with Wille nodes at random positions
        if self.fixed_seed:
            np.random.seed(42)  # Fixed seed for reproducibility
        wille_position = np.random.uniform(0, self.number_node - 1,self.num_willes)
        wille_penalty = []
        for i in range(self.num_willes):
            wille_penalty.append(np.random.uniform(0,1))
            # wille_penalty.append(np.random.uniform(0, 10))
        return wille_position, wille_penalty

    @property
    def observation_space(self):
        # Return the observation space
        return self._observation_space

    @property
    def action_space(self):
        # Return the action space
        return self._action_space

    @property
    def state(self):
        # Provide the current state to the agent
        # rng = np.random.default_rng(seed=0)
        # states1 = rng.uniform(1, 2, 5)
        # states2 = rng.uniform(0, 1, 5)

        # self.wille_param = self._initialize_wille_wille_param()

        # reward_in = []
        # states = np.zeros(self.number_node)
        # # action = np.random.uniform(0, self.num_channels, size=self.number_node)
        # # reward, _, _, _ = CovertUtility(
        # #     states,
        # #     self.channel_param,
        # #     self.wille_position,
        # #     self.wille_penalty,
        # #     action,
        # #     self.number_node,
        # #     self.num_willes,
        # #     self.num_channels
        # # )
        # # self.channel_gains = action
        # reward_in.append(0)
        # # self.last
        # states = np.concatenate([states , reward_in])
        # self._laststate = states
        return self._laststate
    def step(self, action):
        # assert not self._terminated, "Episode has terminated"
        # assert not self._terminated, "One episodic has terminated"
         # Decode the action into target node and channel
        states = self._laststate[0:-1]
        action = np.round(action)
        reward, expert_action, sub_expert_action, real_action = CovertUtility(
            states,
            self.channel_param,
            self.wille_position,
            self.wille_penalty,
            action,
            self.number_node,
            self.num_willes,
            self.num_channels
        )
        reward = self.covert_reward(states, action)
        self._laststate[-1] = reward
        self._laststate[0:-1] = np.clip((states + action),0,self.num_channels-1)
        # self._laststate[0:-1] = self.channel_gains * real_action
        self._num_steps += 1
        # Check if episode should end based on number of steps taken
        if self._num_steps >= self._steps_per_episode:
            self._terminated = True

        info = {'num_steps': self._num_steps, 'expert_action': expert_action, 'sub_expert_action': sub_expert_action}
        return self._laststate, reward, self._terminated, info

    # def covert_reward(self, state, action):
    #     actmap= np.floor( action * 2)
    #     state = np.floor(state)
    #     actions = actmap + state
    #     rewards = []
    #     actions = np.clip(actions, 0, self.num_channels - 1)
    #     state = np.clip(state, 0, self.num_channels - 1)
    #     for i in range(self.number_node):
    #         s = int(state[i])
    #         a = int(actions[i])
    #         # reward1 = self.channel_param[i][a] - self.wille_penalty[a // self.num_willes]
    #         # reward2 = self.channel_param[i][s] - self.wille_penalty[s // self.num_willes]
    #         reward1 = self.channel_param[i][s] - (abs(self.wille_position[s // self.num_willes] - i) ** -2) * self.wille_penalty[
    #             s // self.num_willes]
    #         reward2 = self.channel_param[i][a] - (abs(self.wille_position[a // self.num_willes] - i) ** -2) * self.wille_penalty[
    #             a // self.num_willes]
    #         rewards.append(reward2 - reward1)
    #     return np.sum(rewards)

    def covert_reward(self, state, action):
        actmap= np.round(((action + 0) / 2) * 4)
        states = np.round(state)
        actions = actmap + states
        rewards = []
        actions = np.clip(actions, 0, self.num_channels - 1)
        state = np.clip(state, 0, self.num_channels - 1)
        for i in range(self.number_node):
            # s = int(states[i])
            a = int(actions[i])
            # reward1 = self.channel_param[i][a] - self.wille_penalty[a // self.num_willes]
            rew = CalculateReward( self.channel_param[i][a],abs(self.wille_position[a // self.num_willes] - i)  , self.wille_penalty[
                a // self.num_willes])
            # reward1 = self.channel_param[i][s] - np.exp(abs(self.wille_position[s // self.num_willes] - i)  * self.wille_penalty[
            #     s // self.num_willes])
            # reward2 = self.channel_param[i][a] - np.exp(abs(self.wille_position[a // self.num_willes] - i) * self.wille_penalty[
            #     a // self.num_willes])
            rewards.append(rew)
        return np.sum(rewards)


    def reset(self, seed=None):
        self._num_steps = 0
        self._terminated = False
        state = self.state
        return state
        # return state, {'num_steps': self._num_steps}

    def seed(self, seed=None):
        np.random.seed(seed)

    def render(self):
        """Render the current state of the number line."""
        rewards,noise = [],[]
        for i in range(self.number_node):
            # s = int(states[i])
            a = int(self._laststate[i])
            # reward1 = self.channel_param[i][a] - self.wille_penalty[a // self.num_willes]
            rew,penalty = self.channel_param[i][a] * np.random.uniform(0.99, 1.01) , np.exp(-abs(self.wille_position[a // self.num_willes] - i)/
                                                                                            self.wille_penalty[a // self.num_willes])
            rewards.append(round(rew,2))
            noise.append(round(penalty,2))
        # print(rewards,noise)
        return rewards, noise

    def close(self):
        pass

    @staticmethod
    def generate_env(number_node, available_channel=None, channel_param=None, wille_position=None):
        return RouteEnv(number_node, available_channel, channel_param, wille_position)


def make_route_env(training_num=0, test_num=0):
    """Wrapper function for Route env.
    :return: a tuple of (single env, training envs, test envs).
    """
    line_length = 20
    env = RouteEnv(number_node=line_length)
    env.seed(0)

    train_envs, test_envs = None, None
    if training_num:
        train_envs = DummyVectorEnv(
            [lambda: RouteEnv(number_node=line_length) for _ in range(training_num)])
        train_envs.seed(0)

    if test_num:
        test_envs = DummyVectorEnv(
            [lambda: RouteEnv(number_node=line_length) for _ in range(test_num)])
        test_envs.seed(0)
    return env, train_envs, test_envs


if __name__ == "__main__":
    # Example input data to fully customize the environment
    number_node = 20
    num_willes = 3

    # Generate environment using the new generation function with custom values
    env = RouteEnv(number_node=number_node)
    print(env.channel_param)
    print(env.wille_position)
    print(env.wille_position)

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
