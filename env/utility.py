import numpy as np
import torch
from scipy.stats import nakagami
from scipy.special import gammainc
import math
from scipy.io import savemat
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def rayleigh_channel_gain(ex, sta):
    num_samples = 1
    gain = np.random.normal(ex, sta, num_samples)
    # Square the absolute value to get Rayleigh-distributed gains
    gain = np.abs(gain) ** 2
    return gain

# Function to implement water filling algorithm for power allocation
def water(s, total_power):
    a = total_power
    # Define the channel gain and noise level
    g_n = s
    N_0 = 1  # Assuming a fixed noise-level of 1 for all transmissions, this can be changed based on your requirement

    # Initialize the upper and lower bounds for the bisection search
    L = 0
    U = a + N_0 * np.sum(1 / (g_n + 1e-6))  # Initial guess for upper bound

    # Define the precision for the bisection search
    precision = 1e-6

    # Perform the bisection search for the power level
    while U - L > precision:
        alpha_bar = (L + U) / 2  # Set the current level to be in the middle of bounds
        p_n = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)  # Calculate the power allocation
        P = np.sum(p_n)  # Calculate the total power

        # Check whether the power budget is under or over-utilized
        if P > a:  # If the power budget is over-utilized
            U = alpha_bar  # Move the upper bound to the current power level
        else:  # If the power level is below the power budget
            L = alpha_bar  # Move the lower bound up

    # Calculate the final power allocation
    p_n_final = np.maximum(alpha_bar - N_0 / (g_n + 1e-6), 0)

    # Calculate the data rate for each channel
    SNR = g_n * p_n_final / N_0  # Calculate the SNR
    data_rate = np.log2(1 + SNR)  # Calculate the data rate
    sumdata_rate = np.sum(data_rate)
    # print('p_n_final', p_n_final)
    # print('data_rate', sumdata_rate)
    expert = p_n_final / total_power
    subexpert = p_n_final / total_power + np.random.normal(0, 0.1, len(p_n_final))
    return expert, sumdata_rate, subexpert

# Function to compute utility (reward) for the given state and action
def CompUtility(State, Aution):
    actions = torch.from_numpy(np.array(Aution)).float()
    actions = torch.abs(actions)
    # actions = torch.sigmoid(actions)
    Aution = actions.numpy()
    total_power = 3
    normalized_weights = Aution / np.sum(Aution)
    a = normalized_weights * total_power

    g_n = State
    SNR = g_n * a

    data_rate = np.log2(1 + SNR)

    expert_action, sumdata_rate, subopt_expert_action = water(g_n, total_power)

    reward = np.sum(data_rate) - sumdata_rate
    # reward = np.sum(data_rate) - sumdata_rate

    return reward, expert_action, subopt_expert_action, Aution
# Function to compute utility (reward) for the given state and action
def CovertUtility(state,
                  channel_param,
                  wille_position,
                  covert_penalty,
                  actions,
                  num_node,
                  num_wille=3,
                  num_channels=9):
    real_actions = actions
    actions = actions+state
    # actions = (actions * 10)//10
    actions = np.clip(actions, 0,  num_channels-1)
    # channel_param = state[:num_node].reshape(num_node * num_channels)
    # wille_position = state[num_node:num_wille + num_node]
    # covert_penalty = state[num_wille + num_node:num_wille*2 + num_node]
    trans_rate_true_set = []
    for i in range(len(actions)):
        action = int(actions[i])
        channel_type = action//num_wille
        # print('channel_type', channel_type)
        # print('action', action)
        # print('channel_param', channel_param[i][action])
        # print('wille_position', wille_position[channel_type])
        # print('covert_penalty', covert_penalty[channel_type])
        tr = CalculateReward(channel_param[i][action] ,abs(wille_position[channel_type] - i) , covert_penalty[channel_type])
        trans_rate_true_set.append(tr)

    trans_rate_sum_best, expert_action, subopt_expert_action = CovertBestAction(channel_param,wille_position,covert_penalty, num_node, num_wille=3, num_channels=num_channels)

    reward = np.sum( trans_rate_true_set) - trans_rate_sum_best
    # reward = np.sum(data_rate) - sumdata_rate
    for i in range(len(expert_action)):
        if math.floor(state[i]) == math.floor(expert_action[i]):
            real_actions[i] = 0
    expert_action = np.clip((expert_action-state),0,num_channels-1)
    subopt_expert_action = np.clip((subopt_expert_action-state),0,num_channels-1)
    return reward, expert_action, subopt_expert_action, real_actions
# Calculate best action

def CovertBestAction(channel_param,wille_position, covert_penalty, num_node, num_wille=3, num_channels=9):
    # channel_param = state[:num_node].reshape(num_node*num_channels)
    # wille_position = state[num_node:num_wille+num_node]
    reward_set =[]
    expert_action = []
    for i in range(num_node):
        reward = []
        for j in range(num_channels):
            reward.append(CalculateReward(channel_param[i][j] ,abs(wille_position[j//num_wille] - i) ,covert_penalty[j//num_wille]))
        reward_set.append(np.max(reward))
        expert_action.append(np.argmax(reward))
    subopt_expert_action = expert_action + np.random.normal(0, 0.1, len(expert_action))
    trans_rate_sum = np.sum(reward_set)
    # best_reward = reward_se
    return trans_rate_sum, expert_action, subopt_expert_action
def CalculateReward(channel_param,abs_position, covert_penalty):
    rew = channel_param * np.random.uniform(0.99, 1.01) * np.exp(-abs(abs_position)/ covert_penalty)
    return rew