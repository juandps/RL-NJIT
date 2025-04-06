import time
import numpy as np
from util import actions_of_tabular_q


import numpy as np

def epsilon_greedy(state, q_table, epsilon):
    """
    Choose an action for 'state' using an epsilon-greedy strategy from 'q_table'.
    
    Parameters:
      state: int, current state
      q_table: 2D array or dict, q_table[state][action] -> Q-value
      epsilon: float in [0,1], exploration rate
      
    Returns:
      action: int, chosen action index
    """
    if np.random.rand() < epsilon:
        # Exploration: choose a random action
        num_actions = q_table.shape[1]
        action = np.random.randint(num_actions)
    else:
        # Exploitation: choose the action(s) with highest Q-value, break ties randomly
        max_q = np.max(q_table[state])
        best_actions = np.where(q_table[state] == max_q)[0]
        action = np.random.choice(best_actions)
    return action


def greedy_policy(tabular_q, state):
    """
    Greedy policy to select action. It is used for rendering.
    :param tabular_q: q table
    :param state: current state to select action
    :return: action: the action to take according to the greedy policy
    """
    action = int(np.argmax(tabular_q[state]))
    return action


def render_single(env, tabular_q, max_steps=100):
    '''
    This function does not need to be modified.
    Renders policy once on environment. Watch your agent play!
    :param env: Environment to play on. Must have nS, nA, and P as attributes.
    :param tabular_q: q table
    :param max_steps: the maximum number of iterations
    :return: episode_reward: total reward for the episode
    '''
    episode_reward = 0
    actions_of_tabular_q(tabular_q)
    state, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.15)
        action = greedy_policy(tabular_q, state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)
    return episode_reward


def evaluate_policy(env, tabular_q, max_steps=100):
    """
    Print action for each state then evaluate the policy for one episode, no rendering
    :param env: environment
    :param tabular_q: q table
    :param max_steps: stop if the episode reaches max_steps
    :return: episode_reward: total reward for the episode.
    """
    episode_reward = 0
    actions_of_tabular_q(tabular_q)
    state, _ = env.reset()
    for t in range(max_steps):
        action = greedy_policy(tabular_q, state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)
    return episode_reward