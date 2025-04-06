import math
import numpy as np
from codebase_train import epsilon_greedy
from util import plot_return


from codebase_train import epsilon_greedy

def n_step(env, state, q_table, n, epsilon):
    """
    Perform up to 'n' steps in the environment from 'state' using epsilon-greedy for actions.

    Parameters:
      env: the CliffWalking environment
      state: int, initial state
      q_table: Q-value table
      n: number of steps to sample
      epsilon: exploration rate

    Returns:
      states: list of state indices visited
      actions: list of actions taken
      rewards: list of rewards received
      acted_steps: int, how many steps actually taken (<= n)
    """
    states = [state]
    actions = []
    rewards = []
    acted_steps = 0

    # Choose first action
    action = epsilon_greedy(state, q_table, epsilon)
    actions.append(action)

    for i in range(n):
        next_state, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        acted_steps += 1

        if done:
            # Episode terminated early
            break
        
        # If continuing, pick next action via epsilon greedy
        next_action = epsilon_greedy(next_state, q_table, epsilon)
        
        # Store the new state & action
        states.append(next_state)
        actions.append(next_action)
        
        # Move forward
        action = next_action

    return states, actions, rewards, acted_steps

def sarsa(env, num_episode, gamma, alpha, init_epsilon, num_steps, init_q_value):
    ############################
    # Your Code #
    import numpy as np
    from codebase_train import epsilon_greedy
    # n_step is in the same file, so you can do: from codebase_Sarsa import n_step
    # or just call n_step(...) directly.

    # 1) Initialize Q-table
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    tabular_q = np.full((num_states, num_actions), init_q_value, dtype=float)

    all_returns = []

    for episode in range(num_episode):
        # Decide whether epsilon decays each episode or stays constant:
        # For constant, just do:
        epsilon = init_epsilon
        
        state, _ = env.reset()
        done = False
        episode_return = 0

        # 2) Keep sampling n-step segments until episode ends
        while not done:
            states, actions, rewards, acted_steps = n_step(env, state, tabular_q, num_steps, epsilon)

            # 3) Compute the return G for these steps
            discount = 1.0
            G = 0.0
            for r in rewards:
                G += discount * r
                discount *= gamma

            # 4) Update Q for the FIRST state-action in this segment
            s0 = states[0]
            a0 = actions[0]
            old_q = tabular_q[s0, a0]

            # If we didn’t terminate early, we can bootstrap from the Q of the last (state, action)
            if acted_steps == num_steps:
                # states[-1] is the last new state appended
                s_last = states[-1]
                a_last = actions[-1]
                G += discount * tabular_q[s_last, a_last]

            # SARSA update
            tabular_q[s0, a0] = old_q + alpha * (G - old_q)

            # 5) Accumulate rewards for the entire episode
            episode_return += sum(rewards)

            # 6) Move our 'current state' forward
            if len(states) > 0:
                state = states[-1]

            # If the environment ended before reaching n steps
            if acted_steps < num_steps:
                done = True
        
        # Store this episode’s total return
        all_returns.append(episode_return)

    plot_return(all_returns, f'sarsa_alpha_{alpha}', all_returns[-1])
    return tabular_q
    ############################
