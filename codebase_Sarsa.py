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
        print(f"Step {i+1}: state={state}, action={action}, reward={reward}, next_state={next_state}") #added logging for each step

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
    MAX_EPISODE_STEPS = 10000  # Add maximum steps limit

    for episode in range(num_episode):
        # Choose epsilon strategy (constant or decaying)
        # Uncomment one of these lines:
        epsilon = init_epsilon  # Constant epsilon
        # epsilon = init_epsilon / (episode + 1)  # Decaying epsilon

        state, _ = env.reset()
        done = False
        episode_return = 0
        episode_steps = 0

        print(f"\nStarting Episode {episode+1} with epsilon {epsilon:.4f}")

        # 2) Keep sampling n-step segments until episode ends
        while not done and episode_steps < MAX_EPISODE_STEPS:
            states, actions, rewards, acted_steps = n_step(env, state, tabular_q, num_steps, epsilon)
            episode_steps += acted_steps

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

            # If we didn't terminate early, we can bootstrap from the Q of the last (state, action)
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

        # Print episode summary
        status = "completed" if done else "terminated due to step limit"
        print(f"Episode {episode+1} {status} after {episode_steps} steps with return {episode_return}")

        # Store this episode's total return
        all_returns.append(episode_return)

    # Save plot with unique name based on parameters
    plot_name = f'sarsa_alpha_{alpha}_eps_{init_epsilon}_nstep_{num_steps}'
    plot_return(all_returns, plot_name, all_returns[-1])
    return tabular_q
    ############################