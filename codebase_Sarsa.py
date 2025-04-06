import math
import numpy as np
from codebase_train import epsilon_greedy
from util import plot_return

from codebase_train import epsilon_greedy


def n_step(env, state, q_table, n, epsilon):
    """
    Perform up to 'n' steps ...
    """
    states = [state]
    actions = []
    rewards = []
    acted_steps = 0

    # Choose first action
    action = epsilon_greedy(state, q_table, epsilon)
    actions.append(action)

    for i in range(n):
        old_state = states[-1]  # The current state before we move
        next_state, reward, done, truncated, info = env.step(action)

        # [ADDED] Print details of *this step*
        print(f"[n_step] i={i}, state={old_state}, action={action}, "
              f"reward={reward}, done={done}, next_state={next_state}")

        rewards.append(reward)
        acted_steps += 1

        if done:
            break

        next_action = epsilon_greedy(next_state, q_table, epsilon)
        states.append(next_state)
        actions.append(next_action)
        action = next_action

    return states, actions, rewards, acted_steps


def sarsa(env, num_episode, gamma, alpha, init_epsilon, num_steps,
          init_q_value):
    ############################
    # Your Code #
    import numpy as np
    from codebase_train import epsilon_greedy

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    tabular_q = np.full((num_states, num_actions), init_q_value, dtype=float)

    all_returns = []

    for episode in range(num_episode):
        epsilon = init_epsilon

        state, _ = env.reset()
        done = False
        episode_return = 0

        max_episode_steps = 10000
        steps_this_episode = 0

        while not done and steps_this_episode < max_episode_steps:
            states, actions, rewards, acted_steps = n_step(
                env, state, tabular_q, num_steps, epsilon)
            steps_this_episode += acted_steps
            discount = 1.0
            G = 0.0
            for r in rewards:
                G += discount * r
                discount *= gamma

            s0 = states[0]
            a0 = actions[0]
            old_q = tabular_q[s0, a0]

            if acted_steps == num_steps:
                s_last = states[-1]
                a_last = actions[-1]
                G += discount * tabular_q[s_last, a_last]

            tabular_q[s0, a0] = old_q + alpha * (G - old_q)

            episode_return += sum(rewards)

            if len(states) > 0:
                state = states[-1]

            if acted_steps < num_steps:
                done = True

        all_returns.append(episode_return)

        # [ADDED for logging] Print a message each episode
        print(f"Episode {episode} finished with return = {episode_return}")

        if steps_this_episode >= max_episode_steps:
            print(f"Episode {episode} terminated after hitting step limit.")

    plot_return(all_returns, f'sarsa_alpha_{alpha}', all_returns[-1])
    return tabular_q
    ############################
