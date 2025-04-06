import gymnasium as gym


def test_each_action(env):
    """
    reset the env and test each action
    :param env: cliff walking environment
    :return:
    """
    state, _ = env.reset()
    print(state)
    for action in range(env.action_space.n):
        state, _ = env.reset()
        next_state, reward, done, _, _ = env.step(action)
        print(f'state:{state}, action:{action}, reward:{reward}, done:{done}, next_state:{next_state}')



def test_moves(env, actions):
    """
    reset the env and test the policy
    :param env: cliff walking environment
    :param actions: a list of actions to act on the environment
    :return:
    """
    total_reward = 0
    state, _ = env.reset()

    for action in actions:
        env.render()  # <-- Add this line to actually update the GUI every step
        next_state, reward, done, _, _ = env.step(action)
        print(f'state:{state}, action:{action}, reward:{reward}, done:{done}, next_state:{next_state}')
        total_reward += reward
        state = next_state

        if done:
            break

    env.render()  # Show the final state
    print(f'total reward:{total_reward}')
