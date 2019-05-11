from dm_control import suite
from double_dqn_pri_replay import Agent
from display import render
import matplotlib.pyplot as plt


def train_agent(
        domain,
        task,
        max_episode,
        is_render=False):

    # Used to plot reward function
    episodes_plot = []
    total_rewards_plot = []

    env = suite.load(domain, task, visualize_reward=True)

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    agent = Agent(action_spec, observation_spec)

    episode = 0

    while episode <= max_episode:

        time_step = env.reset()
        if is_render:
            render(env)
        total_reward = 0

        while not time_step.last():

            state = time_step.observation

            action = agent.choose_action(state)

            time_step = env.step(action)
            if is_render:
                render(env)
            state_ = time_step.observation
            reward = time_step.reward
            end = 1 if time_step.last() else 0

            agent.store_transition(state_1=state, action=action, reward=reward, end=end, state_2=state_)

            if episode > 0:
                agent.learn()

            total_reward += reward

        if episode % 50 == 0:
            agent.save_weights("{}_{}_{}_weights_{}.h5".format(agent, domain, task, episode))
            print("Weights saved.")

        print(episode, total_reward)

        episodes_plot.append(episode)
        total_rewards_plot.append(total_reward)

        episode += 1

    env.close()

    fig = plt.gcf()
    plt.plot(episodes_plot, total_rewards_plot, 'ro')
    plt.draw()
    fig.savefig("{}_{}_{}_reward_plot.png".format(agent, domain, task), dpi=100)
    print("Plot saved.")


if __name__ == "__main__":
    train_agent("pendulum", "swingup", 250, is_render=False)