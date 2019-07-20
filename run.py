from dm_control import suite
from sac import Agent
from utils import render


def train_agent(
        domain,
        task,
        max_episode,
        is_render=False):

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

        print(episode, total_reward)

        if episode != 0 and episode % 50 == 0:
            agent.save_weights(domain, task)
            print("Weights saved.")

        episode += 1

    env.close()

    print("Training finished.")


if __name__ == "__main__":
    train_agent("cartpole", "balance", 500, is_render=False)
