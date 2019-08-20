from sac import Agent
from utils import get_normalized_env


def demo_agent(
        task,
        max_episode,
        identifier):

    env = get_normalized_env(task)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(state_dim, action_dim)

    agent.load_weights(task, str(identifier))
    episode = identifier + 1

    while episode <= max_episode + identifier + 1:

        state = env.reset()
        total_reward = 0

        for step in range(500):

            action = agent.choose_action(state)
            next_state, reward, end, _ = env.step(action.numpy())
            env.render()
            state = next_state
            total_reward += reward

            if end:
                break

        print(episode, total_reward)

        episode += 1

    env.close()

    print("Training finished.")


if __name__ == "__main__":
    demo_agent(task="Humanoid-v2", max_episode=500, identifier=50)
