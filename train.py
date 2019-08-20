from sac import Agent
from utils import get_normalized_env


def train_agent(
        task,
        max_episode,
        is_render=False,
        identifier=-1,
        output_file=None):

    env = get_normalized_env(task)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = Agent(state_dim, action_dim)

    episode = 0
    frame = 0

    if identifier > 0:
        agent.load_weights(task, str(identifier))
        episode = identifier + 1

    while episode <= max_episode + identifier + 1:

        state = env.reset()
        total_reward = 0

        for step in range(500):

            action = agent.choose_action(state)
            next_state, reward, end, _ = env.step(action.numpy())

            if is_render:
                env.render()

            agent.store_transition(state, action, reward, next_state, end)

            state = next_state
            total_reward += reward
            frame += 1

            if frame > 2 * agent.batch_size:
                agent.learn()

            if end:
                break

        print(episode, frame, total_reward)

        if output_file is not None:
            with open(output_file, "a") as file:
                file.write("{},{}".format(episode, total_reward))

        if episode != 0 and episode % 50 == 0:
            agent.save_weighs(task, str(episode))
            print("Weights saved.")

        episode += 1

    env.close()

    print("Training finished.")


if __name__ == "__main__":
    train_agent(task="Ant-v2", max_episode=500, is_render=False, identifier=400)
