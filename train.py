import sys
from sac import Agent as SAC
from sac_trps import Agent as TRPS
from utils import get_normalized_env

def train_agent(
        agent,
        env,
        max_episode,
        is_render,
        output_file):

    episode = 0
    frame = 0

    while episode <= max_episode:

        state = env.reset()
        total_reward = 0
        step_taken = 0

        for step in range(500):
            step_taken += 1
            action = agent.choose_action(state)
            next_state, reward, end, _ = env.step(action.numpy())

            if step >= 499:
                end = True

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

        print(episode, frame, total_reward / step_taken, total_reward)

        if output_file is not None:
            with open(output_file, "a") as file:
                file.write("{},{}\n".format(episode, total_reward))

        episode += 1

    env.close()

    print("Training finished.")


if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Please provide agent name and file id")
        exit(1)
    agent_name = sys.argv[1]
    task = "Ant-v2"
    file_id = sys.argv[2]
    debug_file = None
    if len(sys.argv) == 4:
        debug_file = agent_name + '_' + task + '_' + str(file_id) + '_debug' + '.csv'
    print(agent_name, file_id)
    file_name = agent_name + '_' + task + '_' + str(file_id) + '.csv'
    environment = get_normalized_env(task)
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    if agent_name == 'sac':
        agent_instance = SAC(state_dim, action_dim, debug_file=debug_file)
    elif agent_name == 'trps':
        agent_instance = TRPS(state_dim, action_dim, debug_file=debug_file)
    else:
        agent_instance = None
    if agent_instance is None:
        print("Unsupported agent name")
        exit(2)

    with open(file_name, "a") as file:
        file.write("{},Round {}\n".format("Episode", str(file_id)))

    train_agent(agent_instance, environment, 250, False, file_name)
    exit(0)
