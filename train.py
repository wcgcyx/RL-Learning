import sys
from sac import Agent as SAC
from sac_per import Agent as PER
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
            # print("State: {}, Action: {}".format(state, action))
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

        print(episode, frame, total_reward)

        if output_file is not None:
            with open(output_file, "a") as file:
                file.write("{},{}\n".format(episode, total_reward))

        episode += 1

    env.close()

    print("Training finished.")


if __name__ == "__main__":
    # Args: train.py agent_name task_name file_id [N Ne t size]
    if len(sys.argv) != 4 and len(sys.argv) != 8:
        print("Please provide agent name and file id")
        exit(1)
    agent_name = sys.argv[1]
    task = sys.argv[2]
    file_id = sys.argv[3]
    N = None
    Ne = None
    t = None
    size = None
    if len(sys.argv) == 8:
        N = int(sys.argv[4])
        Ne = int(sys.argv[5])
        t = int(sys.argv[6])
        size = float(sys.argv[7])
    print("Agent: {} Task: {} File: {}".
          format(agent_name, task, file_id))
    file_name = agent_name + '_' + str(N) + '_' + str(Ne) + '_' + str(t) + '_' + str(size) + '_' + task + '_' + str(file_id) + '.csv'
    environment = get_normalized_env(task)
    state_dim = environment.observation_space.shape[0]
    action_dim = environment.action_space.shape[0]
    if agent_name == 'sac':
        agent_instance = SAC(state_dim, action_dim)
    elif agent_name == 'per':
        agent_instance = PER(state_dim, action_dim)
    elif agent_name == 'trps':
        agent_instance = TRPS(state_dim, action_dim, N=N, Ne=Ne, t=t, size=size)
    else:
        agent_instance = None
    if agent_instance is None:
        print("Unsupported agent name")
        exit(2)

    with open(file_name, "a") as file:
        file.write("{},Round {}\n".format("Episode", str(file_id)))

    train_agent(agent_instance, environment, 1000, False, file_name)
    exit(0)
