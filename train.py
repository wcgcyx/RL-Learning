import sys
import numpy as np
from sac import Agent as SAC
from sac_trps_per import Agent as PER
from sac_trps import Agent as TRPS
from sac_ce50 import Agent as CE50
from utils import get_normalized_env

def train_agent(
        agent,
        env,
        max_episode,
        is_render,
        output_file,
        task):

    episode = 0
    frame = 0

    while episode <= max_episode:

        state = env.reset()
        if task == 'jitterbug':
            state = np.concatenate([state['position'], state['velocity'],
                                    state['motor_position'], state['motor_velocity']])

        total_reward = 0
        step_taken = 0

        for step in range(500):

            print("{}\r".format(total_reward), end='')

            step_taken += 1
            action = agent.choose_action(state)
            # print("State: {}, Action: {}".format(state, action))
            next_state, reward, end, _ = env.step(action.numpy())
            if task == 'jitterbug':
                next_state = np.concatenate([next_state['position'], next_state['velocity'],
                                        next_state['motor_position'], next_state['motor_velocity']])

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
    # Args: train.py agent_name task_name episode file_id [N Ne t size]
    if len(sys.argv) != 5 and len(sys.argv) != 9:
        print("Please provide agent name and file id")
        exit(1)
    agent_name = sys.argv[1]
    task = sys.argv[2]
    episode = int(sys.argv[3])
    file_id = sys.argv[4]
    N = None
    Ne = None
    t = None
    size = None
    if len(sys.argv) == 8:
        N = int(sys.argv[5])
        Ne = int(sys.argv[6])
        t = int(sys.argv[7])
        size = float(sys.argv[8])
    print("Agent: {} Task: {} Episode: {} File: {}".
          format(agent_name, task, episode, file_id))
    file_name = agent_name + '_' + str(N) + '_' + str(Ne) + '_' + str(t) + '_' + str(size) + '_' + task + '_' + str(file_id) + '.csv'
    environment = get_normalized_env(task)
    if task == 'jitterbug':
        state_dim = 15
        action_dim = 1
    else:
        state_dim = environment.observation_space.shape[0]
        action_dim = environment.action_space.shape[0]
    if agent_name == 'sac':
        agent_instance = SAC(state_dim, action_dim)
    elif agent_name == 'per':
        agent_instance = PER(state_dim, action_dim)
    elif agent_name == 'trps':
        agent_instance = TRPS(state_dim, action_dim, N=N, Ne=Ne, t=t, size=size)
    elif agent_name == 'ce50':
        agent_instance = CE50(state_dim, action_dim, N=N, Ne=Ne, t=t, size=size)
    else:
        agent_instance = None
    if agent_instance is None:
        print("Unsupported agent name")
        exit(2)

    with open(file_name, "a") as file:
        file.write("{},Round {}\n".format("Episode", str(file_id)))

    train_agent(agent_instance, environment, episode, False, file_name, task)
    exit(0)
