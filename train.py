import sys
from sac import Agent as SAC
from sac_dpn import Agent as DPN
from sac_per import Agent as PER
from sac_trls import Agent as TRLS
from sac_trr import Agent as TRR
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
                file.write("{},{}\n".format(episode, total_reward))

        episode += 1

    env.close()

    print("Training finished.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide agent name")
        exit(1)
    agent_name = sys.argv[1]
    for i in range(5):
        environment = get_normalized_env(task="Pendulum-v0")
        state_dim = environment.observation_space.shape[0]
        action_dim = environment.action_space.shape[0]
        if agent_name == 'sac':
            agent_instance = SAC(state_dim, action_dim)
        elif agent_name == 'dpn':
            agent_instance = DPN(state_dim, action_dim)
        elif agent_name == 'per':
            agent_instance = PER(state_dim, action_dim)
        elif agent_name == 'trls':
            agent_instance = TRLS(state_dim, action_dim)
        elif agent_name == 'trps':
            agent_instance = TRLS(state_dim, action_dim)
        elif agent_name == 'trr':
            agent_instance = TRR(state_dim, action_dim)
        else:
            agent_instance = None
        if agent_instance is None:
            print("Unsupported agent name")
            exit(2)
        file_name = agent_name + '_' + str(i) + '.csv'
        train_agent(agent_instance, environment, 250, False, file_name)
    exit(0)
