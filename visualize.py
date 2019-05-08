from dm_control import suite
from double_dqn import Agent
from display import render


def test_agent(
        domain,
        task,
        weights):

    env = suite.load(domain, task, visualize_reward=True)

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    for weight in weights:

        agent = Agent(action_spec, observation_spec)
        agent.e_greedy = 0

        filename = "{}_{}_{}_weights_{}.h5".format(agent, domain, task, weight)
        print("Load " + filename)
        agent.load_weights(filename)

        time_step = env.reset()
        render(env)

        total_reward = 0

        while not time_step.last():

            state = time_step.observation

            action = agent.choose_action(state)

            time_step = env.step(action)
            render(env)

            total_reward += time_step.reward

        print("Total reward: {}".format(total_reward))

    env.close()


if __name__ == "__main__":
    test_agent("cartpole", "balance_sparse",
               ["0", "50", "100", "150", "200", "250"])