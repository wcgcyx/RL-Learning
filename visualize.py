from dm_control import suite
from dqn import Agent
from display import render


def test_agent(
        domain,
        task,
        weights):

    env = suite.load(domain, task, visualize_reward=True)

    action_spec = env.action_spec()
    observation_spec = env.observation_spec()

    for weight in weights:

        print("Weight {}".format(weight))
        agent = Agent(action_spec, observation_spec)
        agent.e_greedy = 0
        agent.load_weights("weights_" + weight + ".h5")

        time_step = env.reset()
        render(env)

        total_reward = 0

        while not time_step.last():

            state = time_step.observation

            action = agent.choose_action(state)

            time_step = env.step(action)
            render(env)

            total_reward = time_step.reward

        print("Total reward: {}".format(total_reward))

    env.close()


if __name__ == "__main__":
    test_agent("cartpole", "balance_sparse",
               ["0", "100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"])