from plots import Plots
from animation import Animation
from mountain_car import MountainCar
from trainer import Trainer
from agent import Agent
import numpy as np

def run(episodes):
    all_steps = []
    epsilon = 0
    # set agent and evviroment
    agent = Agent
    env = MountainCar
    # set initial_weights to a random float in between -0.001 and 0
    initial_weights = np.random.uniform(-0.001, 0)
    # dictionary for agent_info
    agent_info = {"num_tilings": 8, "num_tiles": 8, "iht_size": 4096, "epsilon": epsilon, "gamma": 1.0, "alpha": 0.1/8, "initial_weights": initial_weights, "num_actions": 3}
    # dictionary for env_info
    env_info = {"min_position": -1.2, "max_position": 0.6, "min_velocity": -0.07, "max_velocity": 0.07, "gravity": 0.0025, "action_discount": 0.001}
    trainer = Trainer(env, agent)
    trainer.init(agent_info, env_info)
    for episode in range(1, episodes+1):
        if episode % 5 == 0:
            print("RUN: {}".format(episode))
        # dictionary for agent_info
        agent_info = {"num_tilings": 8, "num_tiles": 8, "iht_size": 4096, "epsilon": epsilon, "gamma": 1.0, "alpha": 0.1/8, "initial_weights": trainer.agent.w, "num_actions": 3}
        trainer.init(agent_info, env_info)
        _, positions = trainer.episode(1000, episode == episodes)
        all_steps.append(trainer.num_steps)
        epsilon *= 0.9
        # add/remove +1 to num_runs for animation
        if episode == episodes:
            anim = Animation(positions[0], positions)
            anim.plot_curve()
    Plots.scatterplot(all_steps, (len(all_steps), 1000))

if __name__ == "__main__":
    episodes = 1
    run(episodes)