from expected_sarsa import ExpectedSarsaAgent
from gridworld_env import Gridworld
from q_learning import QAgent
from sarsa import SarsaAgent

import matplotlib.pyplot as plt
import math

q_episode_scores = []
sarsa_episode_scores = []
expected_sarsa_episode_scores = []

# Epsilon greedy action selection
eps_decay_factor = 0.9 # After every episode, eps is 0.9 times the previous one
eps_min = 0.05 # 10% exploration is compulsory till the end

num_episodes = 200
alpha = 0.6
gamma = 0.999

random_seed = 60

# <------------Expected Sarsa ---------------------->
print('Stats for Expected Sarsa')
print('===================================================================================================================')
epsilon = 1
gridworld = Gridworld()
actions = gridworld.action_space

agent = ExpectedSarsaAgent(actions = actions, alpha = alpha, gamma = alpha, random_seed=random_seed)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()
best_score_episodes_taken = 0

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=epsilon)
        new_state, reward, done = gridworld.step(action)

        episode_score += reward

        agent.learn(state, action, reward, new_state, epsilon)

        state = new_state
        episode_actions.append(action)
        if done:
            break

    expected_sarsa_episode_scores.append(episode_score)
    # Decay epsilon
    epsilon = max(epsilon * eps_decay_factor, eps_min)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions
        best_score_episodes_taken = i_episode

    print(f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(expected_sarsa_episode_scores[:-20])/len(expected_sarsa_episode_scores)}, epsilon: {epsilon}', end='')

print(f'\nAfter {num_episodes}, average score: {sum(expected_sarsa_episode_scores)/len(expected_sarsa_episode_scores)}, Average(last 20): {sum(expected_sarsa_episode_scores[:-20])/len(expected_sarsa_episode_scores)}')
print(f'Best score: {best_score}, Sequence of actions: {[gridworld.num2action[action] for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
print('===================================================================================================================')

# <------------Sarsa ---------------------->
print('Stats for Sarsa')
print('===================================================================================================================')
epsilon = 1
gridworld = Gridworld()
actions = gridworld.action_space

agent = SarsaAgent(actions = actions, alpha = alpha, gamma = alpha, random_seed=random_seed)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()
best_score_episodes_taken = 0

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=epsilon)
        new_state, reward, done = gridworld.step(action)

        episode_score += reward

        new_state_action = agent.act(new_state)
        agent.learn(state, action, reward, new_state, new_state_action)

        state = new_state
        episode_actions.append(action)
        if done:
            break

    sarsa_episode_scores.append(episode_score)
    # Decay epsilon
    epsilon = max(epsilon * eps_decay_factor, eps_min)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions
        best_score_episodes_taken = i_episode

    print(f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(sarsa_episode_scores[:-20])/len(sarsa_episode_scores)}, epsilon: {epsilon}', end='')

print(f'\nAfter {num_episodes}, average score: {sum(sarsa_episode_scores)/len(sarsa_episode_scores)}, Average(last 20): {sum(sarsa_episode_scores[:-20])/len(sarsa_episode_scores)}')
print(f'Best score: {best_score}, Sequence of actions: {[gridworld.num2action[action] for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
print('===================================================================================================================')

# <------------Q-Learning ---------------------->
print('Stats for Q Learning')
print('===================================================================================================================')
epsilon = 1
gridworld = Gridworld()
actions = gridworld.action_space

agent = QAgent(actions = actions, alpha = alpha, gamma = alpha, random_seed=random_seed)

# Storing the path taken and score for the best episode
best_score = -math.inf
best_path_actions = list()
best_score_episodes_taken = 0

for i_episode in range(1, num_episodes+1):
    state = gridworld.reset()
    episode_score = 0
    episode_actions = []
    while True:
        action = agent.act(state, epsilon=epsilon)
        new_state, reward, done = gridworld.step(action)

        episode_score += reward

        agent.learn(state, action, reward, new_state)

        state = new_state
        episode_actions.append(action)
        if done:
            break

    q_episode_scores.append(episode_score)
    # Decay epsilon
    epsilon = max(epsilon * eps_decay_factor, eps_min)

    # For best episode data
    if episode_score > best_score:
        best_score = episode_score
        best_path_actions = episode_actions
        best_score_episodes_taken = i_episode

    print(f'\rEpisode: {i_episode}/{num_episodes}, score: {episode_score}, Average(last 20): {sum(q_episode_scores[:-20])/len(q_episode_scores)}, epsilon: {epsilon}', end='')

print(f'\nAfter {num_episodes}, average score: {sum(q_episode_scores)/len(q_episode_scores)}, Average(last 20): {sum(q_episode_scores[:-20])/len(q_episode_scores)}')
print(f'Best score: {best_score}, Sequence of actions: {[gridworld.num2action[action] for action in best_path_actions]}, Reached in {best_score_episodes_taken} episodes')
print('===================================================================================================================')


plt.plot(range(len(q_episode_scores)), q_episode_scores, label='Q Learning')
plt.plot(range(len(sarsa_episode_scores)), sarsa_episode_scores, label='Sarsa')
plt.plot(range(len(expected_sarsa_episode_scores)), expected_sarsa_episode_scores, label='Expected Sarsa')
plt.legend(loc="lower right")
plt.xlabel('Episodes ->')
plt.ylabel('Score ->')
plt.title('Training progress')
plt.savefig('comparision.png')
plt.show()
