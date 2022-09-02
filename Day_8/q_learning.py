import random

class QAgent:

    def __init__(self, actions, alpha = 0.4, gamma=0.9, random_seed=0):
        """
        The Q-values will be stored in a dictionary. Each key will be of the format: ((x, y), a). 
        params:
            actions (list): A list of all the possible action values.
            alpha (float): step size
            gamma (float): discount factor
        """
        self.Q = {}
        
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        random.seed(random_seed)

    def get_Q_value(self, state, action):
        """
        Get q value for a state action pair.
        params:
            state (tuple): (x, y) coords in the grid
            action (int): an integer for the action
        """
        return self.Q.get((state, action), 0.0) # Return 0.0 if state-action pair does not exist

    def act(self, state, epsilon=0.1):
        # Choose a random action
        if random.random() < epsilon:
            action = random.choice(self.actions)
        # Choose the greedy action
        else:
            action = self.greedy_action_selection(state)
        
        return action

    def learn(self, state, action, reward, next_state):
        """
        Q-Learning update
        """
        q_next = self.get_Q_value(
            state = next_state,
            action = self.greedy_action_selection(next_state)
            )

        q_current = self.Q.get((state, action), None) # If this is the first time the state action pair is encountered
        if q_current is None:
            self.Q[(state, action)] = reward
        else:
            self.Q[(state, action)] = q_current + (self.alpha * (reward + self.gamma * q_next - q_current))

    def greedy_action_selection(self, state):
        """
        Selects action with the highest Q-value for the given state.
        """
        # Get all the Q-values for all possible actions for the state
        q_values = [self.get_Q_value(state, action) for action in self.actions]
        maxQ = max(q_values)
        # There might be cases where there are multiple actions with the same high q_value. Choose randomly then
        count_maxQ = q_values.count(maxQ)
        if count_maxQ > 1:
            # Get all the actions with the maxQ
            best_action_indexes = [i for i in range(len(self.actions)) if q_values[i] == maxQ]
            action_index = random.choice(best_action_indexes)
        else:
            action_index = q_values.index(maxQ)
            
        return self.actions[action_index]