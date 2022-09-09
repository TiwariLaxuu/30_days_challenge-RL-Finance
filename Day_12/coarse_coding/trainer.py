class Trainer:
    def __init__(self, state_manager, agent):
        self.state_manager = state_manager()
        self.agent = agent()

        self.total_reward = None
        self.last_action = None
        self.num_steps = None
        self.num_episodes = None

    def init(self, agent_init_info={}, env_init_info={}):
        # Initialize with parameters for state manager and agent
        self.state_manager.init(env_init_info)
        self.agent.init(agent_init_info)
        # Initialize different counters
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self):
        self.total_reward = 0.0
        self.num_steps = 1

        last_state = self.state_manager.start()
        self.last_action = self.agent.start(last_state)

        observation = (last_state, self.last_action)

        return observation

    def step(self):
        (reward, last_state, term) = self.state_manager.step(self.last_action)

        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.step(reward, last_state)
            roat = (reward, last_state, self.last_action, term)

        return roat
    
    def episode(self, max_steps_this_episode, animate=False):
        is_terminal = False
        positions = []
        self.start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.step()
            if animate:
                positions.append(rl_step_result[1][0])
            is_terminal = rl_step_result[3]
        return is_terminal, positions