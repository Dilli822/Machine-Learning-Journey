class Agent:
    def __init__(self):
        self.states = {'s1': 1, 's2': 2, 's3': 3}
        self.connections = {'s1': {'s2': 5, 's3': 2},
                            's2': {'s1': 1, 's3': 8},
                            's3': {'s1': 100, 's2': 5}}
        self.paths = {}

    def compute_highest_reward_transitions(self):
        for state in self.states:
            max_reward = 0
            next_state = None
            for next_state_candidate, reward in self.connections[state].items():
                if reward > max_reward:
                    max_reward = reward
                    next_state = next_state_candidate
            self.paths[state] = next_state

    def update_table(self):
        print("State\tValue")
        for state, value in self.states.items():
            print(f"{state}\t{value}")

    def update_connections(self):
        print("\Rewards:")
        for from_state, transitions in self.connections.items():
            for to_state, reward in transitions.items():
                print(f"{from_state} -> {to_state}: {reward}")

    def update_paths(self):
        print("\nActions of Agent:")
        for state, next_state in self.paths.items():
            print(f"{state} -> {next_state}")
            
if __name__ == "__main__":
    agent = Agent()
    
    # Compute highest reward transitions
    agent.compute_highest_reward_transitions()

    # Update and print the state table, connections, and paths
    agent.update_table()
    agent.update_connections()
    agent.update_paths()
