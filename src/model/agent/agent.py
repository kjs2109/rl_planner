
class PlanningAgent(object):
    def __init__(self, rl_agent) -> None:
        self.agent = rl_agent

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.agent, name)
    
    def get_log_prob(self, obs, action):
        return self.agent.get_log_prob(obs, action)
     
    def get_action(self, obs):
        return self.agent.get_action(obs)
            
    def push_memory(self, experience):
        self.agent.push_memory(experience)

    def update(self,):
        return self.agent.update()
    
    def save(self, *args, **kwargs ):
        self.agent.save(*args, **kwargs )

    def load(self, *args, **kwargs ):
        self.agent.load(*args, **kwargs)