from agent.controllers.DreamerController import DreamerController
from agent_constrained.controllers.DreamerController import DreamerController as DreamerControllerConstrained

from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerControllerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()

        self.EXPL_DECAY = 0.9999
        self.EXPL_NOISE = 0.
        self.EXPL_MIN = 0.
        self.USE_SEQUENTIAL_IMITATION = True
        self.USE_HYBRID_IMITATION = False
        self.USE_DAGGER = False
        self.DAGGER_BETA = 0.9
        self.POLICY_TO_FOLLOW = "policy" # "expert" or "policy"
        self.ABLATION_WM = False
        self.USE_IMITATION = False
        self.EXPERT_TO_FOLLOW = "solver"  # "solver" or "pcc" or "random"

    def create_controller(self):
        return DreamerController(self)
    
    def create_controller_constrained(self):
        return DreamerControllerConstrained(self)
