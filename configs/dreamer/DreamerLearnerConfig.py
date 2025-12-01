from agent.learners.DreamerLearner import DreamerLearner
from agent_constrained.learners.DreamerLearner import DreamerLearner as DreamerLearnerConstrained
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4
        self.VALUE_LR = 5e-4
        self.CAPACITY = 500000
        self.MIN_BUFFER_SIZE = 2500
        self.MODEL_EPOCHS = 40
        self.EPOCHS = 4
        self.PPO_EPOCHS = 5
        self.SUPERVISED_EPOCHS = 1000
        self.MODEL_BATCH_SIZE = 40
        self.BATCH_SIZE = 16
        self.SEQ_LENGTH = 50
        self.N_SAMPLES = 1
        self.TARGET_UPDATE = 1
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0
        self.HORIZON = 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 0.99998
        self.GRAD_CLIP_POLICY = 100.
        self.USE_SEQUENTIAL_IMITATION = True
        self.STEPS_SEQUENTIAL_SUPERVISED_PHASE = 800000
        self.STEPS_SEQUENTIAL_RL_PHASE = 950000
        self.USE_HYBRID_IMITATION = False
        self.USE_DAGGER = False
        self.FINETUNE = False
        self.FINETUNE_PATH = "path/to/checkpoint.pth"
        self.ABLATION_WM = False


    def create_learner(self):
        return DreamerLearner(self)
    
    def create_learner_constrained(self):
        return DreamerLearnerConstrained(self)
