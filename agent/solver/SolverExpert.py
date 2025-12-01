import torch

from agent.solver.libPythonCBS import PythonCBS
from flatland.core.env_observation_builder import DummyObservationBuilder
from flatland.envs.agent_utils import RailAgentStatus


class SolverAgentObs(DummyObservationBuilder):
     pass

class SolverAgent:
    def __init__(self):
        print("SolverAgent init")
        self.type = "solver"
    
    def init_solver(self, env):
        self.env = env
        framework = "LNS"  # "LNS" for large neighborhood search
        default_group_size = env.get_num_agents() # max number of agents in a group.
        max_iterations = 1000
        stop_threshold = 10
        agent_priority_strategy = 3
        neighbor_generation_strategy = 3
        debug = False
        time_limit = 200.
        replan = True
        self.solver= PythonCBS(env, 
                        framework, 
                        time_limit, 
                        default_group_size, 
                        debug, 
                        replan,
                        stop_threshold,
                        agent_priority_strategy,
                        neighbor_generation_strategy)
        self.solver.search(1.1, max_iterations) # build the initials paths for the agent without malfunctions using PP & LNS
        self.solver.buildMCP()


    def act(self, step):
        """Return the actions for all the agents as one hot encoding"""
        actions =  self.solver.getActions(self.env, step, float(3.0))
        action_as_actor = self.transform_action(actions)
        tensor_list = [torch.tensor(v) for v in action_as_actor.values()]
        indices = torch.stack(tensor_list)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=3).float()
        return one_hot

    def transform_action(self, action_dict):
        for handle, value in action_dict.items():
            if self.env.agents[handle].status == RailAgentStatus.DONE or self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED:
                action_dict[handle] = 2
                continue # we can't get transition of the agent if it is done
            avail_actions = self.get_available_actions(handle)
            if value == 4: # if solver stops the agent then action stop (2)
                action_dict[handle] = 2
            elif value == 0: # if solver action is do nothing
                agent_moving = self.env.agents[handle].moving
                if agent_moving: # and the agent is moving then it keeps giong forward
                    if avail_actions == [2,3] or len(avail_actions)==1:
                        action_dict[handle] = 0
                    if avail_actions == [1,2] or len(avail_actions)==1:
                        action_dict[handle] = 1
                else: # if the agent is not moving then action stop (2)
                    action_dict[handle] = 2
            elif len(avail_actions) == 1: # if only one action is available then take it
                if avail_actions[0] == 2: # if the only action is forward then go forward (leftmost direction with only one direction possible=forward)
                    action_dict[handle] = 0 
                if avail_actions[0] == 1:
                    action_dict[handle] = 0 
                if avail_actions[0] == 3:
                    action_dict[handle] = 1
            elif len(avail_actions) > 1: # if more than one action is available
                if value == 1: # and (avail_actions == [1,2] or avail_actions == [1,3]): # if solver tells to go left then go left
                    action_dict[handle] = 0
                if value == 3: # and (avail_actions == [1,3] or avail_actions == [2,3]): # if solver tells to go right then go right
                    action_dict[handle] = 1
                if value == 2 and avail_actions == [2,3]: # if solver tells to go forward then go forward
                    action_dict[handle] = 0
                if value == 2 and avail_actions == [1,2]: # if solver tells to go forward then go forward
                    action_dict[handle] = 1
        return action_dict

    def get_available_actions(self, handle):
        agent = self.env.agents[handle]
        position = agent.position
        direction = agent.direction
        if agent.status == RailAgentStatus.READY_TO_DEPART:
            position = agent.initial_position
            direction = agent.initial_direction
        
        transitions = self.env.rail.get_transitions(*position, direction)
        available_actions = []
        for i in range(-1, 2): # 'L', 'F', 'R'
            new_dir = (direction + i + 4) % 4
            if transitions[new_dir]:
                available_actions.append(i + 2)
        return available_actions

    def reset(self):
        self.solver.clearMCP()