from copy import deepcopy
from enum import IntEnum

import ray
import torch
from flatland.envs.agent_utils import RailAgentStatus
from collections import defaultdict
import random
import numpy as np

from environments import Env

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.schedule_generators import sparse_schedule_generator
from flatland.envs.malfunction_generators import malfunction_from_params, MalfunctionParameters

from agent.solver.SolverExpert import SolverAgent
from agent.solver.GloutonExpert import SingleAgentNavifgationPolicy, SingleAgentNavigationObs
from flatland.core.grid.grid4_utils import get_new_position

class TrainAction(IntEnum):
    NOTHING = 0
    LEFT = 1
    FORWARD = 2
    RIGHT = 3
    STOP = 4

@ray.remote
class DreamerWorker:

    def __init__(self, idx, env_config, controller_config):
        self.runner_handle = idx
        self.env = env_config.create_env()
        self.controller = controller_config.create_controller()
        self.in_dim = controller_config.IN_DIM
        self.env_type = env_config.ENV_TYPE
        self.use_dagger = controller_config.USE_DAGGER
        self.dagger_initial_beta = controller_config.DAGGER_BETA
        self.policy_to_follow = controller_config.POLICY_TO_FOLLOW
        self.use_imitation = controller_config.USE_IMITATION
        self.expert_to_follow = controller_config.EXPERT_TO_FOLLOW

        if self.use_imitation and self.expert_to_follow == "solver":
            self.solver = SolverAgent()

    def _check_handle(self, handle):
        if self.env_type == Env.STARCRAFT:
            return self.done[handle] == 0
        else:
            return self.env.agents[handle].status in (RailAgentStatus.ACTIVE, RailAgentStatus.READY_TO_DEPART) \
                   and not self.env.obs_builder.deadlock_checker.is_deadlocked(handle)

    def _select_actions(self, state):
        avail_actions = []
        observations = []
        fakes = []
        if self.env_type == Env.FLATLAND:
            nn_mask = (1. - torch.eye(self.env.n_agents)).bool()
        else:
            nn_mask = None

        for handle in range(self.env.n_agents):
            if self.env_type == Env.FLATLAND:
                for opp_handle in self.env.obs_builder.encountered[handle]:
                    if opp_handle != -1:
                        nn_mask[handle, opp_handle] = False
            else:
                avail_actions.append(torch.tensor(self.env.get_avail_agent_actions(handle)))

            if self._check_handle(handle) and handle in state:
                fakes.append(torch.zeros(1, 1))
                observations.append(state[handle].unsqueeze(0))
            elif self.done[handle] == 1:
                fakes.append(torch.ones(1, 1))
                observations.append(self.get_absorbing_state())
            else:
                fakes.append(torch.zeros(1, 1))
                obs = torch.tensor(self.env.obs_builder._get_internal(handle)).float().unsqueeze(0)
                observations.append(obs)

        observations = torch.cat(observations).unsqueeze(0)
        av_action = torch.stack(avail_actions).unsqueeze(0) if len(avail_actions) > 0 else None
        nn_mask = nn_mask.unsqueeze(0).repeat(8, 1, 1) if nn_mask is not None else None
        actions = self.controller.step(observations, av_action, nn_mask)
        return actions, observations, torch.cat(fakes).unsqueeze(0), av_action

    def _wrap(self, d):
        for key, value in d.items():
            d[key] = torch.tensor(value).float()
        return d

    def get_absorbing_state(self):
        state = torch.zeros(1, self.in_dim)
        return state

    def augment(self, data, inverse=False):
        aug = []
        default = list(data.values())[0].reshape(1, -1)
        for handle in range(self.env.n_agents):
            if handle in data.keys():
                aug.append(data[handle].reshape(1, -1))
            else:
                aug.append(torch.ones_like(default) if inverse else torch.zeros_like(default))
        return torch.cat(aug).unsqueeze(0)

    def _check_termination(self, info, steps_done):
        if self.env_type == Env.STARCRAFT:
            return "episode_limit" not in info
        else:
            return steps_done < self.env.max_time_steps

    def run(self, dreamer_params, episode):
        self.controller.receive_params(dreamer_params)

        env_reset = self.env.reset()
        state = self._wrap(env_reset)
        steps_done = 0
        self.done = defaultdict(lambda: False)

        if self.use_imitation and self.expert_to_follow == "solver":
            self.solver.init_solver(self.env.env.env.env.env)

        while True:
            if self.use_dagger:
                use_expert = False
                use_policy = False
                initial_probability = self.dagger_initial_beta
                beta = initial_probability**(episode - 1)
                nombre = random.uniform(0, 1)
                if episode == 0:
                    use_expert = True
                else:
                    if nombre < beta:
                        use_expert = True
                    else:
                        use_policy = True

            if self.use_imitation and self.expert_to_follow == "solver":
                expert_action = self.solver.act(steps_done)
            actions, obs, fakes, av_actions = self._select_actions(state)
            steps_done += 1

            if self.use_dagger:
                if use_expert:
                    actions_to_use = solver_action
                elif use_policy:
                    actions_to_use = actions
            else:
                if self.policy_to_follow == "expert":
                    actions_to_use = solver_action
                elif self.policy_to_follow == "policy":
                    actions_to_use = actions

            expert_action = torch.zeros_like(actions)
            if self.use_imitation and (self.expert_to_follow == "pcc" or self.expert_to_follow == "random"):
                
                for i, a in enumerate(self.env.env.env.env.env.agents):
                    if self.expert_to_follow == "pcc":
                        action_pcc = self.get_shortest_path_action(self.env.env.env.env.env,i)
                        action_transfomed_mamba = self.transform_pcc_action_into_mamba_action(i, action_pcc)
                        expert_action[i,action_transfomed_mamba] = 1

                    if self.expert_to_follow == "random":
                        n = torch.randint(0, 3, (1,))
                        ind = n.item()
                        expert_action[i,ind] = 1

            next_state, reward, done, info = self.env.step([action.argmax() for i, action in enumerate(actions_to_use)])

    
            next_state, reward, done = self._wrap(deepcopy(next_state)), self._wrap(deepcopy(reward)), self._wrap(deepcopy(done))
            self.done = done
            self.controller.update_buffer({"action": (actions),
                                           "observation": obs,
                                           "reward": self.augment(reward),
                                           "done": self.augment(done),
                                           "fake": fakes,
                                           "avail_action": av_actions,
                                           "expert_action": (expert_action)})

            state = next_state
            if all([done[key] == 1 for key in range(self.env.n_agents)]):
                if self._check_termination(info, steps_done):
                    obs = torch.cat([self.get_absorbing_state() for i in range(self.env.n_agents)]).unsqueeze(0)
                    actions = torch.zeros(1, self.env.n_agents, actions.shape[-1])
                    index = torch.randint(0, actions.shape[-1], actions.shape[:-1], device=actions.device)
                    actions.scatter_(2, index.unsqueeze(-1), 1.)

                    if self.use_imitation:
                        expert_action = torch.zeros(1, self.env.n_agents, expert_action.shape[-1])
                        index = torch.randint(0, expert_action.shape[-1], expert_action.shape[:-1], device=expert_action.device)
                        expert_action.scatter_(2, index.unsqueeze(-1), 1.)

                    items = {"observation": obs,
                             "action": actions,
                             "reward": torch.zeros(1, self.env.n_agents, 1),
                             "fake": torch.ones(1, self.env.n_agents, 1),
                             "done": torch.ones(1, self.env.n_agents, 1),
                             "avail_action": torch.ones_like(actions) if self.env_type == Env.STARCRAFT else None,
                             "expert_action": expert_action}
                    self.controller.update_buffer(items)
                    self.controller.update_buffer(items)
                break

        if self.env_type == Env.FLATLAND:
            reward = sum(
                [1 for agent in self.env.agents if agent.status == RailAgentStatus.DONE_REMOVED]) / self.env.n_agents
        else:
            reward = 1. if 'battle_won' in info and info['battle_won'] else 0.

        if self.use_imitation and self.expert_to_follow == "solver":
            self.solver.reset()

        return self.controller.dispatch_buffer(), {"idx": self.runner_handle,
                                                   "reward": reward,
                                                   "steps_done": steps_done}

    
    def get_available_actions(self, handle):
        agent = self.env.env.agents[handle]
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

    def get_shortest_path_action(self, env, handle: int = 0): # -> List[List[int]]:

        agent = env.agents[handle]
        if agent.position:
            possible_transitions = env.rail.get_transitions(*agent.position, agent.direction)
        else:
            possible_transitions = env.rail.get_transitions(*agent.initial_position, agent.direction)
        # print(f"possible_transitions {possible_transitions}")

        num_transitions = np.count_nonzero(possible_transitions)

        # Start from the current orientation, and see which transitions are available;
        # organize them as [left, forward, right], relative to the current orientation
        # If only one transition is possible, the forward branch is aligned with it.
        if num_transitions == 1:
            observation = [0, 1, 0]
        else:
            min_distances = []
            for direction in [(agent.direction + i) % 4 for i in range(-1, 2)]:
                if possible_transitions[direction]:
                    new_position = get_new_position(agent.position, direction)
                    min_distances.append(
                        self.env.distance_map.get()[handle, new_position[0], new_position[1], direction])
                else:
                    min_distances.append(np.inf)

            observation = [0, 0, 0]
            observation[np.argmin(min_distances)] = 1
        return observation

    def transform_pcc_action_into_mamba_action(self, handle, action):
        if self.env.agents[handle].status == RailAgentStatus.DONE_REMOVED:
            return 2
        if action == [0,0,0]:
            return 0
        available_actions = self.get_available_actions(handle)
        if action == [0,0,1] and len(available_actions) == 1:
            return available_actions[0]
        if action == [1,0,0]:
            return 0
        if action == [0,0,1]:
            return 1
        if action == [0,1,0]:
            if len(available_actions) == 1:
                return 0
            else:
                agent = self.env.env.agents[handle]
                if agent.status == RailAgentStatus.READY_TO_DEPART:
                    position = agent.initial_position
                    direction = agent.initial_direction
                else:
                    position = agent.position
                    direction = agent.direction
                transitions = self.env.rail.get_transitions(*position, direction)

                if transitions[3]==1 and direction==0:
                    return 1
                if transitions[0]==1 and direction==1:
                    return 1
                if transitions[1]==1 and direction==2:
                    return 1
                if transitions[2]==1 and direction==3:
                    return 1
                if transitions[1]==1 and direction==0:
                    return 0
                if transitions[2]==1 and direction==1:
                    return 0
                if transitions[3]==1 and direction==2:
                    return 0
                if transitions[0]==1 and direction==3:
                    return 0
            return -1