import argparse

from agent.runners.DreamerRunner import DreamerRunner
from agent_constrained.runners.DreamerRunner import DreamerRunner as DreamerRunnerConstrained

from configs import Experiment, SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
from configs.EnvConfigs import EnvCurriculumConfig #StarCraftConfig, 
from configs.flatland.RewardConfigs import FinishRewardConfig
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents, FiftyOfAgents, HundredOfAgents
from environments import Env, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="flatland", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--saving_name', type=str, default="model", help='Saving name for .pth model')
    return parser.parse_args()


def train_dreamer(exp, n_workers, saving_name, time_limit, constrained=False):
    if constrained:
        runner = DreamerRunnerConstrained(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    else:
        runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes, saving_name, time_limit)
    runner.learner.save_params("final_"+ saving_name)


def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
    env.close()


def get_env_info_flatland(configs):
    for config in configs:
        config.IN_DIM = FLATLAND_OBS_SIZE
        config.ACTION_SIZE = FLATLAND_ACTION_SIZE


def prepare_flatland_configs(env_name):
    if env_name == FlatlandType.FIVE_AGENTS:
        env_config = SeveralAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.TEN_AGENTS:
        env_config = PackOfAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.FIFTEEN_AGENTS:
        env_config = LotsOfAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.FIFTY_AGENTS:
        env_config = FiftyOfAgents(RANDOM_SEED + 200)
    elif env_name == FlatlandType.HUNDRED_AGENTS:
        env_config = HundredOfAgents(RANDOM_SEED + 300)
    else:
        raise Exception("Unknown flatland environment")
    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3,
                                                 timetable_config=AllAgentLauncherConfig())
    reward_config = RewardsComposerConfig((FinishRewardConfig(finish_value=10),
                                           NearRewardConfig(coeff=0.01),
                                           DeadlockPunishmentConfig(value=-5)))
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    get_env_info_flatland(agent_configs)
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": reward_config,
            "obs_builder_config": obs_builder_config}


if __name__ == "__main__":
    RANDOM_SEED = 29
    args = parse_args()
    if args.env == Env.FLATLAND:
        configs = prepare_flatland_configs(args.env_name)
    else:
        raise Exception("Unknown environment")
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    exp = Experiment(steps=2400000,
                     episodes=1700000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers, saving_name=args.saving_name, time_limit=60*60, constrained=True)
