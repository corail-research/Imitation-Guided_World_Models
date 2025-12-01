import ray
import wandb
import time 

from agent.workers.DreamerWorker import DreamerWorker


class DreamerServer:
    def __init__(self, n_workers, env_config, controller_config, model):
        ray.init()

        self.workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
        self.tasks = [worker.run.remote(model, 0) for worker in self.workers]

    def append(self, idx, update, episode):
        self.tasks.append(self.workers[idx].run.remote(update, episode))

    def run(self):
        done_id, tasks = ray.wait(self.tasks)
        self.tasks = tasks
        recvs = ray.get(done_id)[0]
        return recvs


class DreamerRunner:

    def __init__(self, env_config, learner_config, controller_config, n_workers):
        self.n_workers = n_workers
        self.learner = learner_config.create_learner()
        self.starting_time = time.time()
        if learner_config.FINETUNE:
            self.learner.load_params(learner_config.FINETUNE_PATH)
        self.server = DreamerServer(n_workers, env_config, controller_config, self.learner.params())
        self.max_reward = 0.6

    def run(self, max_steps=10 ** 10, max_episodes=10 ** 10, saving_name=None, time_limit=60*60*24):
        cur_steps, cur_episode = 0, 0

        wandb.define_metric("steps")
        wandb.define_metric("reward", step_metric="steps")

        while True:
            rollout, info = self.server.run()
            self.learner.step(rollout, cur_steps, max_steps)
            cur_steps += info["steps_done"]
            cur_episode += 1
            wandb.log({'reward': info["reward"], 'steps': cur_steps})

            print(cur_episode, self.learner.total_samples, info["reward"], time.time()-self.starting_time)
            if cur_episode >= max_episodes or cur_steps >= max_steps or (time.time()-self.starting_time)>time_limit:
                break
            self.server.append(info['idx'], self.learner.params(), cur_episode)
            if cur_episode%500==0:
                self.learner.save_params("episode_"+str(cur_episode)+saving_name)
            if info["reward"] > self.max_reward:
                self.max_reward = info["reward"]
                self.learner.save_params("best_episode_"+str(cur_episode)+saving_name)

