import cv2
import numpy as np
import torch
import random

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2', 'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch',
                      'walker-walk', 'reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2,
                                'cheetah': 4, 'ball_in_cup': 6, 'walker': 2, 'humanoid': 2, 'fish': 2, 'acrobot': 4}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 **
                                                         bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(
        2, 0, 1), dtype=torch.float32)  # Resize and put channel first
    # Quantise, centre and dequantise inplace
    preprocess_observation_(images, bit_depth)
    return images.unsqueeze(dim=0)  # Add batch dimension


class MountainEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from mountain.mountain import MountainOfDeath, MountainOfDeathAction
        self.max_episode_length = max_episode_length
        self._env = MountainOfDeath()
        self._t = 0
        self._actions = list(MountainOfDeathAction)
        # A list of one-hot tensors representing actions
        self._action_tensors = [
            torch.tensor(
                [1 if i == j else -1 for i in range(len(self._actions))])
            for j in range(len(self._actions))
        ]  # Can't use a dictionary unfortunately because tensor equality is screwed

    def reset(self):
        self._env.reset()
        self._t = 0
        return self._env.render()

    def step(self, action):
        decoded_action = self.__tensor_to_action__(action)
        # Do action
        reward, violation = self._env.step(decoded_action)
        observation = torch.tensor(self._env.render(), dtype=torch.float32)
        self._t += 1
        done = self._t == self.max_episode_length
        if done:
            print("DONE")
        return observation, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return 1

    @property
    def action_size(self):
        return len(self._actions)

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return random.choice(self._action_tensors)

    # Convert a tensor into a MountainOfDeathAction
    def __tensor_to_action__(self, t):
        # Assume action to be in 1-hot representation
        idx = t.cpu().argmax()
        # Find MountainOfDeathAction
        return self._actions[idx.item()]

class DrivingEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, stickiness):
        from driving.driving import CarEnvironment, CarAction
        self.max_episode_length = max_episode_length
        self._env = CarEnvironment(p_sticky=stickiness)
        self._t = 0
        self._actions = list(CarAction)
        # A list of one-hot tensors representing actions
        self._action_tensors = [
            torch.tensor(
                [1 if i == j else -1 for i in range(len(self._actions))])
            for j in range(len(self._actions))
        ]  # Can't use a dictionary unfortunately because tensor equality is screwed

    def reset(self):
        self._env.reset()
        self._t = 0
        return self._env.render()

    def step(self, action):
        # decoded_action = self.__tensor_to_action__(action)
        # Do action
        reward, violation, done = self._env.step(action)
        observation = torch.tensor(self._env.render(), dtype=torch.float32)
        self._t += 1
        done = self._t == self.max_episode_length or done
        return observation, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return 2

    @property
    def action_size(self):
        return 1

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.tensor(random.random() * 2 - 1)

    # Convert a tensor into a MountainOfDeathAction
    def __tensor_to_action__(self, t):
        # Assume action to be in 1-hot representation
        idx = t.cpu().argmax()
        # Find MountainOfDeathAction
        return self._actions[idx.item()]


class GridEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from gridworld.gridworld import GridWorld, GridWorldAction

        self.max_episode_length = max_episode_length
        self.bit_depth = bit_depth
        self._env = GridWorld((3, 3), fixed_seed=seed, no_enemies=0)
        self._t = 0

        self._actions = list(GridWorldAction)
        # A list of one-hot tensors representing actions
        self._action_tensors = [
            torch.tensor(
                [1 if i == j else -1 for i in range(len(self._actions))])
            for j in range(len(self._actions))
        ]  # Can't use a dictionary unfortunately because tensor equality is screwed

    def reset(self):
        self._env.reset()
        self._t = 0
        return self.observation()

    def observation(self):
        flat_obs = torch.as_tensor(list(self._env._agent_position) + \
            [coord for unsafe_pos in self._env._unsafe_positions for coord in unsafe_pos] + \
            list(self._env._target_position), dtype=torch.float)
        return torch.reshape(flat_obs, (1, flat_obs.size(dim=0)))
    
    def step(self, action):
        decoded_action = self.__tensor_to_action__(action)
        # Do action
        reward, violation = self._env.step(decoded_action)
        # observation = _images_to_observation(
        #     self._env.render(), self.bit_depth)
        observation = self.observation()

        self._t += 1
        done = self._t == self.max_episode_length
        if done:
            print("DONE")
        return observation, reward, violation, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return len(self._env.size) + sum(len(u) for u in self._env._unsafe_positions) + len(self._env._target_position)

    @property
    def action_size(self):
        return len(self._actions)

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return random.choice(self._action_tensors)

    # Convert a tensor into a GridWorldAction
    def __tensor_to_action__(self, t):
        # Assume action to be in 1-hot representation
        idx = t.cpu().argmax()
        # Find GridWorldAction
        return self._actions[idx.item()]


class ControlSuiteEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self.symbolic = symbolic
        self._env = suite.load(
            domain_name=domain, task_name=task, task_kwargs={'random': seed})
        if not symbolic:
            self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        if action_repeat != CONTROL_SUITE_ACTION_REPEATS[domain]:
            print('Using action repeat %d; recommended action repeat for domain is %d' % (
                action_repeat, CONTROL_SUITE_ACTION_REPEATS[domain]))
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(np.concatenate([np.asarray([obs]) if isinstance(
                obs, float) else obs for obs in state.observation.values()], axis=0), dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(
                self._env.physics.render(camera_id=0), self.bit_depth)
        return observation, reward, done

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class GymEnv():
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        import gym
        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(
                state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(
                self._env.render(mode='rgb_array'), self.bit_depth)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


def Env(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, stickiness):
    # TODO: remove this short circuit
    # if symbolic:
    #     return DrivingEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth, stickiness)
    # else:
    return GridEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    if env in GYM_ENVS:
        return GymEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)
    elif env in CONTROL_SUITE_ENVS:
        return ControlSuiteEnv(env, symbolic, seed, max_episode_length, action_repeat, bit_depth)


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_class, env_args, env_kwargs, n):
        self.n = n
        self.envs = [env_class(*env_args, **env_kwargs) for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

     # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        # Done mask to blank out observations and zero rewards for previously terminated environments
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
        observations, rewards, violations, dones = zip(
            *[env.step(action) for env, action in zip(self.envs, actions)])
        # Env should remain terminated if previously terminated
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones
        observations, rewards, violations, dones = torch.cat(observations), torch.tensor(
            rewards, dtype=torch.float32), torch.tensor(violations, dtype=torch.float32),  torch.tensor(dones, dtype=torch.uint8)
        observations[done_mask] = 0
        rewards[done_mask] = 0
        violations[done_mask] = 0
        return observations, rewards, violations, dones

    def close(self):
        [env.close() for env in self.envs]
