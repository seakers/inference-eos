import torch
import numpy as np

from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from scripts.client import Client

RT = 6371.0 # Earth radius in km

class DataCollectorFromEarthGymTester():
    def __init__(
            self,
            client: Client,
            conf: object,
            policy: ProbabilisticActor,
            device: torch.device = torch.device("cpu")
        ):
        # Init variables
        self._client = client
        self._conf = conf
        self._policy = policy
        self._device = device

        # Additional variables
        self._current_step = 0
        self._traj_id = 0
        self._step_count = 0

        self._conf.trajectory_len = float(self._conf.trajectory_len)

        # Warm-up the agent in the environment
        self.initialize_env()
    
    def initialize_env(self):
        """
        Initialize the environment. Make the agent do a dummy move to stabilize the observation state.
        """
        sending_data = {
            "agent_id": 0,
            "action": {
                "d_pitch": 0,
                "d_roll": 0
            },
            "delta_time": 0
        }
        state, _, _ = self._client.get_next_state("get_next", sending_data)

        # Normalize the state given by the environment
        vec_state = self.normalize_state(state)

        # Input tensor of 1 batch and 1 sequence of state_dim dimensional states
        self._states = torch.tensor([[vec_state]], dtype=torch.float32, device=self._device)

        # Input tensor of 1 batch and 1 sequence of action_dim dimensional actions
        self._actions = torch.tensor([[[0] * self._conf.action_dim]], dtype=torch.float, device=self._device)

        # Make max_len dummy moves to have a long enough observation
        self.n_dummy_moves(n=self._conf.max_len)

    def n_dummy_moves(self, n: int):
        """
        Do n dummy moves to stabilize the environment.
        """
        for _ in range(n):
            _, _, _, _, _, _ = self.move_once(torch.tensor([0] * self._conf.action_dim, dtype=torch.float32))

    def move_once(self, action: torch.Tensor):
        """
        Do an environment step for the SAC algorithm.
        """
        with torch.no_grad():
            # Get the current observation
            curr_policy_obs, curr_value_fn_obs = self.prettify_observation(self._states, self._actions)

            # --------------- Environment's job to provide info ---------------
            sending_data = {
                "agent_id": 0,
                "action": {
                    "d_pitch": action[(-1,) * (action.dim() - 1) + (0,)].item() * self._conf.a_conversions[0],
                    "d_roll": action[(-1,) * (action.dim() - 1) + (1,)].item() * self._conf.a_conversions[1]
                },
                "delta_time": self._conf.time_increment
            }
            
            state, reward, done = self._client.get_next_state("get_next", sending_data)

            # Break if time is up
            if done:
                print("Time is up!")
                return None, None, None, None, None, True

            # Normalize the state
            vec_state = self.normalize_state(state)

            # Get the reward
            r = torch.tensor(reward * self._conf.reward_scale, dtype=torch.float32)

            # Get the next state
            s_next = torch.tensor(vec_state, dtype=torch.float32)
            # --------------- Environment's job to provide info ---------------

            # Add it to the states
            while s_next.dim() < self._states.dim():
                s_next = s_next.unsqueeze(0)
            self._states = torch.cat([self._states, s_next.to(self._device)], dim=1)

            # Add it to the actions
            while action.dim() < self._actions.dim():
                action = action.unsqueeze(0)
            self._actions = torch.cat([self._actions, action.to(self._device)], dim=1)

            # Adjust the maximum length of the states and actions
            self._states = self._states[:, -self._conf.max_len:, :]
            self._actions = self._actions[:, -self._conf.max_len:, :]

            # Arrange the next observation as the model expects
            next_policy_obs, next_value_fn_obs = self.prettify_observation(self._states, self._actions)

            return curr_policy_obs, curr_value_fn_obs, next_policy_obs, next_value_fn_obs, r, False
        
    def prettify_observation(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Arrange the next observation as the model expects.
        """
        # Clone the tensors to avoid in-place operations
        states = states.clone()
        actions = actions.clone()

        sequential_models = ["Transformer", "TransformerEncoder"]

        # See if the policy is a sequential model
        if self._conf.policy_arch in sequential_models:
            policy_obs = states
        else:
            policy_obs = states.view(-1)

        # Check if we have a transformer policy but not a transformer value function
        if self._conf.policy_arch == "Transformer" and self._conf.value_fn_arch != "Transformer":
            value_fn_obs = torch.cat([states, actions], dim=-1)
        else:
            value_fn_obs = states

        # See if the value function is a sequential model
        if self._conf.value_fn_arch not in sequential_models:
            value_fn_obs = value_fn_obs.view(-1)

        return policy_obs, value_fn_obs
        
    def test(self, n_steps: int=10000):
        """
        Test the environment.
        """
        total_rewards = []

        self._policy.eval()
        for i in range(int(n_steps)):
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                observation, _ = self.prettify_observation(self._states, self._actions)
                actions_as_tgt = self._actions.clone()
                loc, scale, action, log_prob = self._policy(observation) if self._conf.policy_arch != "Transformer" else self._policy(observation, actions_as_tgt)
                curr_policy_obs, curr_value_fn_obs, next_policy_obs, next_value_fn_obs, reward, done = self.move_once(action)

            if done:
                n_steps = i
                # total_rewards.append(reward.detach().item())
                break

            total_rewards.append(reward.detach().item())

        print(f"Average rewards per step: {sum(total_rewards)/n_steps:.4f}")
        
    def normalize_state(self, state: dict) -> list:
        """
        Normalize the state dictionary to a list.
        """
        # Conversion dictionary: each has two elements, the first is the gain and the second is the offset
        conversion_dict = {
            "a": (1/RT, -1), "e": (1, 0), "i": (1/180, 0), "raan": (1/360, 0), "aop": (1/360, 0), "ta": (1/360, 0), # orbital elements
            "az": (1/360, 0), "el": (1/180, 0.5), # azimuth and elevation
            "pitch": (1/180, 0.5), "roll": (1/360, 0.5), # attitude
            "detic_lat": (1/180, 0.5), "detic_lon": (1/360, 0), "detic_alt": (1/RT, 0), # nadir position
            "lat": (1/180, 0.5), "lon": (1/360, 0), "priority": (1/10, 0) # targets clues
        }

        vec_state = []
        for key, value in state.items():
            if key.startswith("lat_") or key.startswith("lon_") or key.startswith("priority_"):
                key = key.split("_")[0]
            vec_state.append(value * conversion_dict[key][0] + conversion_dict[key][1])

        return vec_state