import torch
import torch.nn as nn
import torch.optim

from collections import defaultdict

from torchrl.modules import ProbabilisticActor, TanhNormal

from tensordict.nn import TensorDictModule

from scripts.data import DataCollectorFromEarthGymTester
from scripts.model import SimpleMLP
from scripts.model import MLPModelEOS
from scripts.model import TransformerEncoderModelEOS
from scripts.model import TransformerModelEOS
from scripts.client import Client
from scripts.utils import DataFromJSON

class Inference():
    """
    Proximal Policy Optimization (PPO).
    """
    def __init__(
            self,
            client: Client,
            conf: DataFromJSON,
            save_path: str = "./",
            input_path: str = "./",
            device: torch.device = torch.device("cpu")
        ):
        ########################### Parameters ###########################
        self._client = client
        self._conf = conf
        self._save_path = save_path
        self._input_path = input_path
        self._device = device
        ########################### Parameters ###########################

    def start(self):
        """
        Start the training process.
        """
        # Create the policy model
        policy_net = self.build_policy_net()

        # Create the PPO algorithm
        wrapper = InferenceWrapper(
            client=self._client,
            conf=self._conf,
            policy=policy_net,
            device=self._device
        )

        # Test the model
        wrapper.test(self._conf.test_steps)

    def build_policy_net(self):
        """
        Build the policy model.
        """
        # Add the configuration file properties of the architecture chosen
        for i in range(len(self._conf.archs_available)):
            if self._conf.archs_available[i]["name"] == self._conf.policy_arch:
                policy_conf: defaultdict = self._conf.archs_available[i].copy()
                break

        print(f"Using {policy_conf.pop('name')} architecture for the policy.")

        # Create the policy network
        if self._conf.policy_arch == "SimpleMLP":
            policy_conf["input_dim"] = self._conf.max_len * self._conf.state_dim
            policy_conf["output_dim"] = self._conf.action_dim
            policy_net = SimpleMLP(**policy_conf, device=self._device)
        elif self._conf.policy_arch == "MLP":
            policy_conf["in_dim"] = self._conf.max_len * self._conf.state_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_net = MLPModelEOS(**policy_conf, device=self._device)
        elif self._conf.policy_arch == "TransformerEncoder":
            policy_conf["src_dim"] = self._conf.state_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_conf["max_len"] = self._conf.max_len
            policy_net = TransformerEncoderModelEOS(**policy_conf, device=self._device)

            # Add forward hooks to the model
            self.add_forward_hooks(policy_net.transformer_encoder, encoder=True, decoder=False)
        elif self._conf.policy_arch == "Transformer":
            policy_conf["src_dim"] = self._conf.state_dim
            policy_conf["tgt_dim"] = self._conf.action_dim
            policy_conf["out_dim"] = self._conf.action_dim
            policy_conf["max_len"] = self._conf.max_len
            policy_net = TransformerModelEOS(**policy_conf, device=self._device)

            # Add forward hooks to the model
            self.add_forward_hooks(policy_net.transformer, encoder=True, decoder=True)
        else:
            raise ValueError(f"Policy architecture {self._conf.policy_arch} not available. Please choose from {[i['name'] for i in self._conf.archs_available]}.")
        
        # Import policy model
        policy_net.load_state_dict(torch.load(self._input_path + "/policy.pt"))

        return policy_net.to(self._device)
    
    def add_forward_hooks(self, model: nn.Module, encoder: bool=True, decoder: bool=True):
        """
        Add forward hooks to the model.
        """
        if encoder:
            # Register the hook to the model's encoder
            for layer in model.encoder.layers:
                layer.self_attn.register_forward_hook(self.attn_forward_hook)
        if decoder:
            # Register the hook to the model's decoder
            for layer in model.decoder.layers:
                layer.self_attn.register_forward_hook(self.attn_forward_hook)

    def attn_forward_hook(self, module, module_input, module_output):
        # Output is a tuple: (attn_output, attn_output_weights)
        # attn_output_weights is what we want to visualize
        # print("Forward hook called!")
        # print("Module:", module)
        # print("Module Input:", module_input)
        # print("Module Output:", module_output)
        attn_output_weights = module_output[1]
        # print("Output:", module_output)
        self.attn_weights_all_layers.append(attn_output_weights)

class InferenceWrapper():
    """
    Proximal Policy Optimization (PPO) <https://arxiv.org/abs/1707.06347> algorithm.
    """
    def __init__(
            self,
            client: Client,
            conf: object,
            policy: nn.Module,
            device: torch.device = torch.device("cpu")
        ):
        ########################### Parameters ###########################
        # General hyperparameters
        self._client = client
        self._conf = conf
        self._policy = policy

        # Cuda device
        self._device = device
        ########################### Parameters ###########################

        policy_td_module = TensorDictModule(
            module=self._policy,
            in_keys=["policy_observation"] if self._conf.policy_arch != "Transformer" else ["policy_observation", "actions_as_tgt"],
            out_keys=["loc", "scale"]
        ).to(self._device)

        self._actor = ProbabilisticActor(
            module=policy_td_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": torch.tensor([-1., -1.]), # e.g. tensor([-1., -1., -1.])
                "high": torch.tensor([1., 1.]), # e.g. tensor([1., 1., 1.])
            },
            return_log_prob=True
        ).to(self._device)

        self._collector = DataCollectorFromEarthGymTester(
            client=self._client,
            conf=self._conf,
            policy=self._actor,
            device=self._device
        )
    
    def test(self, n_steps: int = 10000):
        """
        Run the agent in the environment for a specified number of timesteps.
        """
        print("Testing the model...")
        self._collector.test(n_steps=n_steps)