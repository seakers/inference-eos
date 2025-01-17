import os
import numpy as np
import matplotlib.pyplot as plt

from scripts.client import Client
from scripts.model import *
from scripts.utils import *

RT = 6371.0 # Earth radius in km

class Inference():
    """
    Class to represent the Soft Actor-Critic algorithm. Children class of nn.Module.
    """
    def __init__(self, conf: DataFromJSON, client: Client, save_path: str, input_path: str):
        self.__role_type = "Inference"
        self.__conf = conf
        self.client = client
        self.save_path = save_path
        self.input_path = input_path
        self.set_properties(conf)
        self.attn_weights_all_layers = []
        self.attn_weights_all_layers_all_runs = []

    def set_properties(self, conf: DataFromJSON):
        """
        Set the properties of the SAC object.
        """
        for key, value in conf.__dict__.items():
            if not key.startswith("__"):
                setattr(self, key, value)

    def start(self):
        """
        Start the inference procedure.
        """
        # Load the model
        model = self.load_model()

        # Adsd forward hooks to the model
        self.add_forward_hooks(model)

        # Run the inference
        self.run(model)

    def load_model(self) -> EOSModel:
        """
        Create the entities for the SAC algorithm.
        """
        # Create the embedder object for states
        states_embedder = FloatEmbedder(
            input_dim=self.state_dim,
            embed_dim=self.d_model
        )
        
        # Create the embedder object for actions
        actions_embedder = FloatEmbedder(
            input_dim=self.action_dim,
            embed_dim=self.d_model
        )
        
        # Create the positional encoder object
        pos_encoder = PositionalEncoder(
            d_model=self.d_model,
            max_len=self.max_len,
            dropout=self.pos_dropout
        )

        # Create the transformer model
        transformer = EOSTransformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.transformer_dropout,
            activation=self.activation,
            batch_first=self.batch_first
        )
        
        # Create a linear outside stochastic layer called projector
        stochastic_projector = StochasticProjector(
            d_model=self.d_model,
            action_dim=self.action_dim
        )
        
        # Create the model object
        model = EOSModel(
            state_embedder=states_embedder,
            action_embedder=actions_embedder,
            pos_encoder=pos_encoder,
            transformer=transformer,
            projector=stochastic_projector
        )

        # Load the previous models if they exist
        if os.path.exists(f"{self.input_path}\\model.pth"):
            print("Loading model...")
            model.load_state_dict(torch.load(f"{self.input_path}\\model.pth", weights_only=True))
        else:
            raise ImportError("There is no model to load.")

        return model
    
    def add_forward_hooks(self, model: EOSModel):
        """
        Add forward hooks to the model.
        """
        # Register the hook to the model's encoder
        for layer in model.transformer.encoder.layers:
            layer.self_attn.register_forward_hook(self.attn_forward_hook)

        # Register the hook to the model's decoder
        for layer in model.transformer.decoder.layers:
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
    
    def run(self, model: EOSModel):
        """
        Run the inference procedure.
        """
        # Set the model to evaluation
        model.eval()

        # Initialize the environment
        list_states: list[torch.Tensor] = []
        list_actions: list[torch.Tensor] = []

        # Loop over all agents
        for agt in self.agents:
            # Sending data to get the initial state
            sending_data = {
                "agent_id": agt,
                "action": {
                    "d_pitch": 0,
                    "d_roll": 0
                },
                "delta_time": 0
            }
            state, _, _ = self.client.get_next_state("get_next", sending_data)

            # Normalize the state given by the environment
            vec_state = self.normalize_state(state)

            # Input tensor of 1 batch and 1 sequence of state_dim dimensional states
            states = torch.FloatTensor([[vec_state]])

            # Input tensor of 1 batch and 1 sequence of action_dim dimensional actions (equal to 0)
            actions = torch.FloatTensor([[[0 for _ in range(self.action_dim)]]])

            # Initialize the state and action tensors
            list_states += [states]
            list_actions += [actions]

        done = False
        print("Starting inference...")

        # Loop until environment is done
        while not done:
            # Loop over all agents
            for idx, agt in enumerate(self.agents):
                if self.debug:
                    print(f"Agent {agt}:")

                # Get the state and action tensors
                states = list_states[idx]
                actions = list_actions[idx]

                # Adjust the maximum length of the states and actions
                states = states[:, -self.max_len:, :]
                actions = actions[:, -self.max_len:, :]

                # Get the action from the model
                stochastic_actions = model(states, actions)

                # Store the attention weights
                self.store_attn_weights()

                # Select the last stochastic action
                a_sto = stochastic_actions[-1, -1, :]

                # Sample and convert the action
                _, a = model.reparametrization_trick(a_sto)

                if self.debug:
                    print(f"    In state {states[-1, -1, :]} took action {a}.")

                # Sending data to get the next state
                sending_data = {
                    "agent_id": agt,
                    "action": {
                        "d_pitch": a[0].item() * 90,
                        "d_roll": a[1].item() * 180
                    },
                    "delta_time": self.time_increment
                }
                state, reward, done = self.client.get_next_state("get_next", sending_data)

                # Normalize the state
                vec_state = self.normalize_state(state if state is not None else {})

                if self.debug:
                    print(f"    Got state {state} and reward {reward}.")

                # Break if time is up
                if done:
                    print("Time is up!")
                    break

                # Get the next state
                s_next = torch.FloatTensor(vec_state)
                # --------------- Environment's job to provide info ---------------

                # Add it to the states
                states = torch.cat([states, s_next.unsqueeze(0).unsqueeze(0)], dim=1)

                # Add it to the actions
                actions = torch.cat([actions, a.unsqueeze(0).unsqueeze(0)], dim=1)

                # Adjust the maximum length of the states and actions
                states = states[:, -self.max_len:, :]
                actions = actions[:, -self.max_len:, :]

                # Replace the states and actions lists
                list_states[idx] = states
                list_actions[idx] = actions

        print("Inference finished! Reward plots are available at Earth Gym.")
        print("Generating attention plots...")
        self.generate_attention_plots(model)

    def normalize_state(self, state: dict) -> list:
        """
        Normalize the action dictionary to a list.
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
    
    def store_attn_weights(self):
        """
        Store the attention weights.
        """
        self.attn_weights_all_layers_all_runs.append(self.attn_weights_all_layers)
        self.attn_weights_all_layers = []

    def generate_attention_plots(self, model: EOSModel):
        """
        Generate the attention plots.
        """
        # Create the attention directory
        if not os.path.exists(f"{self.save_path}\\attention"):
            os.makedirs(f"{self.save_path}\\attention")

        num_encoder_layers = model.transformer.encoder.num_layers
        num_decoder_layers = model.transformer.decoder.num_layers

        # Find a square distribution for the plot
        rows = int(np.ceil(np.sqrt(num_encoder_layers + num_decoder_layers)))
        cols = int(np.ceil((num_encoder_layers + num_decoder_layers) / rows))

        # Create a plot for the attention weights
        fig = plt.figure(figsize=(10, 10))

        for run_idx, attn_weights_all_layers in enumerate(self.attn_weights_all_layers_all_runs):
            for i, attn_weights in enumerate(attn_weights_all_layers):
                if len(attn_weights[0]) < self.max_len:
                    print(f"Skipping because there are only {len(attn_weights[0])} sequences...")
                    continue_plot = False
                else:
                    ax = fig.add_subplot(rows, cols, i+1)
                    ax.matshow(attn_weights[0].detach().numpy(), cmap='viridis')
                    if i < num_encoder_layers:
                        ax.set_title(f"Encoder Layer {i+1}")
                    else:
                        ax.set_title(f"Decoder Layer {i+1-num_decoder_layers}")
                    ax.set_xticks([0, self.max_len-1])
                    ax.set_yticks([0, self.max_len-1])
                    plt.colorbar(ax.matshow(attn_weights[0].detach().numpy(), cmap='viridis'))
                    continue_plot = True

            if continue_plot:
                plt.tight_layout()
                plt.savefig(f"{self.save_path}\\attention\\attention_{run_idx}.png", dpi=500)
                plt.clf()

        