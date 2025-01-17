import json
import sys
import argparse
import traceback

from scripts.infer import Inference
from scripts.utils import DataFromJSON
from scripts.client import Client

if __name__ == "__main__":
    try:
        # Gather the arguments
        argparse = argparse.ArgumentParser()

        argparse.add_argument("--host", default="localhost", type=str, help="Host address.")
        argparse.add_argument("--port", default=5555, type=int, help="Port number.")
        argparse.add_argument("--save", type=str, help="Plot save path.")
        argparse.add_argument("--inpt", type=str, help="Model path.")

        args = argparse.parse_args()

        # Create agent
        client = Client(gym_host=args.host, gym_port=args.port)

        # Load the configuration file
        with open(f"{sys.path[0]}\\infer-configuration.json", "r") as file:
            config = json.load(file)

        # Create configuration object
        conf = DataFromJSON(config, "configuration")

        # Create the Inference object
        infer = Inference(conf, client, args.save, args.inpt)

        # Start the inference
        infer.start()
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        infer.client.shutdown_gym()
        