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
        argparse.add_argument("--inpt", type=str, help="Configuration file.")

        args = argparse.parse_args()

        # Create agent
        client = Client(gym_host=args.host, gym_port=args.port)

        # Load the configuration file
        with open(f"{sys.path[0]}\\sac-configuration.json", "r") as file:
            config = json.load(file)

        # Create configuration object
        conf = DataFromJSON(config, "configuration")

        # Create the SAC algorithm
        infer = Inference(conf, client, args.inpt)

        # Start the SAC algorithm
        infer.start()
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        infer.client.shutdown_gym()
        