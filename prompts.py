import yaml
import os

def load_prompts(path=None):
    if path is None:
        # Default to prompts/prompts.yaml relative to this file
        path = os.path.join(os.path.dirname(__file__), "prompts", "prompts.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)