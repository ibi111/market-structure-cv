import yaml
from pathlib import Path
from inference.live_inference import FVGOverlay



def load_config() -> dict:
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)




if __name__ == '__main__':
    config = load_config()
    app = FVGOverlay(config)
    app.run()