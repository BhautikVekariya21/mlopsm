# src/utils.py
import yaml
import logging
from pathlib import Path

def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)