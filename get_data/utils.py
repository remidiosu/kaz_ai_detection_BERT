from pathlib import Path
import yaml

def get_yaml_data(filename):
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    DATA_YAML = ROOT / "configs" / f"{filename}.yaml"
    
    with DATA_YAML.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data, ROOT


