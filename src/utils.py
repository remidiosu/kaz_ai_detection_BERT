from pathlib import Path

def get_yaml_path(filename):
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent.parent
    DATA_YAML = ROOT / "configs" / f"{filename}.yaml"
    return DATA_YAML, ROOT


