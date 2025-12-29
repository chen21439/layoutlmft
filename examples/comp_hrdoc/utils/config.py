"""Configuration utilities for comp_hrdoc.

Handles config loading, GPU setup, and environment detection.

Config Structure:
    configs/
    ├── base.yaml     # 通用参数（模型、训练等）
    ├── dev.yaml      # dev 环境（本地路径、GPU等）
    └── test.yaml     # test 环境（服务器路径、GPU等）

Usage:
    # At the VERY START of any script, before importing torch:
    from examples.comp_hrdoc.utils.config import setup_environment, load_config
    env = setup_environment()  # Must be called before `import torch`

    import torch  # Now torch sees only configured GPU

    config = load_config(env)  # Load merged config
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Config directory
CONFIG_DIR = Path(__file__).parent.parent / "configs"


def get_env_from_argv() -> Optional[str]:
    """Extract --env value from command line arguments."""
    for i, arg in enumerate(sys.argv):
        if arg == "--env" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        elif arg.startswith("--env="):
            return arg.split("=", 1)[1]
    return None


def load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load a YAML file.

    Args:
        path: Path to YAML file

    Returns:
        Parsed config dict
    """
    if not path.exists():
        return {}

    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return _parse_simple_yaml(path)


def _parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """Simple YAML parser for basic key-value configs (fallback)."""
    config = {}
    current_section = config
    section_stack = [config]
    indent_stack = [0]

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            indent = len(line) - len(line.lstrip())

            while indent_stack and indent <= indent_stack[-1] and len(section_stack) > 1:
                section_stack.pop()
                indent_stack.pop()
            current_section = section_stack[-1]

            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()

                if value:
                    if value == "null" or value == "~":
                        current_section[key] = None
                    elif value == "true":
                        current_section[key] = True
                    elif value == "false":
                        current_section[key] = False
                    elif value.startswith('"') and value.endswith('"'):
                        current_section[key] = value[1:-1]
                    elif value.startswith('[') and value.endswith(']'):
                        # Simple list parsing
                        items = value[1:-1].split(',')
                        current_section[key] = [int(x.strip()) for x in items if x.strip()]
                    else:
                        try:
                            current_section[key] = int(value)
                        except ValueError:
                            try:
                                current_section[key] = float(value)
                            except ValueError:
                                current_section[key] = value
                else:
                    current_section[key] = {}
                    section_stack.append(current_section[key])
                    indent_stack.append(indent)
                    current_section = current_section[key]

    return config


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries. Override values take precedence.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(env: Optional[str] = None) -> Dict[str, Any]:
    """Load merged configuration for environment.

    Loads base.yaml and merges with environment-specific config (dev.yaml/test.yaml).

    Args:
        env: Environment name (dev/test). Auto-detected from argv if None.

    Returns:
        Merged config dict
    """
    if env is None:
        env = get_env_from_argv() or os.environ.get("COMP_HRDOC_ENV", "dev")

    # Load base config
    base_config = load_yaml_file(CONFIG_DIR / "base.yaml")

    # Load environment config
    env_config = load_yaml_file(CONFIG_DIR / f"{env}.yaml")

    # Merge configs (env overrides base)
    config = deep_merge(base_config, env_config)

    # Add environment info
    config["_env"] = env

    return config


def get_artifact_path(env: Optional[str] = None) -> str:
    """Get artifact directory path for environment.

    Args:
        env: Environment name (dev/test)

    Returns:
        Artifact directory path
    """
    config = load_config(env)
    return config.get("artifact", {}).get("root", "artifact")


def get_data_path(env: Optional[str] = None, key: str = "root") -> str:
    """Get data path for environment.

    Args:
        env: Environment name
        key: Data path key (root, train_dir, test_dir, features_dir)

    Returns:
        Data path
    """
    config = load_config(env)
    return config.get("data", {}).get(key, "")


def get_gpu_config(env: Optional[str] = None) -> Optional[str]:
    """Get GPU configuration for environment.

    Args:
        env: Environment name

    Returns:
        CUDA_VISIBLE_DEVICES string or None
    """
    config = load_config(env)
    return config.get("gpu", {}).get("cuda_visible_devices")


def setup_gpu(env: Optional[str] = None, gpu_id: Optional[str] = None) -> None:
    """Setup CUDA_VISIBLE_DEVICES before importing torch.

    MUST be called before `import torch`.

    Priority (highest to lowest):
        1. gpu_id parameter (command line arg)
        2. CUDA_VISIBLE_DEVICES environment variable
        3. Config file (gpu.cuda_visible_devices)

    Args:
        env: Environment name (dev/test)
        gpu_id: Explicit GPU ID (overrides all)
    """
    # 1. Command line arg has highest priority
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"[GPU] CUDA_VISIBLE_DEVICES={gpu_id} (from argument)")
        return

    # 2. Environment variable takes precedence over config
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f"[GPU] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (from environment)")
        return

    # 3. Config file as fallback
    cuda_devices = get_gpu_config(env)
    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        print(f"[GPU] CUDA_VISIBLE_DEVICES={cuda_devices} (from config)")


def setup_environment(env: Optional[str] = None, gpu_id: Optional[str] = None) -> str:
    """Full environment setup. Call before importing torch.

    Args:
        env: Environment name (dev/test). Auto-detected if None.
        gpu_id: Explicit GPU ID (overrides config)

    Returns:
        Resolved environment name
    """
    if env is None:
        env = get_env_from_argv()
    if env is None:
        env = os.environ.get("COMP_HRDOC_ENV", "dev")

    # Setup GPU
    setup_gpu(env, gpu_id)

    return env


def get_device():
    """Get torch device after setup. Call after importing torch."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_gpu_info():
    """Print GPU information. Call after importing torch."""
    import torch
    print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GPU] Device count: {torch.cuda.device_count()}")
        print(f"[GPU] Current device: {torch.cuda.current_device()}")
        print(f"[GPU] Device name: {torch.cuda.get_device_name(0)}")


def print_config(config: Dict[str, Any], indent: int = 0) -> None:
    """Pretty print configuration."""
    for key, value in config.items():
        if key.startswith("_"):
            continue
        prefix = "  " * indent
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_config(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")
