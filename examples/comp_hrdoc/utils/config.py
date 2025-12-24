"""Configuration utilities for comp_hrdoc.

Handles config loading, GPU setup, and environment detection.

Usage:
    # At the VERY START of any script, before importing torch:
    from examples.comp_hrdoc.utils.config import setup_environment
    setup_environment()  # Must be called before `import torch`

    import torch  # Now torch sees only configured GPU
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


def load_yaml_config(config_name: str = "order.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_name: Name of config file in configs/ directory

    Returns:
        Parsed config dict
    """
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: parse simple yaml without pyyaml
        return _parse_simple_yaml(config_path)


def _parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """Simple YAML parser for basic key-value configs."""
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

            # Pop sections based on indent
            while indent_stack and indent <= indent_stack[-1] and len(section_stack) > 1:
                section_stack.pop()
                indent_stack.pop()
            current_section = section_stack[-1]

            if ":" in stripped:
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()

                if value:
                    # Parse value
                    if value == "null" or value == "~":
                        current_section[key] = None
                    elif value == "true":
                        current_section[key] = True
                    elif value == "false":
                        current_section[key] = False
                    elif value.startswith('"') and value.endswith('"'):
                        current_section[key] = value[1:-1]
                    else:
                        try:
                            current_section[key] = int(value)
                        except ValueError:
                            try:
                                current_section[key] = float(value)
                            except ValueError:
                                current_section[key] = value
                else:
                    # New section
                    current_section[key] = {}
                    section_stack.append(current_section[key])
                    indent_stack.append(indent)
                    current_section = current_section[key]

    return config


def get_gpu_config(env: Optional[str] = None, config_name: str = "order.yaml") -> Optional[str]:
    """Get GPU configuration for environment.

    Args:
        env: Environment name (dev/test). Auto-detected from argv if None.
        config_name: Config file name

    Returns:
        CUDA_VISIBLE_DEVICES string or None
    """
    if env is None:
        env = get_env_from_argv()

    if env is None:
        env = os.environ.get("COMP_HRDOC_ENV", "dev")

    config = load_yaml_config(config_name)
    gpu_config = config.get("gpu", {})

    return gpu_config.get(env)


def setup_gpu(env: Optional[str] = None, gpu_id: Optional[str] = None) -> None:
    """Setup CUDA_VISIBLE_DEVICES before importing torch.

    MUST be called before `import torch`.

    Args:
        env: Environment name (dev/test)
        gpu_id: Explicit GPU ID (overrides config)
    """
    if gpu_id is not None:
        cuda_devices = str(gpu_id)
    else:
        cuda_devices = get_gpu_config(env)

    if cuda_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        print(f"[GPU] CUDA_VISIBLE_DEVICES={cuda_devices}")


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


def get_config(env: Optional[str] = None, config_name: str = "order.yaml") -> Dict[str, Any]:
    """Get full resolved config for environment.

    Args:
        env: Environment name
        config_name: Config file name

    Returns:
        Config dict with environment-specific values resolved
    """
    if env is None:
        env = get_env_from_argv() or os.environ.get("COMP_HRDOC_ENV", "dev")

    config = load_yaml_config(config_name)
    config["_env"] = env

    # Resolve environment-specific paths
    if "data" in config and env in config["data"]:
        config["data"]["_resolved"] = config["data"][env]

    if "artifact" in config and env in config["artifact"]:
        config["artifact"]["_resolved"] = config["artifact"][env]

    return config


def print_gpu_info():
    """Print GPU information. Call after importing torch."""
    import torch
    print(f"[GPU] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[GPU] Device count: {torch.cuda.device_count()}")
        print(f"[GPU] Current device: {torch.cuda.current_device()}")
        print(f"[GPU] Device name: {torch.cuda.get_device_name(0)}")
