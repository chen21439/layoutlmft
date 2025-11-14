# cuda_env_report.py
import os
import sys
import platform
import subprocess

def try_import(name):
    try:
        mod = __import__(name)
        return mod, None
    except Exception as e:
        return None, e

def run(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True, timeout=10)
        return out.strip()
    except Exception as e:
        return f"[!] Failed to run `{cmd}`: {e}"

def print_kv(k, v):
    print(f"{k:<32}: {v}")

def main():
    print("="*80)
    print("CUDA / PyTorch Environment Report")
    print("="*80)
    print_kv("Python", sys.version.split()[0])
    print_kv("Platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    print_kv("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    print("-"*80)

    torch, torch_err = try_import("torch")
    if torch is None:
        print("[!] torch not importable:", torch_err)
        print("    -> This looks like a CPU-only env or PyTorch not installed.")
        # 仍尝试打印 nvidia-smi
        print("-"*80)
        print("[nvidia-smi]")
        print(run("nvidia-smi"))
        return

    # Torch basic
    tv = getattr(torch, "__version__", "unknown")
    print_kv("torch.__version__", tv)

    # CUDA baked version in the wheel (None if CPU/ROCm builds)
    torch_cuda_version = getattr(torch.version, "cuda", None)
    torch_hip_version = getattr(torch.version, "hip", None) or getattr(torch.version, "rocm", None)

    if torch_cuda_version:
        print_kv("torch.version.cuda (wheel)", torch_cuda_version)
    if torch_hip_version:
        print_kv("torch.version.hip/rocm (wheel)", torch_hip_version)

    print_kv("torch.cuda.is_available()", torch.cuda.is_available())

    # torchvision / torchaudio (optional)
    for pkg in ("torchvision", "torchaudio"):
        mod, err = try_import(pkg)
        if mod:
            print_kv(f"{pkg}.__version__", getattr(mod, "__version__", "unknown"))
        else:
            print_kv(f"{pkg}", f"not installed ({err.__class__.__name__})")

    print("-"*80)

    # If CUDA available via PyTorch, enumerate devices
    if torch.cuda.is_available():
        try:
            n = torch.cuda.device_count()
            print_kv("CUDA device count", n)
            print_kv("Current device", torch.cuda.current_device())
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                cc = f"{props.major}.{props.minor}"
                mem_gb = f"{props.total_memory / (1024**3):.2f} GB"
                print(f"[GPU {i}] {props.name}")
                print_kv("  Compute Capability", cc)
                print_kv("  Total Memory", mem_gb)
                print_kv("  MultiProcessorCount", props.multi_processor_count)
        except Exception as e:
            print(f"[!] Failed to query torch.cuda devices: {e}")

    # Try to get driver & runtime via nvidia-smi (works even if torch built for CPU)
    print("-"*80)
    print("[nvidia-smi]")
    smi = run("nvidia-smi")
    print(smi)

    # Common concise query for driver & CUDA version via nvidia-smi (if supported)
    if "NVIDIA-SMI" in smi:
        q = run('nvidia-smi --query-gpu=driver_version,cuda_version,name --format=csv,noheader')
        if not q.startswith("[!]"):
            print("-"*80)
            print("[nvidia-smi --query-gpu (driver,cuda,name)]")
            # possibly multiple lines if multiple GPUs
            for line in q.splitlines():
                drv, cuda_v, name = [x.strip() for x in line.split(",")]
                print_kv("Driver / CUDA / GPU", f"{drv} / {cuda_v} / {name}")

    # NVCC version if CUDA Toolkit installed (optional)
    print("-"*80)
    print("[nvcc --version]")
    print(run("nvcc --version"))

    print("-"*80)
    print("Hints:")
    if torch_cuda_version:
        print(f"- Your PyTorch wheel is built with CUDA {torch_cuda_version}.")
        print("- NVIDIA driver must be >= the minimum required by that CUDA version for GPU to work.")
    elif torch_hip_version:
        print(f"- Your PyTorch is a ROCm/HIP build ({torch_hip_version}). NVIDIA CUDA info may not apply.")
    else:
        print("- Your PyTorch looks like a CPU-only build or a non-CUDA backend. Install cu118/cu121 wheel if you need CUDA.")

if __name__ == "__main__":
    main()
