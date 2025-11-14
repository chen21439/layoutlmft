import sys, os, ssl, subprocess, textwrap, urllib.request, socket
from datetime import datetime

def print_kv(k, v):
    print(f"{k:<28}: {v}")

def safe_run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, shell=True)
        return out.strip()
    except subprocess.CalledProcessError as e:
        return f"[ERR] {e}\n{e.output}"

def try_import_ver(mod):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "unknown")
        print_kv(f"pkg {mod}", ver)
    except Exception as e:
        print_kv(f"pkg {mod}", f"IMPORT ERROR -> {repr(e)}")

def https_probe(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return f"OK (status={r.status}, len={r.length})"
    except Exception as e:
        return f"ERR -> {repr(e)}"

def main():
    print("="*80)
    print("pip / TLS diagnostic")
    print("="*80)

    print_kv("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print_kv("Platform", sys.platform)
    print_kv("Python", sys.version.split()[0])
    print_kv("Executable", sys.executable)

    # pip version & path
    try:
        import pip
        print_kv("pip version", pip.__version__)
    except Exception as e:
        print_kv("pip version", f"ERR -> {repr(e)}")

    # OpenSSL info
    print_kv("OpenSSL", getattr(ssl, "OPENSSL_VERSION", "unknown"))

    # certifi CA bundle (if available)
    ca_path = None
    try:
        import certifi
        ca_path = certifi.where()
        print_kv("certifi.where()", ca_path)
        print_kv("CA file exists", os.path.exists(ca_path))
    except Exception as e:
        print_kv("certifi.where()", f"ERR -> {repr(e)}")

    # env vars
    for key in ["PIP_CERT", "REQUESTS_CA_BUNDLE", "HTTPS_PROXY", "HTTP_PROXY", "CONDA_PREFIX", "CONDA_DEFAULT_ENV"]:
        print_kv(f"env:{key}", os.environ.get(key))

    # pip config (global/user/site)
    print("\n[pip config list -v]")
    print(safe_run_cmd("python -m pip config list -v"))

    # Network probes
    print("\n[HTTPS probe]")
    print_kv("https://pypi.org", https_probe("https://pypi.org/simple/"))
    print_kv("https://files.pythonhosted.org", https_probe("https://files.pythonhosted.org/"))
    print_kv("DNS pypi.org", safe_run_cmd("nslookup pypi.org"))
    print_kv("DNS files.pythonhosted.org", safe_run_cmd("nslookup files.pythonhosted.org"))

    # Optional: show active proxies
    print("\n[urllib proxy handlers]")
    proxy = urllib.request.getproxies()
    for k, v in proxy.items():
        print_kv(f"proxy {k}", v if v else None)

    # Optional imports
    print("\n[package versions]")
    for mod in ["datasets", "transformers", "seqeval"]:
        try_import_ver(mod)

    print("\nDone.")
    print("="*80)

if __name__ == "__main__":
    main()
