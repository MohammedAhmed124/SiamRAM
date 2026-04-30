import shutil
import subprocess
import sys


def _run_nvidia_smi() -> tuple[bool, str]:
    if shutil.which("nvidia-smi") is None:
        return False, "nvidia-smi not found in PATH"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        first_line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "no gpu rows returned"
        return True, first_line
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    print("python:", sys.version.split()[0])
    print("platform:", sys.platform)

    try:
        import torch
    except Exception as exc:
        print("torch: import failed")
        print("error:", exc)
        return 1

    print("torch:", torch.__version__)
    cuda_available = torch.cuda.is_available()
    print("cuda available:", cuda_available)
    print("cuda device count:", torch.cuda.device_count())

    if cuda_available:
        for i in range(torch.cuda.device_count()):
            print(f"gpu {i}:", torch.cuda.get_device_name(i))

    smi_ok, smi_info = _run_nvidia_smi()
    print("nvidia-smi:", "ok" if smi_ok else "not available")
    print("nvidia-smi info:", smi_info)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
