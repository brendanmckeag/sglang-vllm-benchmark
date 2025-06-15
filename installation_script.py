import subproce6ss
import sys
import shutil

def check_system_requirements():
    """
    Function 1: Check system requirements and dependencies
    This ensures we have everything needed before installing vLLM
    """
    print("=== Checking System Requirements ===")
    
    # Check Python version
    python_version = sys.version_info
    print(f"✓ Python found: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("✗ Python 3.8+ is required")
        return False
    
    # Check if pip is available
    if shutil.which("pip") or shutil.which("pip3"):
        print("✓ pip found")
    else:
        print("✗ pip is required but not found")
        return False
    
    # Check for NVIDIA GPU (should be available in RunPod)
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True, check=True)
        print("✓ NVIDIA GPU detected:")
        print(f"  {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠ No NVIDIA GPU detected - vLLM will run on CPU (slower)")
    
    print("System requirements check complete!")
    print("")
    return True

def install_system_dependencies():
    """
    Function 2: Install system-level dependencies
    This installs required system libraries that vLLM and SGLang need
    """
    print("=== Installing System Dependencies ===")
    
    try:
        # In RunPod, we run as root so no sudo needed
        print("Installing system dependencies (running as root)...")
        
        # Install libnuma and other dependencies
        dependencies = [
            "libnuma1",      # NUMA library runtime
            "libnuma-dev",   # NUMA library development files
            "build-essential", # Basic build tools
            "python3-dev"    # Python development headers
        ]
        
        print("Updating package list...")
        subprocess.run(["apt", "update"], check=True)
        
        print("Installing dependencies:", " ".join(dependencies))
        subprocess.run(["apt", "install", "-y"] + dependencies, check=True)
        
        print("✓ System dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install system dependencies: {e}")
        print("If you're not running as root, you may need to run manually:")
        print("  apt update")
        print("  apt install -y libnuma1 libnuma-dev build-essential python3-dev")
        return False
    except FileNotFoundError:
        print("✗ apt command not found - this might not be a Debian/Ubuntu system")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def install_vllm():
    """
    Function 3: Install vLLM and its dependencies
    This installs the main vLLM package with CUDA support for GPU acceleration
    """
    print("=== Installing vLLM ===")
    
    try:
        # Upgrade pip first (important for compatibility)
        print("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✓ pip upgraded successfully")
        
        # Install vLLM with CUDA support
        print("Installing vLLM (this may take a few minutes)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "vllm",
            "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ], check=True)
        
        print("✓ vLLM installed successfully!")
        
        # Verify installation
        print("Verifying installation...")
        result = subprocess.run([sys.executable, "-c", "import vllm; print(f'vLLM version: {vllm.__version__}')"], 
                              capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        print("This might be due to CUDA compatibility issues or network problems")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during installation: {e}")
        return False

def install_sglang():
    """
    Function 4: Install SGLang (Structured Generation Language)
    SGLang provides structured prompting and generation on top of vLLM
    """
    print("=== Installing SGLang ===")
    
    try:
        # Install SGLang from PyPI
        print("Installing SGLang...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "sglang[all]"  # Install with all optional dependencies
        ], check=True)
        
        print("✓ SGLang installed successfully!")
        
        # Verify SGLang installation
        print("Verifying SGLang installation...")
        result = subprocess.run([
            sys.executable, "-c", 
            "import sglang as sgl; print(f'SGLang version: {sgl.__version__}')"
        ], capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        
        # Also check if we can import the main components
        result = subprocess.run([
            sys.executable, "-c", 
            "from sglang import function, system, user, assistant, gen, Runtime; print('✓ SGLang components imported successfully')"
        ], capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("✗ SGLang installation failed: {e}")
        print("This might be due to dependency conflicts or network issues")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during SGLang installation: {e}")
        return False

def run_full_installation():
    """
    Main installation function - runs all installation steps in sequence
    """
    print("🚀 Starting vLLM + SGLang Installation")
    print("=" * 50)
    
    # Run installation steps
    success = check_system_requirements()
    if not success:
        print("❌ Installation failed at system requirements check")
        return False
    
    success = install_system_dependencies()
    if not success:
        print("❌ Installation failed at system dependencies")
        return False
    
    success = install_vllm()
    if not success:
        print("❌ Installation failed at vLLM installation")
        return False
    
    success = install_sglang()
    if not success:
        print("❌ Installation failed at SGLang installation")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("You can now run the Hello World examples in the next cell.")
    return True

# Run the installation when this script is executed
if __name__ == "__main__":
    run_full_installation()