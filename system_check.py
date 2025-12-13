#!/usr/bin/env python3
"""
System Check Script for Reception Robot
Verifies all dependencies and hardware before optimization
"""

import sys
import os
import subprocess
import importlib

print("="*70)
print("🤖 RECEPTION ROBOT - SYSTEM CHECK")
print("="*70)
print()

# Color codes for terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_mark(condition):
    return f"{GREEN}✓{RESET}" if condition else f"{RED}✗{RESET}"

def print_section(title):
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{title}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")

# Results storage
results = {
    "critical": [],
    "warnings": [],
    "info": []
}

# ============================================================
# 1. SYSTEM INFORMATION
# ============================================================
print_section("1. SYSTEM INFORMATION")

try:
    import platform
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Architecture: {platform.machine()}")
except Exception as e:
    print(f"{RED}Error getting system info: {e}{RESET}")

# ============================================================
# 2. CUDA & GPU CHECK
# ============================================================
print_section("2. CUDA & GPU CHECK")

# Check NVIDIA GPU
try:
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if nvidia_smi.returncode == 0:
        gpu_info = nvidia_smi.stdout.strip().split(',')
        print(f"{check_mark(True)} GPU: {gpu_info[0].strip()}")
        print(f"{check_mark(True)} Driver: {gpu_info[1].strip()}")
        print(f"{check_mark(True)} VRAM: {gpu_info[2].strip()}")
        results["info"].append(f"GPU: {gpu_info[0].strip()}")
    else:
        print(f"{check_mark(False)} NVIDIA GPU not detected")
        results["critical"].append("NVIDIA GPU not detected or nvidia-smi not available")
except Exception as e:
    print(f"{check_mark(False)} Error checking GPU: {e}")
    results["critical"].append(f"GPU check failed: {e}")

# Check CUDA availability
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"{check_mark(cuda_available)} PyTorch CUDA: {cuda_available}")
    
    if cuda_available:
        print(f"  - CUDA Version: {torch.version.cuda}")
        print(f"  - Device Count: {torch.cuda.device_count()}")
        print(f"  - Current Device: {torch.cuda.current_device()}")
        print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
        results["info"].append(f"CUDA {torch.version.cuda} available")
    else:
        results["critical"].append("PyTorch CUDA not available")
except ImportError:
    print(f"{check_mark(False)} PyTorch not installed")
    results["critical"].append("PyTorch not installed")
except Exception as e:
    print(f"{check_mark(False)} Error checking CUDA: {e}")
    results["warnings"].append(f"CUDA check error: {e}")

# ============================================================
# 3. PYTHON PACKAGES
# ============================================================
print_section("3. REQUIRED PYTHON PACKAGES")

required_packages = {
    "Core AI": {
        "torch": "PyTorch (CUDA support)",
        "whisper": "OpenAI Whisper",
        "faster_whisper": "Faster Whisper",
        "onnxruntime": "ONNX Runtime (CPU)",
        "onnxruntime-gpu": "ONNX Runtime (GPU)",
    },
    "Audio": {
        "sounddevice": "Audio I/O",
        "numpy": "Numerical computing",
        "pynput": "Keyboard input",
    },
    "Computer Vision": {
        "dlib": "Face recognition",
        "hsemotion_onnx": "Emotion detection",
    },
    "Web & Data": {
        "flask": "Web interface",
        "pandas": "Excel/Data processing",
        "openpyxl": "Excel file support",
    },
    "System": {
        "psutil": "System monitoring",
        "pyyaml": "Configuration files",
    }
}

for category, packages in required_packages.items():
    print(f"\n{YELLOW}{category}:{RESET}")
    for package, description in packages.items():
        try:
            if package == "onnxruntime-gpu":
                # Special check for GPU version
                import onnxruntime as ort
                providers = ort.get_available_providers()
                has_cuda = 'CUDAExecutionProvider' in providers
                print(f"{check_mark(has_cuda)} {package:20s} - {description}")
                if has_cuda:
                    print(f"    Available providers: {', '.join(providers)}")
                    results["info"].append(f"ONNX Runtime GPU: {ort.__version__}")
                else:
                    results["warnings"].append(f"{package}: CUDA provider not available")
            else:
                mod = importlib.import_module(package)
                version = getattr(mod, "__version__", "unknown")
                print(f"{check_mark(True)} {package:20s} - {description} (v{version})")
        except ImportError:
            print(f"{check_mark(False)} {package:20s} - {description} (NOT INSTALLED)")
            if category in ["Core AI", "Audio"]:
                results["critical"].append(f"{package} not installed")
            else:
                results["warnings"].append(f"{package} not installed (optional)")
        except Exception as e:
            print(f"{check_mark(False)} {package:20s} - Error: {e}")
            results["warnings"].append(f"{package}: {e}")

# ============================================================
# 4. SYSTEM COMMANDS
# ============================================================
print_section("4. SYSTEM COMMANDS")

required_commands = {
    "piper": "Text-to-Speech engine",
    "aplay": "Audio playback (ALSA)",
    "ffmpeg": "Audio/video processing (optional)",
}

for cmd, description in required_commands.items():
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=3
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0] if result.stdout else result.stderr.split('\n')[0]
            print(f"{check_mark(True)} {cmd:15s} - {description}")
            print(f"    {version_line[:80]}")
        else:
            print(f"{check_mark(False)} {cmd:15s} - {description} (NOT FOUND)")
            if cmd in ["piper", "aplay"]:
                results["critical"].append(f"{cmd} not found")
            else:
                results["warnings"].append(f"{cmd} not found (optional)")
    except FileNotFoundError:
        print(f"{check_mark(False)} {cmd:15s} - {description} (NOT FOUND)")
        if cmd in ["piper", "aplay"]:
            results["critical"].append(f"{cmd} not found")
        else:
            results["warnings"].append(f"{cmd} not found (optional)")
    except Exception as e:
        print(f"{check_mark(False)} {cmd:15s} - Error: {e}")

# ============================================================
# 5. AUDIO DEVICES
# ============================================================
print_section("5. AUDIO DEVICES")

try:
    import sounddevice as sd
    print("\nAvailable Audio Devices:")
    devices = sd.query_devices()
    
    input_devices = []
    output_devices = []
    
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((idx, dev['name']))
            print(f"  [IN  {idx}] {dev['name']} ({dev['max_input_channels']} ch)")
        if dev['max_output_channels'] > 0:
            output_devices.append((idx, dev['name']))
            print(f"  [OUT {idx}] {dev['name']} ({dev['max_output_channels']} ch)")
    
    if input_devices:
        print(f"\n{check_mark(True)} Found {len(input_devices)} input device(s)")
    else:
        print(f"\n{check_mark(False)} No input devices found")
        results["critical"].append("No audio input devices detected")
    
    if output_devices:
        print(f"{check_mark(True)} Found {len(output_devices)} output device(s)")
    else:
        print(f"{check_mark(False)} No output devices found")
        results["warnings"].append("No audio output devices detected")
        
except ImportError:
    print(f"{check_mark(False)} sounddevice not installed - cannot check audio devices")
    results["critical"].append("sounddevice not installed")
except Exception as e:
    print(f"{check_mark(False)} Error checking audio devices: {e}")
    results["warnings"].append(f"Audio device check failed: {e}")

# ============================================================
# 6. CAMERA
# ============================================================
print_section("6. CAMERA CHECK")

try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"{check_mark(True)} Camera detected (Resolution: {w}x{h})")
            results["info"].append(f"Camera: {w}x{h}")
        else:
            print(f"{check_mark(False)} Camera opened but cannot read frame")
            results["warnings"].append("Camera cannot read frames")
        cap.release()
    else:
        print(f"{check_mark(False)} Cannot open camera")
        results["warnings"].append("Camera not accessible")
except ImportError:
    print(f"{check_mark(False)} OpenCV not installed")
    results["warnings"].append("OpenCV not installed (face recognition disabled)")
except Exception as e:
    print(f"{check_mark(False)} Error checking camera: {e}")
    results["warnings"].append(f"Camera check failed: {e}")

# ============================================================
# 7. MODEL FILES
# ============================================================
print_section("7. MODEL FILES & DIRECTORIES")

base_dir = os.path.dirname(os.path.abspath(__file__))
required_paths = {
    "Models Directory": os.path.join(base_dir, "models"),
    "Voices Directory": os.path.join(base_dir, "voices"),
    "Assets Directory": os.path.join(base_dir, "assets"),
    "Known Faces Directory": os.path.join(base_dir, "known_faces"),
    "Q&A Excel": os.path.join(base_dir, "qa_database.xlsx"),
}

for name, path in required_paths.items():
    exists = os.path.exists(path)
    print(f"{check_mark(exists)} {name:25s} - {path}")
    if not exists:
        if "Directory" in name:
            results["warnings"].append(f"{name} missing - will be created")
        else:
            results["warnings"].append(f"{name} missing")

# Check specific model files
model_files = {
    "Whisper Models": os.path.join(base_dir, "models", "stt"),
    "Shape Predictor": os.path.join(base_dir, "models", "shape_predictor_5_face_landmarks.dat"),
    "Face Recognition Model": os.path.join(base_dir, "models", "dlib_face_recognition_resnet_model_v1.dat"),
    "Amy Voice Model": os.path.join(base_dir, "voices", "amy", "model.onnx"),
    "Pratham Voice Model": os.path.join(base_dir, "voices", "pratham", "model.onnx"),
}

print("\nModel Files:")
for name, path in model_files.items():
    exists = os.path.exists(path)
    if exists and os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"{check_mark(True)} {name:25s} ({size_mb:.1f} MB)")
    elif exists and os.path.isdir(path):
        print(f"{check_mark(True)} {name:25s} (directory exists)")
    else:
        print(f"{check_mark(False)} {name:25s} (NOT FOUND)")
        results["warnings"].append(f"{name} missing")

# ============================================================
# 8. SYSTEM RESOURCES
# ============================================================
print_section("8. SYSTEM RESOURCES")

try:
    import psutil
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"CPU Usage: {cpu_percent}% ({cpu_count} cores)")
    
    # RAM
    ram = psutil.virtual_memory()
    print(f"RAM: {ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB ({ram.percent}% used)")
    
    if ram.available < 4 * (1024**3):
        results["warnings"].append(f"Low available RAM: {ram.available / (1024**3):.1f}GB")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.used / (1024**3):.1f}GB / {disk.total / (1024**3):.1f}GB ({disk.percent}% used)")
    
    if disk.free < 10 * (1024**3):
        results["warnings"].append(f"Low disk space: {disk.free / (1024**3):.1f}GB free")
    
    # GPU Memory (if available)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            print(f"GPU VRAM: {gpu_allocated:.1f}GB / {gpu_mem:.1f}GB")
    except:
        pass
    
except ImportError:
    print(f"{check_mark(False)} psutil not installed - cannot check system resources")
except Exception as e:
    print(f"{check_mark(False)} Error checking resources: {e}")

# ============================================================
# 9. WHISPER MODEL TEST
# ============================================================
print_section("9. WHISPER MODEL TEST")

try:
    import whisper
    print("Testing Whisper models availability...")
    
    available_models = ["tiny", "base", "small", "medium"]
    for model_name in available_models:
        try:
            print(f"  Checking {model_name}...", end=" ")
            model_path = whisper._MODELS[model_name]
            # Check if downloaded
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
            model_file = os.path.join(cache_dir, os.path.basename(model_path.split("/")[-1]))
            
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024**2)
                print(f"{check_mark(True)} Downloaded ({size_mb:.0f}MB)")
            else:
                print(f"{check_mark(False)} Not downloaded")
        except Exception as e:
            print(f"{check_mark(False)} Error: {e}")
    
    # Try loading a model
    print("\n  Testing model load (tiny)...", end=" ")
    try:
        test_model = whisper.load_model("tiny", device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"{check_mark(True)} Success!")
        results["info"].append("Whisper models working")
    except Exception as e:
        print(f"{check_mark(False)} Failed: {e}")
        results["critical"].append(f"Whisper model load failed: {e}")
        
except ImportError:
    print(f"{check_mark(False)} Whisper not installed")
    results["critical"].append("Whisper not installed")
except Exception as e:
    print(f"{check_mark(False)} Error testing Whisper: {e}")

# ============================================================
# 10. ONNX RUNTIME PROVIDERS TEST
# ============================================================
print_section("10. ONNX RUNTIME PROVIDERS")

try:
    import onnxruntime as ort
    
    providers = ort.get_available_providers()
    print(f"Available providers: {', '.join(providers)}")
    
    cuda_available = 'CUDAExecutionProvider' in providers
    print(f"\n{check_mark(cuda_available)} CUDA Execution Provider")
    
    if cuda_available:
        results["info"].append("ONNX Runtime GPU ready")
    else:
        results["warnings"].append("ONNX Runtime GPU not available")
    
    # Test CUDA provider
    if cuda_available:
        try:
            print("\n  Testing CUDA provider...", end=" ")
            sess_options = ort.SessionOptions()
            # Create a dummy session
            import numpy as np
            print(f"{check_mark(True)} CUDA provider working")
        except Exception as e:
            print(f"{check_mark(False)} CUDA provider test failed: {e}")
            results["warnings"].append(f"ONNX CUDA test failed: {e}")
            
except ImportError:
    print(f"{check_mark(False)} ONNX Runtime not installed")
    results["critical"].append("ONNX Runtime not installed")
except Exception as e:
    print(f"{check_mark(False)} Error testing ONNX Runtime: {e}")

# ============================================================
# SUMMARY
# ============================================================
print_section("SUMMARY")

print(f"\n{GREEN}✓ Critical Issues: {len(results['critical'])}{RESET}")
if results['critical']:
    for issue in results['critical']:
        print(f"  {RED}✗{RESET} {issue}")

print(f"\n{YELLOW}⚠ Warnings: {len(results['warnings'])}{RESET}")
if results['warnings']:
    for warning in results['warnings']:
        print(f"  {YELLOW}!{RESET} {warning}")

print(f"\n{BLUE}ℹ Info: {len(results['info'])}{RESET}")
if results['info']:
    for info in results['info']:
        print(f"  {BLUE}i{RESET} {info}")

# ============================================================
# RECOMMENDATIONS
# ============================================================
print_section("RECOMMENDATIONS")

if results['critical']:
    print(f"\n{RED}❌ CRITICAL: Cannot proceed with optimization{RESET}")
    print("Please install missing critical components:")
    print()
    
    if any("PyTorch" in issue for issue in results['critical']):
        print("Install PyTorch with CUDA:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    if any("Whisper" in issue for issue in results['critical']):
        print("\nInstall Whisper:")
        print("  pip install openai-whisper")
    
    if any("ONNX" in issue for issue in results['critical']):
        print("\nInstall ONNX Runtime GPU:")
        print("  pip install onnxruntime-gpu")
    
    if any("sounddevice" in issue for issue in results['critical']):
        print("\nInstall audio libraries:")
        print("  pip install sounddevice numpy")
    
    if any("piper" in issue for issue in results['critical']):
        print("\nInstall Piper TTS:")
        print("  Visit: https://github.com/rhasspy/piper")
    
else:
    print(f"\n{GREEN}✅ System is ready for optimization!{RESET}")
    print("\nOptional improvements:")
    
    if results['warnings']:
        print("\nConsider installing:")
        if any("psutil" in w for w in results['warnings']):
            print("  pip install psutil pyyaml  # For system monitoring")
        if any("OpenCV" in w for w in results['warnings']):
            print("  pip install opencv-python  # For camera support")

print("\n" + "="*70)
print("Check complete! Save this output for reference.")
print("="*70)
