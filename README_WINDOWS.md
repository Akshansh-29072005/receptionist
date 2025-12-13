Windows Quickstart — Receptionist Robot (Native Windows)

This guide walks through running the `receptionist` project natively on Windows (recommended for real-time mic/camera access).

Prerequisites
- Windows 10 or 11 (11 recommended for WSLg if you ever switch)
- Admin access to install tools

1) Install tools
- Install Miniconda (recommended): https://docs.conda.io/en/latest/miniconda.html
- Install Git for Windows: https://git-scm.com/download/win
- Install Microsoft Build Tools: https://visualstudio.microsoft.com/downloads/ → "Build Tools for Visual Studio" and select "C++ build tools" workload.
- (Optional) Install Docker Desktop if you prefer containerization

2) Clone or copy the project
Open PowerShell (or Git Bash) and run:

```powershell
# Replace <your-path> with desired folder
cd C:\path\to\projects
git clone <your-repo-url> receptionist
cd receptionist
```

If you copied the project manually, ensure the folder contains:
- `SSIPMT.py` (or `ssipmt4.py` / `ssipmt3.py` depending on which you use)
- `models/`, `voices/`, `known_faces/`, `assets/`, `qa_database.xlsx`

3) Create conda environment (recommended)

```powershell
# From project root
conda env create -f environment.yml -n receptionist
conda activate receptionist
```

Notes:
- The `environment.yml` includes many packages from `conda-forge`. If some packages (like `dlib`) are unavailable, install them manually:

```powershell
conda install -c conda-forge dlib
```

4) Install remaining pip packages

If you prefer pip inside the conda env, run:

```powershell
pip install -r requirements.txt
```

Special notes for audio / pyaudio / portaudio:
- `pyaudio` may not install via pip on Windows easily. Use `pipwin` to install the appropriate wheel:

```powershell
pip install pipwin
pipwin install pyaudio
```

- `sounddevice` requires a working PortAudio. Using conda-forge to install `portaudio` usually fixes this:

```powershell
conda install -c conda-forge portaudio
```

5) Models & Assets
- Ensure the following folders exist and contain the proper files:
  - `models/` (contains whisper/onnx/dlib models)
  - `voices/` (Piper ONNX voice models)
  - `known_faces/` (your face images)
  - `assets/` (logo, videos)
  - `qa_database.xlsx`

Copy these from your Ubuntu laptop (via Git, `scp`, or zip+transfer) if needed.

6) Optional: GPU (NVIDIA) setup
- If you have an NVIDIA GPU and want GPU acceleration for Whisper, install matching CUDA toolkit and drivers.
- This is advanced: make sure `faster-whisper`/`ctranslate2` versions match the CUDA toolkit.

7) Run the app

```powershell
conda activate receptionist
python SSIPMT.py
```

8) Troubleshooting
- If `dlib` import fails: try `conda install -c conda-forge dlib`.
- If microphone doesn't work: check Windows privacy settings → Microphone access for apps.
- If camera doesn't work: check Windows Camera privacy settings and choose the correct camera index in `SSIPMT.py` (variable `CAMERA_INDEX`).
- If `pyaudio` installation fails: use `pipwin` or download a matching wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/ and `pip install <wheel-file>`.
- If `faster-whisper` or `ctranslate2` raise errors: fall back to CPU mode. The project already includes a CPU fallback for Whisper.

9) Next steps I can do for you
- Create a PowerShell setup script to automate the above steps.
- Create a Windows-specific quick-run `.ps1` file to activate env and launch the app.
- Help transfer models/assets from your Ubuntu laptop to the Windows machine.

If you want, I can now create an automated `setup_windows.ps1` and `run_windows.ps1` to streamline these steps — shall I add them to the repo?