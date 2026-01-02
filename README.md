# शंकरा मित्र — Offline Receptionist Robot 🤖✨

**A lightweight offline receptionist assistant for college labs and lobbies.** It listens for wake words in English and Hindi, answers visitor questions using an offline QA database, speaks responses via local TTS, and can optionally recognize known faces.

---

## 🚀 Project Overview

- Name: **शंकरा मित्र (Shankara Mitra)**
- Purpose: An offline, privacy-first receptionist that runs locally, answers frequently asked questions, greets visitors, and optionally recognizes known people.
- Languages: English (en) and Hindi (hi)
- Web UI: Runs a simple Flask-based UI at http://127.0.0.1:8000

---

## ✅ Features

- 🔊 Wakeword detection: "Hello Robot" (English) and fuzzy matching for the Hindi phrase `नमस्ते शंकरा मित्र`.
- 🗣️ Speech-to-text: Local Whisper (faster-whisper) for robust on-device ASR.
- 🗨️ Q&A: Local Excel-backed Q&A database (`qa_database.xlsx`) for fast, customizable responses.
- 🧾 Text normalization and fuzzy matching (Jaccard + sequence similarity) for tolerant matching of questions.
- 🗣️ TTS: Piper-based local TTS (Amy/Pratham voices) played via pygame.
- 📷 Optional face recognition using dlib (CPU) and images in `known_faces/`.
- 🌐 UI: Minimal dashboard showing current state, last user input and reply, and looping video clips for visual feedback.

---

## 📁 Repository layout (important files/folders)

- `ssipmt5.py`, `ssipmt_llm.py`, `reception_windows.py` — main service files (Linux vs Windows variants).
- `qa_database.xlsx` — Excel file storing questions / answers.
- `models/` — STT models (Whisper), Porcupine wakeword model, dlib models, etc.
- `voices/` — TTS voice models (e.g., `amy`, `pratham`).
- `known_faces/` — subfolders named after persons containing reference images.
- `assets/` — static UI videos and logo.
- `requirements.txt`, `environment.yml`, `setup.sh` — install / environment helpers.

---

## 🔧 Prerequisites

- Linux (recommended) or Windows (use `reception_windows.py` on Windows).
- Python 3.10+ (use `environment.yml` or `requirements.txt` to set up environment).
- A working microphone and speakers.
- Optional hardware acceleration for Whisper (CUDA) for best performance but CPU works too.

Suggested install methods:

Linux (Conda):

```bash
conda env create -f environment.yml -n receptionist
conda activate receptionist
```

or pip:

```bash
python -m pip install -r requirements.txt
```

You may also run the project helper script:

```bash
./setup.sh
```

Note: On Windows, Piper may be distributed as an exe; set `PIPER_PATH` in `reception_windows.py`.

---

## ⚙️ Configuration & Important Notes

- Porcupine: Replace `PORCUPINE_ACCESS_KEY` in the main file if needed and ensure `models/hello_en.ppn` exists.
- TTS (Piper): On Linux the code calls `piper` binary — make sure it is installed and on PATH. On Windows set `PIPER_PATH` to your `piper.exe` location.
- Whisper models: Put local Whisper model files into `models/stt/` or let the code use the bundled model names (it falls back to a remote model).
- Dlib face recognition: Place dlib models in `models/` (shape predictor and face recognition resnet). If not present, face recognition is disabled gracefully.
- QA: Edit `qa_database.xlsx` to add or update question-answer pairs. English and Hindi columns are supported (questions/answers and normalized forms are used).
- Known faces: create a folder under `known_faces/` with the person's name and place one or more face images inside.

---

## ▶️ Running the Receptionist

Linux (recommended):

```bash
python ssipmt5.py
```

Windows (use Windows-specific script / config):

```powershell
python reception_windows.py
```

The web UI will open or can be accessed at: http://127.0.0.1:8000

---

## 🧪 Testing & Usage Tips

- Say **"Hello Robot"** to wake in English. For Hindi say something close to **"नमस्ते शंकरा मित्र"** — the code uses a fuzzy match so small transcription errors are tolerated.
- Ask short, focused questions (2+ words) or include punctuation like `?` for best QA matching.
- For debugging microphone issues, check OS-level microphone permissions and try simple `arecord` / `pactl` commands.

---

## 🛠️ Troubleshooting

- PyAudio / sounddevice errors: Ensure system dependencies are installed (ALSA/PulseAudio libs on Linux).
- Piper failures: Confirm Piper is installed and accessible from the command line (or set `PIPER_PATH`).
- Whisper CUDA errors: Install compatible CUDA and torch versions — otherwise the code falls back to CPU.
- Dlib not found: The app will print `[FACE] dlib not available:` and continue without face recognition.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or PRs. Good beginner contributions:

- Improve the QA dataset (`qa_database.xlsx`).
- Add new UI improvements to `HTML_PAGE` in the Flask app.
- Add unit tests for the text normalization or QA matching functions.

---

## 📜 License & Credits

This project is created by the students of SSIPMT, Raipur (CSE AI & AIML). Check `LICENSE` in the repo (if present) or add a license of your choice.

Acknowledgements:
- Porcupine (wakeword), faster-whisper, Piper, dlib, Flask, pygame, and the many OSS projects that make offline assistants possible.

---

## 💬 Questions / Next Steps

If you'd like, I can:
- Add a CONTRIBUTING.md or PR template
- Add code comments or improve in-repo documentation
- Add a short quickstart script that auto-installs models

Have suggestions or edits? Open an issue or tell me what to change and I’ll update the README. 🙌
