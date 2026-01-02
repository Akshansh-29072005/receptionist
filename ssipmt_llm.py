# ssipmt5_llm.py - Modified to use Local LLM instead of Excel QA
import os
import time
import threading
import queue
import subprocess
import traceback
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import cv2
import pygame
import sounddevice as sd
from collections import deque
from flask import Flask, jsonify, render_template_string, send_from_directory
from flask import make_response
import webbrowser
import signal
import requests
import json

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Porcupine (only "Hello Robot")
PORCUPINE_ACCESS_KEY = "2NqhVmHPn6ZyLyqPNygsMDAu2I2Vq5kslZaLZhAUbbBwM1XwJFk5DQ=="  # ← REPLACE THIS!
HELLO_ROBOT_PPN = os.path.join(BASE_DIR, "models", "hello_en.ppn")

VOICES_DIR = os.path.join(BASE_DIR, "voices")
MODELS_DIR = os.path.join(BASE_DIR, "models")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# LLM Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2"  # Change this to your preferred model (llama2, mistral, phi, etc.)
LLM_TIMEOUT = 30  # seconds

AMY_MODEL = os.path.join(VOICES_DIR, "amy", "model.onnx")
PRATHAM_MODEL = os.path.join(VOICES_DIR, "pratham", "model.onnx")
WHISPER_LOCAL_DIR = os.path.join(MODELS_DIR, "stt")

# Constants
MIC_RATE = 16000
CONVO_IDLE_TIMEOUT = 15.0  # Auto-return to IDLE after 15s silence
SIMILARITY_THRESHOLD = 0.45

# State
state_lock = threading.Lock()
state = {
    "mode": "IDLE",
    "status_ui": "Say Hello Robot",  # UI message
    "language": "en",
    "last_user_text": "",
    "last_answer_text": "",
    "last_known_person": None,
    "current_user_name": None,
    "last_activity_time": time.time(),
    "last_face_time": 0.0,
    "last_wake_time": 0.0,
}

stop_event = threading.Event()
question_queue = queue.Queue()
tts_queue = queue.Queue()
wakeword_queue = queue.Queue()
is_speaking = threading.Event()
audio_segment_queue = queue.Queue(maxsize=10)
wake_request_queue = queue.Queue()

# ==============================
# STATE CONTROLLER (AUTHORITATIVE)
# ==============================

VALID_TRANSITIONS = {
    "IDLE": {"CONVO"},
    "CONVO": {"IDLE"},
}

def transition_state(new_mode, reason=""):
    with state_lock:
        current = state["mode"]
        if new_mode == current:
            return False

        allowed = VALID_TRANSITIONS.get(current, set())
        if new_mode not in allowed:
            print(f"[STATE] ❌ Illegal transition {current} → {new_mode} ({reason})")
            return False

        print(f"[STATE] ✅ {current} → {new_mode} ({reason})")
        state["mode"] = new_mode
        state["last_activity_time"] = time.time()
        return True
    
def wake_gate_worker():
    while not stop_event.is_set():
        try:
            source, lang, message = wake_request_queue.get(timeout=0.2)

            if is_speaking.is_set():
                wake_request_queue.task_done()
                continue

            # Try to enter CONVO
            if transition_state("CONVO", f"{source} wake"):
                with state_lock:
                    state.update({
                        "language": lang,
                        "status_ui": "Ask any question",
                        "last_answer_text": message,
                        "current_user_name": None,
                        "last_wake_time": time.time()
                    })
                tts_queue.put((message, lang, False))

            wake_request_queue.task_done()

        except queue.Empty:
            continue

def convo_timeout_worker():
    while not stop_event.is_set():
        time.sleep(1)
        with state_lock:
            if state["mode"] == "CONVO":
                idle_for = time.time() - state["last_activity_time"]
                if idle_for >= CONVO_IDLE_TIMEOUT and not is_speaking.is_set():
                    print("[CONVO] Timeout → returning to IDLE")
                    state.update({
                        "mode": "IDLE",
                        "status_ui": "Say Hello Robot",
                        "current_user_name": None,
                        "last_user_text": "",
                        "last_answer_text": ""
                    })

# ==============================
# GRACEFUL SHUTDOWN
# ==============================
def signal_handler(sig, frame):
    print("\n🛑 Shutting down gracefully...")
    stop_event.set()
    time.sleep(1.5)
    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==============================
# LOCAL LLM Q&A (REPLACES EXCEL)
# ==============================

class LocalLLMDatabase:
    """
    Replacement for Excel-based QA using a local LLM via Ollama.
    """
    def __init__(self, model_name=LLM_MODEL, api_url=OLLAMA_API_URL):
        self.model_name = model_name
        self.api_url = api_url
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges for context
        
        # System prompts for different languages
        self.system_prompts = {
            "en": """You are a helpful receptionist assistant for SSIPMT (Shri Shankaracharya Institute of Professional Management & Technology) in Raipur.
Your role is to answer questions about the institute politely and concisely.
Keep responses brief (2-3 sentences max) and friendly.
If you don't know something specific about SSIPMT, politely say you don't have that information and suggest they contact the office.""",
            
            "hi": """आप SSIPMT (श्री शंकराचार्य इंस्टीट्यूट ऑफ प्रोफेशनल मैनेजमेंट एंड टेक्नोलॉजी) रायपुर के लिए एक सहायक रिसेप्शनिस्ट हैं।
आपका काम संस्थान के बारे में सवालों का विनम्र और संक्षिप्त उत्तर देना है।
जवाब संक्षिप्त (अधिकतम 2-3 वाक्य) और मित्रवत रखें।
यदि आपको SSIPMT के बारे में कुछ विशिष्ट नहीं पता है, तो विनम्रता से कहें कि आपके पास वह जानकारी नहीं है।"""
        }
        
        # Check if Ollama is running
        self._check_ollama_availability()
    
    def _check_ollama_availability(self):
        """Check if Ollama is running and model is available."""
        try:
            # Try to connect to Ollama
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                if any(self.model_name in name for name in model_names):
                    print(f"[LLM] ✅ Connected to Ollama - Model '{self.model_name}' available")
                else:
                    print(f"[LLM] ⚠️ Model '{self.model_name}' not found. Available models: {model_names}")
                    print(f"[LLM] To install, run: ollama pull {self.model_name}")
            else:
                print(f"[LLM] ⚠️ Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("[LLM] ⚠️ Cannot connect to Ollama at localhost:11434")
            print("[LLM] Please ensure Ollama is running: https://ollama.ai/")
        except Exception as e:
            print(f"[LLM] ⚠️ Error checking Ollama: {e}")
    
    def _build_prompt(self, question, lang):
        """Build prompt with system instructions and conversation history."""
        system_prompt = self.system_prompts.get(lang, self.system_prompts["en"])
        
        # Build conversation context
        context = f"{system_prompt}\n\n"
        
        # Add recent conversation history
        for q, a in self.conversation_history[-self.max_history:]:
            context += f"User: {q}\nAssistant: {a}\n\n"
        
        context += f"User: {question}\nAssistant:"
        return context
    
    def find_answer(self, question, lang):
        """
        Query the local LLM for an answer.
        Returns: (answer_text, confidence_score)
        """
        if not question or not question.strip():
            return None, 0.0
        
        try:
            # Build the prompt
            prompt = self._build_prompt(question, lang)
            
            # Prepare request payload
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 150,  # Keep responses concise
                }
            }
            
            print(f"[LLM] Querying {self.model_name} for: {question[:50]}...")
            
            # Make API request
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                if answer:
                    # Add to conversation history
                    self.conversation_history.append((question, answer))
                    
                    # Trim history if too long
                    if len(self.conversation_history) > self.max_history:
                        self.conversation_history.pop(0)
                    
                    print(f"[LLM] ✅ Got answer: {answer[:80]}...")
                    return answer, 0.9  # High confidence from LLM
                else:
                    print("[LLM] ⚠️ Empty response from LLM")
                    return None, 0.0
            else:
                print(f"[LLM] ❌ API error: {response.status_code} - {response.text}")
                return None, 0.0
                
        except requests.exceptions.Timeout:
            print(f"[LLM] ❌ Request timeout after {LLM_TIMEOUT}s")
            return None, 0.0
        except requests.exceptions.ConnectionError:
            print("[LLM] ❌ Cannot connect to Ollama. Is it running?")
            return None, 0.0
        except Exception as e:
            print(f"[LLM] ❌ Error: {e}")
            traceback.print_exc()
            return None, 0.0
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("[LLM] Conversation history cleared")


# Initialize LLM database (replaces Excel loading)
qa_db = LocalLLMDatabase()

# ==============================
# PIPER TTS
# ==============================
def generate_tts_piper(text, model_path):
    """Generate TTS audio using Piper."""
    try:
        cmd = ["piper", "--model", model_path, "--output-raw"]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        audio_bytes, stderr = proc.communicate(input=text.encode("utf-8"), timeout=15)
        
        if proc.returncode != 0:
            print(f"[TTS] Piper error: {stderr.decode()}")
            return None
        
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return None

def tts_worker():
    """TTS worker thread."""
    while not stop_event.is_set():
        try:
            text, lang, _ = tts_queue.get(timeout=0.5)
            
            is_speaking.set()
            with state_lock:
                state["last_activity_time"] = time.time()
            
            model = AMY_MODEL if lang.startswith("en") else PRATHAM_MODEL
            audio = generate_tts_piper(text, model)
            
            if audio is not None:
                sd.play(audio, samplerate=22050)
                sd.wait()
            
            is_speaking.clear()
            tts_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[TTS] Error: {e}")
            is_speaking.clear()

# ==============================
# Q&A WORKER (Modified to use LLM)
# ==============================
def qa_worker():
    """Process questions using Local LLM."""
    while not stop_event.is_set():
        try:
            question_text, lang = question_queue.get(timeout=0.5)
            
            with state_lock:
                if state["mode"] != "CONVO":
                    question_queue.task_done()
                    continue
                
                state["last_user_text"] = question_text
                state["last_activity_time"] = time.time()
            
            print(f"[QA] Processing: '{question_text}' (lang: {lang})")
            
            # Query LLM instead of Excel
            answer, confidence = qa_db.find_answer(question_text, lang)
            
            if answer and confidence > 0.3:  # Lower threshold since LLM is more flexible
                print(f"[QA] ✅ Answer found (confidence: {confidence:.2f})")
                with state_lock:
                    state["last_answer_text"] = answer
                tts_queue.put((answer, lang, False))
            else:
                fallback = (
                    "Sorry, I don't have information about that. Please contact the office."
                    if lang.startswith("en")
                    else "क्षमा करें, मेरे पास इसकी जानकारी नहीं है। कृपया कार्यालय से संपर्क करें।"
                )
                print(f"[QA] ⚠️ No answer found, using fallback")
                with state_lock:
                    state["last_answer_text"] = fallback
                tts_queue.put((fallback, lang, False))
            
            question_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[QA] Error: {e}")
            traceback.print_exc()

# ==============================
# FACE RECOGNITION
# ==============================
def face_worker():
    """Face recognition worker (unchanged)."""
    global state
    
    known_encodings = []
    known_names = []
    
    if not os.path.isdir(KNOWN_FACES_DIR):
        print("[FACE] No known_faces directory found. Skipping face recognition.")
        return
    
    try:
        import face_recognition
        
        for fname in os.listdir(KNOWN_FACES_DIR):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(KNOWN_FACES_DIR, fname)
                img = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_encodings.append(encodings[0])
                    name = os.path.splitext(fname)[0]
                    known_names.append(name)
        
        print(f"[FACE] Loaded {len(known_names)} known faces: {known_names}")
    except ImportError:
        print("[FACE] face_recognition not installed. Skipping.")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[FACE] Cannot open camera.")
        return
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb)
        face_encodings = face_recognition.face_encodings(rgb, face_locations)
        
        recognized = None
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
            if True in matches:
                idx = matches.index(True)
                recognized = known_names[idx]
                break
        
        with state_lock:
            now = time.time()
            if recognized:
                state["last_known_person"] = recognized
                state["last_face_time"] = now
                
                # Auto-wake if in IDLE
                if state["mode"] == "IDLE" and (now - state.get("last_wake_time", 0)) > 10:
                    greeting = f"Hello {recognized}, how can I help you?"
                    wake_request_queue.put(("face", "en", greeting))
            else:
                if (now - state["last_face_time"]) > 3.0:
                    state["last_known_person"] = None
        
        time.sleep(0.5)
    
    cap.release()

# ==============================
# WAKEWORD DETECTION
# ==============================
def wakeword_listener():
    """Porcupine wakeword detection (unchanged)."""
    try:
        import pvporcupine
    except ImportError:
        print("[WAKEWORD] pvporcupine not installed.")
        return
    
    if not os.path.isfile(HELLO_ROBOT_PPN):
        print(f"[WAKEWORD] Model not found: {HELLO_ROBOT_PPN}")
        return
    
    try:
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[HELLO_ROBOT_PPN]
        )
    except Exception as e:
        print(f"[WAKEWORD] Failed to initialize: {e}")
        return
    
    print(f"[WAKEWORD] Listening for 'Hello Robot' ({porcupine.sample_rate} Hz)...")
    
    def audio_callback(indata, frames, time_info, status):
        if status:
            print(f"[WAKEWORD] Audio status: {status}")
        audio_segment_queue.put(indata.copy())
    
    with sd.InputStream(
        samplerate=porcupine.sample_rate,
        channels=1,
        dtype=np.int16,
        blocksize=porcupine.frame_length,
        callback=audio_callback
    ):
        while not stop_event.is_set():
            try:
                audio_chunk = audio_segment_queue.get(timeout=0.5)
                pcm = audio_chunk.flatten()
                
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    with state_lock:
                        mode = state["mode"]
                    
                    if mode == "IDLE":
                        print("[WAKEWORD] ✅ 'Hello Robot' detected!")
                        wake_request_queue.put((
                            "wakeword",
                            "en",
                            "Hello! How can I help you today?"
                        ))
                
            except queue.Empty:
                continue
    
    porcupine.delete()

# ==============================
# HINDI WAKEWORD (Keyword spotting)
# ==============================
def hindi_wakeword_handler():
    """Simple Hindi wakeword detection (unchanged)."""
    hindi_keywords = ["नमस्ते", "हैलो", "शंकरा", "मित्र"]
    
    while not stop_event.is_set():
        try:
            text = wakeword_queue.get(timeout=0.5)
            text_lower = text.lower()
            
            if any(kw in text_lower for kw in hindi_keywords):
                with state_lock:
                    if state["mode"] == "IDLE":
                        print(f"[HINDI_WAKE] Detected: {text}")
                        wake_request_queue.put((
                            "hindi_wakeword",
                            "hi",
                            "नमस्ते! मैं आपकी कैसे मदद कर सकता हूं?"
                        ))
            
            wakeword_queue.task_done()
        except queue.Empty:
            continue

# ==============================
# STT (WHISPER)
# ==============================
def audio_stream_worker():
    """Continuous audio streaming (unchanged)."""
    buffer = deque(maxlen=int(MIC_RATE * 2))
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"[MIC] {status}")
        buffer.extend(indata[:, 0])
    
    with sd.InputStream(
        samplerate=MIC_RATE,
        channels=1,
        dtype=np.float32,
        callback=callback
    ):
        print("[MIC] Audio stream started")
        while not stop_event.is_set():
            time.sleep(0.5)

def audio_processor_worker():
    """Process audio in CONVO mode (unchanged)."""
    import whisper
    
    model_path = os.path.join(WHISPER_LOCAL_DIR, "tiny.pt")
    if not os.path.isfile(model_path):
        print(f"[STT] Model not found: {model_path}")
        return
    
    model = whisper.load_model("tiny", download_root=WHISPER_LOCAL_DIR)
    print("[STT] Whisper model loaded")
    
    last_transcription_time = 0
    buffer = deque(maxlen=int(MIC_RATE * 2))
    
    def callback(indata, frames, time_info, status):
        buffer.extend(indata[:, 0])
    
    with sd.InputStream(
        samplerate=MIC_RATE,
        channels=1,
        dtype=np.float32,
        callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1)
            
            with state_lock:
                mode = state["mode"]
                lang = state["language"]
            
            if mode != "CONVO":
                continue
            
            if is_speaking.is_set():
                continue
            
            now = time.time()
            if (now - last_transcription_time) < 2.0:
                continue
            
            if len(buffer) < MIC_RATE * 1.5:
                continue
            
            audio_np = np.array(list(buffer), dtype=np.float32)
            energy = np.sqrt(np.mean(audio_np ** 2))
            
            if energy < 0.01:
                continue
            
            last_transcription_time = now
            
            try:
                result = model.transcribe(
                    audio_np,
                    language=lang,
                    fp16=False,
                    temperature=0.0
                )
                text = result["text"].strip()
                
                if text and len(text) > 3:
                    print(f"[STT] Transcribed ({lang}): {text}")
                    
                    if mode == "IDLE":
                        wakeword_queue.put(text)
                    elif mode == "CONVO":
                        question_queue.put((text, lang))
                    
                    with state_lock:
                        state["last_activity_time"] = time.time()
                
            except Exception as e:
                print(f"[STT] Error: {e}")

# ==============================
# FLASK WEB UI
# ==============================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SSIPMT Receptionist</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
  color: #fff;
  min-height: 100vh;
}
header {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem 2rem;
  backdrop-filter: blur(12px);
  background: rgba(255, 255, 255, 0.1);
  box-shadow: 0 0 20px rgba(255,255,255,0.1);
  border-bottom: 1px solid rgba(255,255,255,0.15);
}
header img {
  height: 70px;
  width: auto;
  border-radius: 8px;
  background: rgba(255,255,255,0.15);
  padding: 6px;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  padding: 2rem;
}
.card {
  background: rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 1.5rem;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.15);
}
.label { color: #a2c4ff; font-weight: 600; }
textarea {
  width: 95%;
  height: 350px;
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 12px;
  padding: 0.8rem;
  color: #fff;
  resize: none;
  font-size: 2.5rem;
}
.videoBox {
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 0 25px rgba(0,0,0,0.3);
  position: relative;
}
video {
  width: 100%;
  display: block;
}
.prompt-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 2.2rem;
  font-weight: 700;
  text-align: center;
  text-shadow: 0 0 12px rgba(0,0,0,0.7);
  width: 100%;
  padding: 0 1rem;
  z-index: 10;
}
footer {
  text-align: center;
  color: #d1d9ff;
  padding: 1rem;
  font-size: 0.9rem;
}
.llm-badge {
  background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
  padding: 0.3rem 0.8rem;
  border-radius: 12px;
  font-size: 0.85rem;
  font-weight: 600;
  display: inline-block;
  margin-left: 1rem;
}
</style>
<script>
let currentVideoType = "listen";

window.addEventListener('DOMContentLoaded', () => {
  const vid = document.getElementById('robot_video');
  vid.src = '/video/listen';
  vid.play().catch(e => console.log("Initial play failed:", e));
});

async function update() {
  try {
    const res = await fetch('/state');
    const s = await res.json();

    // Update debug fields
    document.getElementById('status').innerText = s.mode + " / " + s.status;
    document.getElementById('mic').innerText = s.mic;
    document.getElementById('person').innerText = s.person || '-';

    // --- Handle Q&A display ---
    const userText = document.getElementById('user_text');
    const botText = document.getElementById('bot_text');

    userText.value = s.user || '';
    botText.value = s.bot || '';

    // --- Video logic ---
    let desiredVideo = "listen";
    if (s.mode === "CONVO") {
      if (s.bot && s.bot.toLowerCase().includes("sorry")) {
        desiredVideo = "sorry";
      } else {
        desiredVideo = "answer";
      }
    }

    const vid = document.getElementById('robot_video');
    if (desiredVideo !== currentVideoType) {
      currentVideoType = desiredVideo;
      vid.src = `/video/${desiredVideo}`;
    }

    // --- Prompt text ---
    document.getElementById('prompt-text').textContent = 
      s.mode === "IDLE" ? "Say Hello Robot" : "Ask any question";

  } catch (e) {
    console.error("UI update error:", e);
  }
  const vid = document.getElementById('robot_video');
    vid.addEventListener('ended', () => {
    vid.currentTime = 0;
    vid.play().catch(e => console.warn("Auto-replay failed:", e));
    });
}

setInterval(update, 500);
</script>
</head>
<body>
<header>
  <img src="/logo" alt="Logo">
  <h1>शंकरा मित्र</h1>
  <span class="llm-badge">🤖 Powered by Local LLM</span>
</header>

<div class="grid">
  <div class="card">
    <div><span class="label">Status:</span> <span id="status">—</span></div>
    <div><span class="label">Mic:</span> <span id="mic">—</span></div>
    <div><span class="label">Known person:</span> <span id="person">-</span></div>
    <div><span class="label">User says:</span></div>
    <textarea id="user_text" readonly></textarea>
    <div><span class="label">Wingman replies:</span></div>
    <textarea id="bot_text" readonly></textarea>
    <footer>Designed & developed by the students of SSIPMT, Raipur</footer>

  </div>

  <div class="card videoBox">
    <h3>Say "Hello Robot" to ask questions</h3>
    <h3>प्रश्न पूछने के लिए "नमस्ते शंकरा मित्र" कहें</h3>
    <video id="robot_video"
       autoplay muted playsinline
       style="display:block;width:100%;border-radius:20px;"
       loop>
    </video>
  </div>
</div>

<footer>Designed & developed by the students of CSE(AI) & AIML at SSIPMT, Raipur</footer>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/state")
def state_json():
    with state_lock:
        s = state.copy()
    is_currently_speaking = is_speaking.is_set()

    if s["mode"] == "IDLE":
        mic_status = "idle (waiting for 'Hello Robot')"
        status_label = "idle"
    else:
        status_label = "speaking" if is_currently_speaking else "listening"
        mic_status = "🔊 Speaking..." if is_currently_speaking else "🎙️ Listening..."

    return jsonify({
        "mode": s["mode"],
        "status": status_label,
        "mic": mic_status,
        "person": s["last_known_person"] or "-",
        "user": s["last_user_text"],
        "bot": s["last_answer_text"],
        "is_speaking": is_currently_speaking
    })

@app.route("/video/<vid_type>")
def serve_video(vid_type):
    mapping = {
        "listen": "robot_listen.mp4",
        "answer": "robot_answer.mp4",
        "sorry": "robot_sorry.mp4"
    }
    filename = mapping.get(vid_type, "robot_listen.mp4")
    full_path = os.path.join(ASSETS_DIR, filename)
    
    if not os.path.isfile(full_path):
        print(f"[ERROR] Video not found: {full_path}")
        full_path = os.path.join(ASSETS_DIR, "robot_listen.mp4")
        if not os.path.isfile(full_path):
            return "Video not available", 404
        filename = "robot_listen.mp4"

    response = make_response(send_from_directory(ASSETS_DIR, filename))
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    return response

@app.route("/logo")
def logo():
    return send_from_directory(ASSETS_DIR, "logo.jpeg")

# ==============================
# MAIN
# ==============================
def main():
    # Start workers
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=qa_worker, daemon=True).start()
    threading.Thread(target=face_worker, daemon=True).start()
    threading.Thread(target=hindi_wakeword_handler, daemon=True).start()
    threading.Thread(target=wakeword_listener, daemon=True).start()
    threading.Thread(target=audio_stream_worker, daemon=True).start()
    threading.Thread(target=audio_processor_worker, daemon=True).start()
    threading.Thread(target=wake_gate_worker, daemon=True).start()
    threading.Thread(target=convo_timeout_worker, daemon=True).start()

    # Auto-open browser
    threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open("http://127.0.0.1:8000")), daemon=True).start()

    print("\n🚀 SSIPMT OFFLINE RECEPTIONIST WITH LOCAL LLM READY!")
    print("🤖 Using Local LLM instead of Excel database")
    print("🗣️  Say 'Hello Robot' to start!")
    print("🌐 UI: http://127.0.0.1:8000\n")
    app.run(host="127.0.0.1", port=8000, debug=False)

if __name__ == "__main__":
    main()