# ssipmt2.py
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "8"

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Porcupine (only "Hello Robot")
PORCUPINE_ACCESS_KEY = "2NqhVmHPn6ZyLyqPNygsMDAu2I2Vq5kslZaLZhAUbbBwM1XwJFk5DQ=="  # ← REPLACE THIS!
HELLO_ROBOT_PPN = os.path.join(BASE_DIR, "models", "hello_en.ppn")

PIPER_PATH = r"C:\piper\piper.exe"

VOICES_DIR = os.path.join(BASE_DIR, "voices")
MODELS_DIR = os.path.join(BASE_DIR, "models")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
QA_EXCEL_PATH = os.path.join(BASE_DIR, "qa_database.xlsx")

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
# Q&A DATABASE
# ==============================
import re
from difflib import SequenceMatcher

class QADatabase:
    def __init__(self, excel_path):
        self.qa_data = []
        if os.path.isfile(excel_path):
            try:
                df = pd.read_excel(excel_path)
                for _, row in df.iterrows():
                    entry = {
                        "question_en": str(row.get("question_en", "")).strip(),
                        "answer_en": str(row.get("answer_en", "")).strip(),
                        "question_hi": str(row.get("question_hi", "")).strip(),
                        "answer_hi": str(row.get("answer_hi", "")).strip(),
                        "keywords": []
                    }
                    if pd.notna(row.get("keywords")):
                        keywords = str(row.get("keywords")).lower().split(",")
                        entry["keywords"] = [self._normalize(k) for k in keywords if k.strip()]
                    # Store normalized versions for faster matching
                    entry["norm_q_en"] = self._normalize(entry["question_en"])
                    entry["norm_q_hi"] = self._normalize(entry["question_hi"])
                    if entry["question_en"] or entry["question_hi"]:
                        self.qa_data.append(entry)
                print(f"[QA] Loaded {len(self.qa_data)} Q&A entries.")
            except Exception as e:
                print(f"[QA] Excel error: {e}")

    def _normalize(self, text):
        """Clean and standardize text for matching."""
        if not text:
            return ""
        text = text.lower()
        # Remove punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove common filler words (English)
        fillers = ["uh", "um", "er", "oh", "please", "thanks", "thank you", "could you", "can you", "tell me"]
        for filler in fillers:
            text = text.replace(filler, "")
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _jaccard_similarity(self, s1, s2):
        """Token-based similarity (good for word overlap)."""
        if not s1 or not s2:
            return 0.0
        set1 = set(s1.split())
        set2 = set(s2.split())
        if not set1 and not set2:
            return 1.0
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / len(union) if union else 0.0

    def _sequence_similarity(self, s1, s2):
        """Character-level similarity (good for typos)."""
        return SequenceMatcher(None, s1, s2).ratio()

    def _keyword_score(self, query_norm, keywords):
        if not keywords:
            return 0.0
        query_tokens = set(query_norm.split())
        matches = sum(1 for kw in keywords if kw in query_tokens or any(kw in qt for qt in query_tokens))
        return min(1.0, matches / len(keywords))  # Cap at 1.0

    def find_answer(self, question, lang):
        if not question or not self.qa_data:
            return None, 0.0

        query_norm = self._normalize(question)
        best_match = None
        best_score = 0.0

        for entry in self.qa_data:
            score = 0.0

            # Handle English
            if lang.startswith("en") and entry["norm_q_en"]:
                jaccard = self._jaccard_similarity(query_norm, entry["norm_q_en"])
                seq_sim = self._sequence_similarity(query_norm, entry["norm_q_en"])
                kw_score = self._keyword_score(query_norm, entry["keywords"])
                # Weighted combination: Jaccard (word overlap) is most important
                score = jaccard * 0.5 + seq_sim * 0.3 + kw_score * 0.2

            # Handle Hindi
            elif lang.startswith("hi") and entry["norm_q_hi"]:
                jaccard = self._jaccard_similarity(query_norm, entry["norm_q_hi"])
                seq_sim = self._sequence_similarity(query_norm, entry["norm_q_hi"])
                kw_score = self._keyword_score(query_norm, entry["keywords"])
                score = jaccard * 0.5 + seq_sim * 0.3 + kw_score * 0.2

            if score > best_score:
                best_score = score
                best_match = entry

        # Raise threshold slightly for better precision
        if best_match and best_score >= 0.4:  # was 0.45 — now more forgiving with better metrics
            ans = best_match["answer_hi"] if lang.startswith("hi") and best_match["answer_hi"] else best_match["answer_en"]
            return ans, best_score

        return None, best_score

qa_db = QADatabase(QA_EXCEL_PATH)

# ==============================
# WHISPER (LOCAL CUDA)
# ==============================
from faster_whisper import WhisperModel
print("[STT] Loading Whisper...")
whisper_model = WhisperModel(
    WHISPER_LOCAL_DIR if os.path.isdir(WHISPER_LOCAL_DIR) and os.listdir(WHISPER_LOCAL_DIR) else "small",
    device="cuda",
    compute_type="float16"
)
print("[STT] Whisper ready.")

# ==============================
# DLIB FACE RECOGNITION (CPU)
# ==============================
known_faces = []
face_detector = sp = facerec = None
HAS_DLIB = False

try:
    import dlib
    SHAPE_PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_5_face_landmarks.dat")
    FACE_RECOG_MODEL_PATH = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")
    if os.path.isfile(SHAPE_PREDICTOR_PATH) and os.path.isfile(FACE_RECOG_MODEL_PATH):
        face_detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        facerec = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)
        HAS_DLIB = True

        # Load known faces
        if os.path.isdir(KNOWN_FACES_DIR):
            for name in os.listdir(KNOWN_FACES_DIR):
                person_dir = os.path.join(KNOWN_FACES_DIR, name)
                if not os.path.isdir(person_dir): continue
                for img_file in os.listdir(person_dir):
                    try:
                        img = dlib.load_rgb_image(os.path.join(person_dir, img_file))
                        dets = face_detector(img, 1)
                        if dets:
                            rect = max(dets, key=lambda r: r.area())
                            shape = sp(img, rect)
                            desc = np.array(facerec.compute_face_descriptor(img, shape))
                            known_faces.append((name, desc))
                            print(f"[FACE] Loaded: {name}")
                    except Exception as e:
                        print(f"[FACE] Error loading {img_file}: {e}")
        print(f"[FACE] Loaded {len(known_faces)} known faces.")
except Exception as e:
    print("[FACE] dlib not available:", e)

# ==============================
# UTILS
# ==============================
def is_thank_you(text):
    t = text.lower()
    return any(w in t for w in ["thank", "thanks", "धन्यवाद", "शुक्रिया"])

def is_garbage_text(text):
    t = text.lower().strip()
    if len(t) < 2: return True
    garbage = ["subscribe", "watching", "[music]", "i don't have"]
    return any(g in t for g in garbage)

def is_valid_question(text):
    t = text.lower()
    if len(t.split()) < 2: return False
    return any(w in t for w in ["what", "where", "how", "tell", "help", "about", "क्या", "कहाँ"]) or "?" in text

# --- HINDI WAKEWORD DETECTOR (must be defined before audio_stream_worker) ---
import re
from difflib import SequenceMatcher

def is_hindi_wakeword(text):
    """Fuzzy match for 'शंकरा मित्र' even with Whisper errors."""
    if not text:
        return False
    clean = re.sub(r'[^\w\s\u0900-\u097F]', ' ', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    target = "नमस्ते शंकरा मित्र"
    ratio = SequenceMatcher(None, clean, target).ratio()
    print(f"[HINDI WAKE] Similarity: {ratio:.2f} (text='{clean}')")
    return ratio >= 0.5

from difflib import SequenceMatcher

def hindi_wakeword_handler():
    """Handle Hindi wakeword events safely outside audio callback."""
    while not stop_event.is_set():
        if is_speaking.is_set():
            # time.sleep(0.01)
            continue
        try:
            lang = wakeword_queue.get(timeout=0.2)
            if lang == "hindi":
                now = time.time()
                wake_request_queue.put((
                    "hindi",
                    "hi",
                    "नमस्ते! मैं आपकी क्या मदद कर सकता हूँ?"
                ))
                print("🗣️ Hindi greeting started.")
            wakeword_queue.task_done()
        except queue.Empty:
            continue

def is_english_text(text):
    if not text:
        return False
    words = text.split()
    if not words:
        return False
    # Count words with only ASCII characters
    ascii_words = 0
    for word in words:
        if all(ord(c) < 128 for c in word):
            ascii_words += 1
    return ascii_words / len(words) > 0.5

# ==============================
# TTS (PIPER + PYGAME)
# ==============================
def tts_speak(text, lang, clear_after=False):
    if not text.strip(): return
    is_speaking.set()
    model = PRATHAM_MODEL if lang.startswith("hi") else AMY_MODEL
    out_wav = os.path.abspath(os.path.join(BASE_DIR, "tmp_tts.wav"))
    try:
        subprocess.run(
            [
                PIPER_PATH,
                "--model", model,
                "--output_file", out_wav
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW
        )

        pygame.mixer.init(frequency=22050)
        pygame.mixer.music.load(out_wav)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not stop_event.is_set():
            time.sleep(0.05)
        pygame.mixer.quit()

        with state_lock:
            state["last_activity_time"] = time.time()


        # ✅ ONLY clear if it's an answer (not greeting)
        if clear_after:
            with state_lock:
                state.update({
                    "mode": "IDLE",
                    "status_ui": "Say Hello Robot",
                    "current_user_name": None,
                    "last_user_text": "",
                    "last_answer_text": ""
                })

    except Exception as e:
        print("[TTS] ERROR:", e)
        if clear_after:
            with state_lock:
                state.update({
                    "mode": "IDLE",
                    "status_ui": "Say Hello Robot",
                    "last_user_text": "",
                    "last_answer_text": ""
                })
    finally:
        if os.path.exists(out_wav):
            os.remove(out_wav)
        is_speaking.clear()

def tts_worker():
    while not stop_event.is_set():
        try:
            item = tts_queue.get(timeout=0.2)
            if len(item) == 3:
                text, lang, clear_after = item
            else:
                text, lang = item
                clear_after = False  # default: don't clear (for greetings)
            tts_speak(text, lang, clear_after)
            tts_queue.task_done()
        except:
            continue

# ==============================
# WAKEWORD: "Hello Robot"
# ==============================
def wakeword_listener():
    try:
        import pvporcupine, pyaudio
        porcupine = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[HELLO_ROBOT_PPN],
            sensitivities=[0.8]
        )
        pa = pyaudio.PyAudio()
        stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        print("[WAKE] Listening for 'Hello Robot'...")
        while not stop_event.is_set():
            with state_lock:
                if state["mode"] != "IDLE":
                    # time.sleep(0.01)
                    continue

            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = np.frombuffer(pcm, dtype=np.int16)
            # 🔊 Software gain (safe range)
            pcm = np.clip(pcm * 1.8, -32768, 32767).astype(np.int16)
            if porcupine.process(pcm) >= 0:
                    wake_request_queue.put((
                        "english",
                        "en",
                        "Hello! How can I help you today?"
                    ))

        stream.close()
        pa.terminate()
        porcupine.delete()
    except Exception as e:
        print(f"[WAKE] Error: {e}")

# ==============================
# STREAMING AUDIO + VAD
# ==============================
def audio_stream_worker():
    """Stream audio and detect speech. In IDLE mode, check for Hindi wakeword. In CONVO, send to QA."""
    try:
        import webrtcvad
        vad = webrtcvad.Vad(2)
        ring_buffer = deque(maxlen=20)
        triggered = False
        voiced_frames = []

        def callback(indata, frames, time, status):
            if is_speaking.is_set():
                return  # Ignore input while speaking
            nonlocal triggered, voiced_frames
            audio = indata[:, 0].copy()
            samples = (audio * 32767).astype(np.int16).tobytes()
            is_speech = vad.is_speech(samples, MIC_RATE)
            ring_buffer.append((audio, is_speech))
            
            if not triggered:
                if sum(1 for _, speech in ring_buffer if speech) > 0.8 * ring_buffer.maxlen:
                    triggered = True
                    voiced_frames = list(ring_buffer)
                    ring_buffer.clear()
            else:
                voiced_frames.append((audio, is_speech))
                if sum(1 for _, speech in ring_buffer if not speech) > 0.9 * ring_buffer.maxlen:
                    audio_seg = np.concatenate([f[0] for f in voiced_frames])
                    if len(audio_seg) >= int(0.4 * MIC_RATE):
                        # Check current mode
                        with state_lock:
                            current_mode = state["mode"]
                        
                        try:
                            audio_segment_queue.put_nowait((current_mode, audio_seg))
                        except queue.Full:
                            pass  # Drop audio safely (real-time rule)

                    
                    triggered = False
                    voiced_frames = []

        with sd.InputStream(channels=1, samplerate=MIC_RATE, dtype='float32', callback=callback, blocksize=int(MIC_RATE * 0.03)):
            while not stop_event.is_set():
                time.sleep(0.1)
    except Exception as e:
        print(f"[AUDIO] Error: {e}")

def audio_processor_worker():
    while not stop_event.is_set():
        if is_speaking.is_set():
            time.sleep(0.05)
            continue

        try:
            mode, audio = audio_segment_queue.get(timeout=0.2)

            # --- IDLE → Hindi wakeword detection ---
            if mode == "IDLE":
                try:
                    segments, _ = whisper_model.transcribe(
                        audio, beam_size=1, language="hi", vad_filter=False
                    )
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    if text and is_hindi_wakeword(text):
                        wakeword_queue.put("hindi")
                except Exception as e:
                    print("[AUDIO_PROC][HINDI] Error:", e)

            # --- CONVO → normal QA flow ---
            elif mode == "CONVO":
                question_queue.put(("PROCESS_AUDIO", "auto", audio))

            audio_segment_queue.task_done()

        except queue.Empty:
            continue

# ==============================
# QA WORKER
# ==============================
def qa_worker():
    while not stop_event.is_set():
        try:
            item = question_queue.get(timeout=0.2)
            if item[0] == "PROCESS_AUDIO":
                _, _, audio = item
                try:
                    segments, info = whisper_model.transcribe(audio, beam_size=1, language=None)
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                    lang = info.language or "en"
                    if text and not is_garbage_text(text):
                        with state_lock:
                            current_lang = state["language"]
                        if lang.startswith("hi"):
                            with state_lock:
                                state["language"] = "hi"
                            current_lang = "hi"
                        question_queue.put((text, current_lang, time.time()))
                except Exception as e:
                    print("[STT] Error:", e)
                question_queue.task_done()
                continue

            question, detected_lang, _ = item  # ← Use detected_lang, not session lang
            # Correct language if needed
            if detected_lang.startswith("hi") and is_english_text(question):
                detected_lang = "en"
                print(f"[LANG] Detected as Hindi, but corrected to English: '{question}'")
            with state_lock:
                state["last_user_text"] = question
                state["last_activity_time"] = time.time()
                # Optional: update session language to match last question
                state["language"] = detected_lang

            # Decide TTS language based on DETECTED language of the question
            tts_lang = detected_lang

            if is_thank_you(question):
                bye = "You're welcome! Have a nice day!"
                # with state_lock:
                #     state.update({
                #         "mode": "IDLE",
                #         "status_ui": "Say Hello Robot",
                #         "current_user_name": None,
                #         "last_user_text": "",
                #         "last_answer_text": bye
                #     })
                tts_queue.put((bye, tts_lang, True))  # Use detected language for goodbye

            elif is_valid_question(question):
                answer, _ = qa_db.find_answer(question, detected_lang)
                if not answer:
                    answer = "I'm sorry, I don't have that information." if detected_lang.startswith("en") else "क्षमा करें, मुझे यह जानकारी नहीं है।"
                
                with state_lock:
                    state["last_answer_text"] = answer
                
                # tts_queue.put((answer, tts_lang, True))  # ✅ Use detected_lang for TTS
                tts_queue.put((answer, tts_lang, False))
            question_queue.task_done()
        except:
            continue

# ==============================
# FACE RECOGNITION WORKER
# ==============================
def face_worker():
    if not HAS_DLIB:
        return
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    frame_count = 0
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret: continue
        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recognized_name = None

        try:
            dets = face_detector(rgb, 0)
            if dets:
                rect = max(dets, key=lambda r: r.area())
                desc = np.array(facerec.compute_face_descriptor(rgb, sp(rgb, rect)))
                for name, ref_desc in known_faces:
                    if np.linalg.norm(desc - ref_desc) < 0.45:
                        recognized_name = name
                        break

                # Auto-greet known person in IDLE mode
                
                now = time.time()
                with state_lock:
                    if (
                        recognized_name
                        and not is_speaking.is_set()
                        and state["mode"] == "IDLE"
                        and now - state["last_face_time"] > 20
                    ):
                        greet = f"Good day, {recognized_name}! How can I help you?"
                        wake_request_queue.put((
                            "face",
                            "en",
                            f"Good day, {recognized_name}! How can I help you?"
                        ))

        except:
            pass

        with state_lock:
            state["last_known_person"] = recognized_name or "-"

        time.sleep(0.1)
    cap.release()

# ==============================
# FLASK UI
# ==============================
app = Flask(__name__)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title> शंकरा मित्र </title>
<style>
body {
  margin: 0;
  font-family: 'Poppins', sans-serif;
  background: #000; /* Solid black */
  color: #e9eefc;
  overflow-x: hidden;
  overflow-y: hidden;
}
@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
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

    // Always reflect backend state — cleanup is handled server-side
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
      // Loop + autoplay will handle continuous playback
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

setInterval(update, 500); // Slightly faster for smoother UX
</script>
</head>
<body>
<header>
  <img src="/logo" alt="Logo">
  <h1>शंकरा मित्र </h1>
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
        "is_speaking": is_currently_speaking  # ← NEW
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
        # Fallback to listen
        full_path = os.path.join(ASSETS_DIR, "robot_listen.mp4")
        if not os.path.isfile(full_path):
            return "Video not available", 404
        filename = "robot_listen.mp4"

    # Use make_response (imported at top!)
    response = make_response(send_from_directory(ASSETS_DIR, filename))
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    return response

@app.route("/logo")
def logo():
    return send_from_directory(ASSETS_DIR, "logo.jpeg")  # or "logo.jpg"

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

    print("\n🚀 SSIPMT OFFLINE RECEPTIONIST READY!")
    print("🗣️  Say 'Hello Robot' to start!")
    print("🌐 UI: http://127.0.0.1:8000\n")
    app.run(host="127.0.0.1", port=8000, debug=False)

if __name__ == "__main__":
    main()