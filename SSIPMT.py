import os
import time
import threading
import queue
import subprocess
import traceback
from pynput import keyboard

import numpy as np
import sounddevice as sd
import pandas as pd
from difflib import SequenceMatcher

from faster_whisper import WhisperModel

# Wrap native libraries that may fail to import due to system compatibility
try:
    import dlib
    HAS_DLIB = True
except Exception as e:
    print(f"[INIT] Warning: dlib import failed: {e}")
    dlib = None
    HAS_DLIB = False

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    HAS_HSEMO = True
except Exception as e:
    print(f"[INIT] Warning: HSEmotionRecognizer import failed: {e}")
    HSEmotionRecognizer = None
    HAS_HSEMO = False

# ============================================================
# PATHS & CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOICES_DIR = os.path.join(BASE_DIR, "voices")
MODELS_DIR = os.path.join(BASE_DIR, "models")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Excel Q&A file path
QA_EXCEL_PATH = os.path.join(BASE_DIR, "qa_database.xlsx")

AMY_MODEL = os.path.join(VOICES_DIR, "amy", "model.onnx")
PRATHAM_MODEL = os.path.join(VOICES_DIR, "pratham", "model.onnx")

WHISPER_LOCAL_DIR = os.path.join(MODELS_DIR, "stt")

SHAPE_PREDICTOR_PATH = os.path.join(MODELS_DIR, "shape_predictor_5_face_landmarks.dat")
FACE_RECOG_MODEL_PATH = os.path.join(MODELS_DIR, "dlib_face_recognition_resnet_model_v1.dat")

LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
ROBOT_VIDEO_PATH = os.path.join(ASSETS_DIR, "robot.mp4")

is_listening = threading.Event() 
mic_enabled = False

# Audio
MIC_DEVICE = None
MIC_RATE = 16000
CHANNELS = 1

CAMERA_INDEX = 0

MODE_IDLE = "IDLE"
MODE_CONVO = "CONVO"

CONVO_IDLE_TIMEOUT = 35.0  # Increased timeout

# Similarity threshold for matching questions
SIMILARITY_THRESHOLD = 0.45

# Audio detection thresholds
MIN_AUDIO_ENERGY = 0.004  # Minimum energy to consider as speech (lowered threshold)
MIN_QUESTION_LENGTH = 2  # Minimum words to consider as valid question

# Flag to prevent mic listening while speaking
is_speaking = threading.Event()

# Global state
state_lock = threading.Lock()
state = {
    "mode": MODE_IDLE,
    "language": "en",
    "status": "idle",
    "last_user_text": "",
    "last_answer_text": "",
    "last_emotion": "-",
    "mic_enabled": mic_enabled,
    "last_known_person": None,
    "current_user_name": None,
    "last_activity_time": time.time(),
    "last_face_time": 0.0,
}

stop_event = threading.Event()

question_queue: "queue.Queue[tuple[str,str,float]]" = queue.Queue()
tts_queue: "queue.Queue[tuple[str,str]]" = queue.Queue()

# ============================================================
# EXCEL Q&A DATABASE
# ============================================================

class QADatabase:
    """
    Loads Q&A from Excel file and provides fast lookup.
    
    Expected Excel format:
    | question_en | answer_en | question_hi | answer_hi | keywords |
    
    - question_en: Question in English
    - answer_en: Answer in English
    - question_hi: Question in Hindi (optional)
    - answer_hi: Answer in Hindi (optional)
    - keywords: Comma-separated keywords for better matching (optional)
    """
    
    def __init__(self, excel_path: str):
        self.qa_data = []
        self.keywords_map = {}
        self.load_database(excel_path)
    
    def load_database(self, excel_path: str):
        """Load Q&A data from Excel file."""
        if not os.path.isfile(excel_path):
            print(f"[QA] ERROR: Excel file not found at {excel_path}")
            print(f"[QA] Please create {excel_path} with columns: question_en, answer_en, question_hi, answer_hi, keywords")
            return
        
        try:
            df = pd.read_excel(excel_path)
            print(f"[QA] Loaded {len(df)} Q&A entries from Excel.")
            
            for _, row in df.iterrows():
                entry = {
                    "question_en": str(row.get("question_en", "")).strip().lower(),
                    "answer_en": str(row.get("answer_en", "")).strip(),
                    "question_hi": str(row.get("question_hi", "")).strip().lower() if pd.notna(row.get("question_hi")) else "",
                    "answer_hi": str(row.get("answer_hi", "")).strip() if pd.notna(row.get("answer_hi")) else "",
                    "keywords": []
                }
                
                # Parse keywords
                if pd.notna(row.get("keywords")):
                    keywords = str(row.get("keywords")).lower().split(",")
                    entry["keywords"] = [k.strip() for k in keywords if k.strip()]
                
                if entry["question_en"] or entry["question_hi"]:
                    self.qa_data.append(entry)
                    
                    # Build keyword index
                    for kw in entry["keywords"]:
                        if kw not in self.keywords_map:
                            self.keywords_map[kw] = []
                        self.keywords_map[kw].append(entry)
            
            print(f"[QA] Database ready with {len(self.qa_data)} entries.")
            
        except Exception as e:
            print(f"[QA] ERROR loading Excel: {e}")
            traceback.print_exc()
    
    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()
    
    def _keyword_match_score(self, query: str, keywords: list) -> float:
        """Calculate keyword match score."""
        if not keywords:
            return 0.0
        query_words = set(query.lower().split())
        matches = sum(1 for kw in keywords if kw in query_words or any(kw in w for w in query_words))
        return matches / len(keywords) if keywords else 0.0
    
    def find_answer(self, question: str, lang: str) -> tuple[str, float]:
        """
        Find the best matching answer for a question.
        Returns (answer, confidence_score)
        """
        question = question.strip().lower()
        
        if not question or not self.qa_data:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for entry in self.qa_data:
            # Check English question
            if entry["question_en"]:
                sim_en = self._calculate_similarity(question, entry["question_en"])
                kw_score = self._keyword_match_score(question, entry["keywords"])
                combined_score = (sim_en * 0.7) + (kw_score * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = entry
            
            # Check Hindi question
            if entry["question_hi"]:
                sim_hi = self._calculate_similarity(question, entry["question_hi"])
                kw_score = self._keyword_match_score(question, entry["keywords"])
                combined_score = (sim_hi * 0.7) + (kw_score * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = entry
        
        if best_match and best_score >= SIMILARITY_THRESHOLD:
            # Return answer in appropriate language
            if lang.startswith("hi") and best_match["answer_hi"]:
                return best_match["answer_hi"], best_score
            return best_match["answer_en"], best_score
        
        return None, best_score

# Load Q&A Database
print("[QA] Loading Q&A Database from Excel...")
qa_db = QADatabase(QA_EXCEL_PATH)

# ============================================================
# UTILS
# ============================================================

def normalize_text(s: str) -> str:
    return s.strip().lower()

def detect_wake_language(text: str):
    t = text.lower()
    if "namaste" in t or "नमस्ते" in t:
        return "hi"
    if "hello" in t or "hallo" in t:
        return "en"
    return None

def is_thank_you(text: str) -> bool:
    t = text.lower()
    return (
        "thank you" in t or
        "thanks" in t or
        "thankyou" in t or
        "धन्यवाद" in t or
        "शुक्रिया" in t
    )

def is_garbage_text(text: str) -> bool:
    """
    Filter out garbage/noise that Whisper might produce.
    Returns True if text should be ignored.
    """
    t = text.lower().strip()
    
    # Empty or single character
    if len(t) < 2:
        return True
    
    # Common Whisper hallucinations/noise - ONLY robot's own outputs
    garbage_phrases = [
        "thanks for watching",
        "thank you for watching", 
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "bye bye",
        "[music]",
        "[applause]",
        "...",
        # Robot output phrases to ignore
        "please check with the reception",
        "how can i help",
        "good day",
        "you look neutral",
        "i don't have information",
        "i don't have an answer",
        "that's beyond my knowledge",
    ]
    
    for phrase in garbage_phrases:
        if phrase in t:
            return True
    
    # Non-English text hallucinations (common Whisper errors)
    # Only reject if it's ENTIRELY non-English
    non_english_indicators = ['ご', '中文', '日本', 'العربية']
    if any(indicator in text for indicator in non_english_indicators):
        # Check if it's ONLY non-English
        if not any(c.isascii() and c.isalpha() for c in text):
            return True
    
    return False

def is_valid_question(text: str) -> bool:
    """Check if text looks like a valid question/request."""
    t = text.lower().strip()
    
    # Must have minimum length
    if len(t.split()) < MIN_QUESTION_LENGTH:
        return False
    
    # Check for question indicators
    question_words = ["what", "where", "when", "how", "who", "which", "why", "is", "are", "can", "do", "does", "tell", "show", "help", "क्या", "कहाँ", "कब", "कैसे", "कौन", "बताओ", "बताइए"]
    
    has_question_word = any(t.startswith(w) or f" {w} " in f" {t} " for w in question_words)
    has_question_mark = "?" in text
    
    # Also accept statements that might be requests
    request_words = ["need", "want", "looking for", "find", "about", "information", "details", "चाहिए", "बताओ"]
    has_request = any(w in t for w in request_words)
    
    return has_question_word or has_question_mark or has_request

# ============================================================
# TTS (Piper)
# ============================================================

def tts_speak(text: str, lang: str):
    text = text.strip()
    if not text:
        return
    
    # Set speaking flag to block mic input
    is_speaking.set()
    
    model = PRATHAM_MODEL if lang.startswith("hi") else AMY_MODEL
    out_wav = os.path.join(BASE_DIR, "tmp_tts.wav")

    print(f"[TTS] ({lang}) {text}")
    try:
        subprocess.run(
            ["piper", "--model", model, "--output_file", out_wav],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        subprocess.run(
            ["aplay", out_wav],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if os.path.exists(out_wav):
            os.remove(out_wav)
    except Exception as e:
        print("[TTS] ERROR:", e)
    finally:
        # Wait a bit after speaking to avoid echo pickup
        time.sleep(0.5)
        is_speaking.clear()

def tts_worker():
    while not stop_event.is_set():
        try:
            text, lang = tts_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        with state_lock:
            if state["mode"] == MODE_CONVO:
                state["status"] = "speaking"
        tts_speak(text, lang)
        with state_lock:
            if state["mode"] == MODE_CONVO:
                state["status"] = "listening"
            else:
                state["status"] = "idle"
        tts_queue.task_done()

# ============================================================
# STT (Whisper)
# ============================================================

print("[STT] Loading Whisper model on CUDA...")
if os.path.isdir(WHISPER_LOCAL_DIR) and os.listdir(WHISPER_LOCAL_DIR):
    whisper_model_path = WHISPER_LOCAL_DIR
else:
    whisper_model_path = "small"
whisper_model = WhisperModel(
    whisper_model_path,
    device="cuda",
    compute_type="float16"
)
print("[STT] Whisper loaded.")

def stt_transcribe(duration_sec: float):
    """Record only when mic is enabled from UI."""
    global mic_enabled

    # Wait if speaker is speaking
    while is_speaking.is_set():
        time.sleep(0.1)

    # Wait for user to enable mic from UI
    while not mic_enabled and not stop_event.is_set():
        print("[STT] Mic is OFF (UI). Waiting for enable...")
        time.sleep(0.5)

    # Mark listening
    is_listening.set()
    text, lang = "", "en"

    try:
        print(f"[STT] Recording {duration_sec:.1f} seconds...")
        audio = sd.rec(
            int(duration_sec * MIC_RATE),
            samplerate=MIC_RATE,
            channels=CHANNELS,
            dtype="float32",
            device=MIC_DEVICE
        )
        sd.wait()
        audio = audio[:, 0]

        energy = np.sqrt(np.mean(audio**2))
        print(f"[STT] Audio energy: {energy:.4f}")

        if energy < MIN_AUDIO_ENERGY:
            print("[STT] Too quiet — skipping.")
            return "", "en"

        print("[STT] Transcribing...")
        segments, info = whisper_model.transcribe(audio, beam_size=1, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        lang = info.language or "en"
        print(f"[STT] → {text} ({lang})")
        return text, lang

    except Exception as e:
        print("[STT] ERROR:", e)
        traceback.print_exc()
        return "", "en"

    finally:
        time.sleep(0.2)
        is_listening.clear()
        print("[STT] Done listening.")

# ============================================================
# Q&A ANSWER FUNCTION (Replaces LLM)
# ============================================================

def build_emotion_response(emotion: str, lang: str) -> str:
    """Generate emotion-aware prefix for response."""
    e = (emotion or "").lower()
    
    if lang.startswith("hi"):
        if "sad" in e:
            return "मुझे आपकी चिंता है। "
        if "anger" in e or "angry" in e:
            return "कृपया शांत रहें। "
        return ""
    else:
        if "sad" in e:
            return "I hope everything is okay. "
        if "anger" in e or "angry" in e:
            return "I understand. Let me help you. "
        return ""

def get_answer(question: str, lang: str, emotion: str) -> str:
    """Get answer from Excel database."""
    print(f"[QA] Looking up: {question}")
    
    answer, confidence = qa_db.find_answer(question, lang)
    
    if answer:
        print(f"[QA] Found answer with confidence: {confidence:.2f}")
        emotion_prefix = build_emotion_response(emotion, lang)
        return emotion_prefix + answer
    else:
        print(f"[QA] No match found (best score: {confidence:.2f})")
        if lang.startswith("hi"):
            return "क्षमा करें, मैं समझ नहीं पाया,कृपया इसी प्रश्न को इंग्लिश में पूछें"
        else:
            return "I'm sorry, I don't have information about that. Please check with the reception desk."

def qa_worker():
    """Worker thread for answering questions (replaces llm_worker)."""
    while not stop_event.is_set():
        try:
            question, lang, q_time = question_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        # Keep only latest question
        while True:
            try:
                q2 = question_queue.get_nowait()
                question, lang, q_time = q2
            except queue.Empty:
                break

        with state_lock:
            emotion = state["last_emotion"]
            state["status"] = "thinking"

        # Get answer from Excel database (much faster than LLM!)
        answer = get_answer(question, lang, emotion)

        with state_lock:
            if state["mode"] != MODE_CONVO:
                question_queue.task_done()
                continue
            state["last_answer_text"] = answer
            state["last_activity_time"] = time.time()

        tts_queue.put((answer, lang))
        question_queue.task_done()

# ============================================================
# EMOTION + FACE RECOGNITION
# ============================================================

print("[EMOTION] Loading HSEmotion model...")
if HAS_HSEMO:
    try:
        emotion_model = HSEmotionRecognizer(model_name="enet_b0_8_best_vgaf")
        print("[EMOTION] Loaded.")
    except Exception as e:
        print("[EMOTION] ERROR loading HSEmotion:", e)
        emotion_model = None
else:
    print("[EMOTION] HSEmotion not available; emotion detection disabled.")
    emotion_model = None

if HAS_DLIB and os.path.isfile(SHAPE_PREDICTOR_PATH) and os.path.isfile(FACE_RECOG_MODEL_PATH):
    try:
        face_detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        facerec = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)
        FACE_RECOG_ENABLED = True
        print("[FACE] dlib face recognition enabled.")
    except Exception as e:
        print(f"[FACE] ERROR initializing dlib: {e}")
        face_detector = sp = facerec = None
        FACE_RECOG_ENABLED = False
else:
    face_detector = sp = facerec = None
    FACE_RECOG_ENABLED = False
    if not HAS_DLIB:
        print("[FACE] dlib not available; face recognition disabled.")
    else:
        print("[FACE] dlib models not found. Face recognition disabled.")

known_faces = []

def compute_descriptor(frame_rgb, rect):
    try:
        if not frame_rgb.flags['C_CONTIGUOUS']:
            frame_rgb = np.ascontiguousarray(frame_rgb)
        if frame_rgb.dtype != np.uint8:
            frame_rgb = frame_rgb.astype(np.uint8)
        shape = sp(frame_rgb, rect)
        vec = facerec.compute_face_descriptor(frame_rgb, shape)
        return np.array(vec)
    except Exception as e:
        print("[FACE] compute_descriptor ERROR:", e)
        return None

def load_known_faces():
    if not FACE_RECOG_ENABLED:
        return
    if not os.path.isdir(KNOWN_FACES_DIR):
        print("[FACE] known_faces directory missing, skipping.")
        return

    for person_name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = dlib.load_rgb_image(img_path)
                dets = face_detector(img, 1)
                if len(dets) == 0:
                    print(f"[FACE] No face found in {img_name}")
                    continue
                rect = max(dets, key=lambda r: r.width() * r.height())
                shape = sp(img, rect)
                vec = facerec.compute_face_descriptor(img, shape)
                desc = np.array(vec)
                known_faces.append((person_name, desc))
                print(f"[FACE] Loaded {person_name} from {img_name}")
            except Exception as e:
                print(f"[FACE] Error loading {img_path}:", e)

load_known_faces()

def best_match_name(desc, tolerance=0.45):
    if not known_faces or desc is None:
        return None
    best_name = None
    best_dist = 1e9
    for name, ref in known_faces:
        d = float(np.linalg.norm(desc - ref))
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist <= tolerance:
        return best_name
    return None

def face_emotion_worker():
    try:
        import cv2
    except ImportError:
        print("[FACE] OpenCV not available. Face/emotion worker disabled.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[FACE] Cannot open camera.")
        return

    print("[FACE] Face+Emotion worker started.")
    
    # ...existing code...
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame for performance
    
    try:
        while not stop_event.is_set():
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % process_every_n_frames != 0:
                time.sleep(0.05)
                continue

            # Convert BGR to RGB and ensure proper format
            try:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if not frame_rgb.flags['C_CONTIGUOUS']:
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                if frame_rgb.dtype != np.uint8:
                    frame_rgb = frame_rgb.astype(np.uint8)
            except Exception as e:
                print("[FACE] Frame conversion error:", e)
                continue

            dets = []
            if FACE_RECOG_ENABLED:
                try:
                    dets = face_detector(frame_rgb, 0)  # Use 0 for faster detection
                except Exception as e:
                    print("[FACE] detector error:", e)
                    dets = []

            primary_rect = None
            primary_area = 0
            recognized_names = set()

            for rect in dets:
                try:
                    w = max(0, rect.width())
                    h = max(0, rect.height())
                    area = w * h
                    if area <= 0:
                        continue
                    desc = compute_descriptor(frame_rgb, rect)
                    if desc is None:
                        continue
                    name = best_match_name(desc)
                    if name:
                        recognized_names.add(name)
                    if area > primary_area:
                        primary_area = area
                        primary_rect = rect
                except Exception as e:
                    print("[FACE] Face processing error:", e)
                    continue

            primary_emotion = None
            if primary_rect is not None and emotion_model is not None:
                x1, y1, x2, y2 = (
                    max(0, primary_rect.left()),
                    max(0, primary_rect.top()),
                    min(frame_rgb.shape[1] - 1, primary_rect.right()),
                    min(frame_rgb.shape[0] - 1, primary_rect.bottom())
                )
                if x2 > x1 and y2 > y1:
                    face_img = frame_rgb[y1:y2, x1:x2, :]
                    try:
                        emo, _ = emotion_model.predict_emotions(face_img, logits=False)
                        primary_emotion = emo
                    except Exception as e:
                        print("[EMOTION] face error:", e)

            elif primary_rect is None and emotion_model is not None:
                try:
                    emo, _ = emotion_model.predict_emotions(frame_rgb, logits=False)
                    primary_emotion = emo
                except Exception as e:
                    print("[EMOTION] frame-level error:", e)

            now = time.time()
            with state_lock:
                if primary_emotion:
                    state["last_emotion"] = primary_emotion

                primary_known = None
                if recognized_names:
                    primary_known = list(recognized_names)[0]
                state["last_known_person"] = primary_known

                if primary_rect is not None and FACE_RECOG_ENABLED and known_faces:
                    desc_primary = compute_descriptor(frame_rgb, primary_rect)
                    name_primary = best_match_name(desc_primary) if desc_primary is not None else None

                    if name_primary:
                        if (
                            state["mode"] == MODE_IDLE and
                            now - state["last_face_time"] > 20
                        ):
                            state["mode"] = MODE_CONVO
                            state["language"] = "en"
                            state["current_user_name"] = name_primary
                            state["last_face_time"] = now

                            greet_text = (
                                f"Good day, {name_primary}! "
                                # f"You look {primary_emotion or 'good'} today. "
                                f"How can I help you?"
                            )

                            state["last_answer_text"] = greet_text
                            state["last_user_text"] = ""
                            state["status"] = "speaking"
                            state["last_activity_time"] = now

                            tts_queue.put((greet_text, "en"))

            time.sleep(0.3)

    except Exception as e:
        print("[FACE] worker ERROR:", e)
        traceback.print_exc()
    finally:
        cap.release()
        print("[FACE] Face+Emotion worker stopped.")

# ============================================================
# CONVERSATION WORKER (Wakeword + Q&A)
# ============================================================

def conversation_worker():
    print("[MAIN] Conversation worker started.")
    while not stop_event.is_set():
        # Skip if TTS is speaking
        if is_speaking.is_set():
            time.sleep(0.2)
            continue
            
        with state_lock:
            mode = state["mode"]
            lang = state["language"]
            last_act = state["last_activity_time"]

        now = time.time()

        if mode == MODE_CONVO and now - last_act > CONVO_IDLE_TIMEOUT:
            with state_lock:
                convo_lang = state["language"]
            if convo_lang == "hi":
                bye = "आप कुछ नहीं बोल रहे हैं, मैं अब वापस प्रतीक्षा मोड में जा रहा हूँ।"
            else:
                bye = "You seem busy, I will go back to idle mode now."
            with state_lock:
                state["last_answer_text"] = bye
                state["status"] = "speaking"
            tts_queue.put((bye, convo_lang))
            with state_lock:
                state["mode"] = MODE_IDLE
                state["current_user_name"] = None
                state["last_user_text"] = ""
                state["last_answer_text"] = ""
                state["status"] = "idle"
            continue

        if mode == MODE_IDLE:
            with state_lock:
                state["status"] = "listening"
            text, det_lang = stt_transcribe(2.5)
            
            # Filter garbage
            if not text.strip() or is_garbage_text(text):
                with state_lock:
                    state["status"] = "idle"
                continue

            wake_lang = detect_wake_language(text)
            if wake_lang:
                with state_lock:
                    state["mode"] = MODE_CONVO
                    state["language"] = wake_lang
                    state["current_user_name"] = None
                    state["last_user_text"] = ""
                    state["last_answer_text"] = ""
                    state["last_activity_time"] = time.time()

                if wake_lang == "hi":
                    greet = "नमस्ते! मैं आपकी क्या मदद कर सकता हूँ?"
                else:
                    greet = "Hello! How can I help you today?"

                with state_lock:
                    state["last_answer_text"] = greet
                    state["status"] = "speaking"
                tts_queue.put((greet, wake_lang))
            else:
                with state_lock:
                    state["status"] = "idle"
            continue

        # MODE_CONVO: listen for user question
        with state_lock:
            state["status"] = "listening"
        text, det_lang = stt_transcribe(4.0)
        
        # Filter garbage and self-heard speech
        if not text.strip() or is_garbage_text(text):
            print(f"[MAIN] Ignoring garbage/noise: '{text}'")
            continue

        with state_lock:
            convo_lang = state["language"]
            state["last_user_text"] = text
            state["last_activity_time"] = time.time()
            # Update language based on Whisper detection if Hindi is detected
            if det_lang and det_lang.startswith("hi"):
                state["language"] = "hi"
                convo_lang = "hi"
            elif det_lang and det_lang.startswith("en"):
                state["language"] = "en"
                convo_lang = "en"

        # Check for goodbye
        if is_thank_you(text):
            if convo_lang == "hi":
                bye = "धन्यवाद! आपका दिन शुभ हो!"
            else:
                bye = "You're welcome! Have a nice day ahead."
            with state_lock:
                state["last_answer_text"] = bye
                state["status"] = "speaking"
            tts_queue.put((bye, convo_lang))
            with state_lock:
                state["mode"] = MODE_IDLE
                state["current_user_name"] = None
                state["last_user_text"] = ""
                state["last_answer_text"] = ""
            continue

        # Only process if it looks like a valid question
        if is_valid_question(text):
            question_queue.put((text, convo_lang, time.time()))
        else:
            print(f"[MAIN] Text doesn't look like a question, ignoring: '{text}'")

    print("[MAIN] Conversation worker stopped.")

# ============================================================
# ADVANCED WEB UI – GLASS DESIGN + CONTEXT VIDEO
# ============================================================

from flask import Flask, jsonify, render_template_string, send_from_directory
import webbrowser

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
  background: #000;
  background-size: 400% 400%;
  animation: gradientShift 15s ease infinite;
  color: #e9eefc;
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
#mic_button {
  width: 100%;
  padding: 1rem;
  font-size: 1.2rem;
  border: none;
  border-radius: 15px;
  background: rgba(0,255,150,0.15);
  color: #00ffaa;
  font-weight: 600;
  text-shadow: 0 0 8px #00ffaa;
  backdrop-filter: blur(10px);
  cursor: pointer;
  transition: all 0.3s ease;
}
#mic_button:hover {
  background: rgba(0,255,150,0.25);
  transform: scale(1.05);
}
#mic_button.active {
  background: rgba(255,0,90,0.25);
  color: #ff5070;
  text-shadow: 0 0 8px #ff5070;
}
.label { color: #a2c4ff; font-weight: 600; }
textarea {
  width: 95%;
  height: 120px;
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 12px;
  padding: 0.8rem;
  color: #fff;
  resize: none;
  font-size: 0.95rem;
}
.videoBox {
  border-radius: 20px;
  overflow: hidden;
  box-shadow: 0 0 25px rgba(0,0,0,0.3);
  position: relative;
}
video {
  width: 100%;
  border-radius: 20px;
}
footer {
  text-align: center;
  color: #d1d9ff;
  padding: 1rem;
  font-size: 0.9rem;
}
</style>
<script>
let currentVideo = "";

async function update() {
  const res = await fetch('/state');
  const s = await res.json();

  document.getElementById('status').innerText = s.mode + " / " + s.status;
  document.getElementById('mic').innerText = s.mic;
  document.getElementById('emotion').innerText = s.emotion;
  document.getElementById('person').innerText = s.person || '-';
  document.getElementById('user_text').value = s.user || '';
  document.getElementById('bot_text').value = s.bot || '';

  const vid = document.getElementById('robot_video');
  if (s.video !== currentVideo) {
    vid.src = '/video/' + s.video;
    vid.play();
    currentVideo = s.video;
  }
}
setInterval(update, 1000);
async function toggleMic() {
  const res = await fetch('/toggle_mic', { method: 'POST' });
  const data = await res.json();
  const btn = document.getElementById('mic_button');
  if (data.mic_enabled) {
    btn.innerText = "🔇 Disable Mic";
    btn.classList.add('active');
  } else {
    btn.innerText = "🎙️ Enable Mic";
    btn.classList.remove('active');
  }
}

async function refreshMicButton() {
  const res = await fetch('/mic_state');
  const data = await res.json();
  const btn = document.getElementById('mic_button');
  if (data.mic_enabled) {
    btn.innerText = "🔇 Disable Mic";
    btn.classList.add('active');
  } else {
    btn.innerText = "🎙️ Enable Mic";
    btn.classList.remove('active');
  }
}

// keep button synced
setInterval(refreshMicButton, 1500);

</script>
</head>
<body>
<header>
  <img src="/logo" alt="Logo">
  <h1>Shankra Mitra Receptionist Robot</h1>
</header>

<div class="grid">
  <div class="card">
    <div><span class="label">Status:</span> <span id="status">loading...</span></div>
    <div><span class="label">Mic:</span> <span id="mic">loading...</span></div>
    <div><span class="label">Emotion:</span> <span id="emotion">-</span></div>
    <div><span class="label">Known person:</span> <span id="person">-</span></div>
    <button id="mic_button" onclick="toggleMic()">🎙️ Enable Mic</button>
    <br>
    <div><span class="label">User says:</span></div>
    <textarea id="user_text" readonly></textarea>
    <div><span class="label">Shankra Mitra replies:</span></div>
    <textarea id="bot_text" readonly></textarea>
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

<footer>Designed & developed in SSIPMT, Raipur</footer>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/logo")
def logo():
    return send_from_directory(ASSETS_DIR, "logo.jpeg")

@app.route("/video/<vid_type>")
def serve_video(vid_type):
    mapping = {
        "idle": "robot_listen.mp4",
        "answer": "robot_answer.mp4",
        "sorry": "robot_sorry.mp4"
    }
    file = mapping.get(vid_type, "robot_listen.mp4")
    path = os.path.join(ASSETS_DIR, file)
    if not os.path.exists(path):
        return "Video not found", 404
    return send_from_directory(ASSETS_DIR, file)

@app.route("/state")
def state_json():
    with state_lock:
        mode = state["mode"]
        status = state["status"]
        emo = state["last_emotion"]
        person = state["last_known_person"]
        user = state["last_user_text"]
        bot = state["last_answer_text"]

    if status == "speaking":
        if "sorry" in (bot or "").lower():
            video_type = "sorry"
        else:
            video_type = "answer"
    elif status in ["listening", "thinking"]:
        video_type = "answer"
    else:
        video_type = "idle"

    if status == "idle":
        mic = "idle (waiting for wakeword 'Hello' / 'Namaste')"
    elif status == "listening":
        mic = "🎙️ Listening..."
    elif status == "thinking":
        mic = "... thinking ..."
    elif status == "speaking":
        mic = "🔊 Speaking..."
    else:
        mic = status

    return jsonify({
        "mode": mode,
        "status": status,
        "mic": mic,
        "emotion": emo or "-",
        "person": person or "-",
        "user": user,
        "bot": bot,
        "video": video_type
    })

@app.route("/toggle_mic", methods=["POST"])
def toggle_mic():
    global mic_enabled
    mic_enabled = not mic_enabled
    print(f"[WEB] 🎙️ Mic {'ENABLED' if mic_enabled else 'DISABLED'} from UI")
    return jsonify({"mic_enabled": mic_enabled})

@app.route("/mic_state")
def mic_state():
    return jsonify({"mic_enabled": mic_enabled})


def keyboard_m_button_listener():
    """
    Listens for the physical USB 'M' button (acts as keyboard M key)
    and toggles the microphone state.
    Works without root or GUI focus.
    """
    global mic_enabled
    print("[KEYBOARD] Waiting for hardware 'M' button input...")

    def on_press(key):
        global mic_enabled
        try:
            if key.char and key.char.lower() == 'm':
                mic_enabled = not mic_enabled
                print(f"[KEYBOARD] 🎙️ Mic {'ENABLED' if mic_enabled else 'DISABLED'} (hardware M button)")
        except AttributeError:
            pass  # ignore other keys

    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

def main():
    threading.Thread(target=tts_worker, daemon=True).start()
    threading.Thread(target=qa_worker, daemon=True).start()
    threading.Thread(target=face_emotion_worker, daemon=True).start()
    threading.Thread(target=conversation_worker, daemon=True).start()
    threading.Thread(target=keyboard_m_button_listener, daemon=True).start()

    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    print("[WEB] Glass UI running at http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=False)


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()