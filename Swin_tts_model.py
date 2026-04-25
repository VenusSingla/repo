import os
import re
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SwinForImageClassification
from huggingface_hub import login
from scipy.io.wavfile import write as wavwrite
import mediapipe as mp

# ═══════════════════════════════════════════════════════════════════════════
#  PUNJABI TTS  (inlined — no separate punjabi_tts.py needed in repo)
#  Based on: Singh & Lehal (2012), COLING 2012
# ═══════════════════════════════════════════════════════════════════════════

_ABBR = {
    "ਡਾ.": "ਡਾਕਟਰ", "ਸ੍ਰ.": "ਸਰਦਾਰ", "ਪ੍ਰੋ.": "ਪ੍ਰੋਫੈਸਰ",
    "ਸ਼੍ਰੀ": "ਸ਼੍ਰੀਮਾਨ", "ਕਿ.ਮੀ.": "ਕਿਲੋਮੀਟਰ", "ਰੁ.": "ਰੁਪਏ",
}
_GUR_DIGITS = {"੦":"0","੧":"1","੨":"2","੩":"3","੪":"4",
               "੫":"5","੬":"6","੭":"7","੮":"8","੯":"9"}
_ONES_PA = ["","ਇੱਕ","ਦੋ","ਤਿੰਨ","ਚਾਰ","ਪੰਜ","ਛੇ","ਸੱਤ","ਅੱਠ","ਨੌਂ","ਦਸ",
            "ਗਿਆਰਾਂ","ਬਾਰਾਂ","ਤੇਰਾਂ","ਚੌਦਾਂ","ਪੰਦਰਾਂ","ਸੋਲਾਂ","ਸਤਾਰਾਂ","ਅਠਾਰਾਂ","ਉਨੀ","ਵੀਹ"]
_TENS_PA = ["","","ਵੀਹ","ਤੀਹ","ਚਾਲੀ","ਪੰਜਾਹ","ਸੱਠ","ਸੱਤਰ","ਅੱਸੀ","ਨੱਬੇ"]

_TTS_CACHE_DIR = "/tmp/pa_tts_cache"
_TTS_MEM: dict = {}
os.makedirs(_TTS_CACHE_DIR, exist_ok=True)


def _num_to_pa(n: int) -> str:
    if n == 0:
        return "ਜ਼ੀਰੋ"
    parts = []
    if n >= 100:
        parts.append(_ONES_PA[n // 100] + " ਸੌ")
        n %= 100
    if n >= 21:
        t, o = n // 10, n % 10
        parts.append(_TENS_PA[t] if o == 0 else _TENS_PA[t] + " " + _ONES_PA[o])
    elif n > 0:
        parts.append(_ONES_PA[n])
    return " ".join(parts)


def _tts_preprocess(text: str) -> str:
    for a, f in _ABBR.items():
        text = text.replace(a, f)
    def _exp(m):
        s = m.group(0)
        for g, a in _GUR_DIGITS.items():
            s = s.replace(g, a)
        try:
            return _num_to_pa(int(s))
        except ValueError:
            return s
    text = re.sub("[੦-੯0-9]+", _exp, text)
    text = re.sub(r"[।॥,;:!?\"'()\[\]{}<>]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _gtts_to_bytes(text: str) -> bytes:
    """
    Synthesise Punjabi text via gTTS and return raw MP3 bytes.

    Uses gTTS.save() → named temp file → read bytes → delete temp.
    write_to_fp(BytesIO) can silently produce 0 bytes in certain
    environments; save() is the most tested and reliable gTTS path.
    Exceptions propagate directly so callers can show them in the UI.
    Results are cached in memory and on disk (/tmp) for reuse.
    """
    import tempfile
    from gtts import gTTS

    cache_key = f"bytes:{text}"
    if cache_key in _TTS_MEM:
        return _TTS_MEM[cache_key]

    safe = re.sub(r"[^a-zA-Z0-9_]", "_",
                  text.encode("unicode_escape").decode("ascii"))[:80]
    cache_file = os.path.join(_TTS_CACHE_DIR, f"pa_{safe}.mp3")

    # Disk cache hit — read and return
    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        with open(cache_file, "rb") as f:
            data = f.read()
        _TTS_MEM[cache_key] = data
        return data

    # Generate: save() to a real named temp file, then read bytes
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        gTTS(text=text, lang="pa", slow=False).save(tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if len(data) == 0:
        raise RuntimeError(
            f"gTTS.save() produced 0 bytes for {text!r}. "
            "This usually means a network/firewall issue blocked the Google TTS API."
        )

    # Persist to disk cache (non-fatal if it fails)
    try:
        with open(cache_file, "wb") as f:
            f.write(data)
    except OSError:
        pass

    _TTS_MEM[cache_key] = data
    return data


def synthesize_punjabi(text: str) -> bytes:
    """Preprocess + synthesise. Returns raw MP3 bytes. Raises on failure."""
    return _gtts_to_bytes(_tts_preprocess(text))


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE HOLISTIC SETUP
# ═══════════════════════════════════════════════════════════════════════════

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


@st.cache_resource
def load_holistic():
    return mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

holistic = load_holistic()


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD ML MODELS  (Swin only — VitsModel removed)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    hf_token = st.secrets.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)
    try:
        swin = SwinForImageClassification.from_pretrained(
            "vsingla/Swin_transformer", token=hf_token
        )
        proc = AutoImageProcessor.from_pretrained(
            "vsingla/Swin_transformer", token=hf_token
        )
        return swin, proc
    except OSError as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

model, processor = load_models()


# ═══════════════════════════════════════════════════════════════════════════
#  TRANSLATION & LABEL MAPS
# ═══════════════════════════════════════════════════════════════════════════

punjabi_translation = {
    'ACCIDENT': 'ਹਾਦਸਾ', 'AEROPLANE': 'ਹਵਾਈ ਜਹਾਜ਼', 'AFRAID': 'ਡਰ', 'AGREE': 'ਸਹਿਮਤ',
    'ALL': 'ਸਾਰੇ', 'ANGRY': 'ਗੁੱਸਾ', 'ANYTHING': 'ਕੁਝ ਵੀ', 'APPRECIATE': 'ਸਰਾਹਨਾ',
    'BABY': 'ਬੱਚਾ', 'BAD': 'ਬੁਰਾ', 'BARK': 'ਭੌਂਕਣਾ', 'BEAUTIFUL': 'ਸੁੰਦਰ',
    'BECOME': 'ਬਣ', 'BED': 'ਬਿਸਤਰੇ', 'BIG': 'ਵੱਡਾ', 'BITE': 'ਦੰਦੀ',
    'BORED': 'ਬੋਰ', 'BRING': 'ਲਿਆਓ', 'BUSY': 'ਰੁੱਝੇ ਹੋਏ', 'CALCULATOR': 'ਕੈਲਕੁਲੇਟਰ',
    'CALL': 'ਕਾਲ ਕਰੋ', 'CHAT': 'ਗੱਲਬਾਤ', 'CLASS': 'ਜਮਾਤ', 'COLD': 'ਠੰਡਾ',
    'COLLEGE': 'ਕਾਲਜ', 'COMB': 'ਕੰਘਾ', 'COME': 'ਆਉਣਾ', 'CONGRATULATIONS': 'ਵਧਾਈਆਂ',
    'COST': 'ਕੀਮਤ', 'CRYING': 'ਰੋਣਾ', 'DANCE': 'ਨੱਚਣਾ', 'DARE': 'ਹਿੰਮਤ',
    'DIFFERENCE': 'ਅੰਤਰ', 'DILEMMA': 'ਦੁਬਿਧਾ', 'DISAPPOINTED': 'ਨਿਰਾਸ਼', 'DO': 'ਕਰਨਾ',
    'DOCTOR': 'ਡਾਕਟਰ', 'DONT CARE': 'ਫਰਕ ਨਹੀਂ ਪੈਂਦਾ', 'DRINK': 'ਪੀਓ', 'ENJOY': 'ਅਨੰਦ ਲਓ',
    'FARM': 'ਖੇਤ', 'FARMER': 'ਕਿਸਾਨ', 'FAVOUR': 'ਉਪਕਾਰ', 'FEVER': 'ਬੁਖ਼ਾਰ',
    'FINE': 'ਠੀਕ ਹੈ', 'FOOD': 'ਭੋਜਨ', 'FREE': 'ਮੁਫਤ', 'FRIEND': 'ਦੋਸਤ',
    'FROM': 'ਤੋਂ', 'GO': 'ਜਾਓ', 'GOOD': 'ਚੰਗਾ', 'GRATEFUL': 'ਸ਼ੁਕਰਗੁਜ਼ਾਰ',
    'HAD': 'ਸੀ', 'HAPPENED': 'ਹੋਇਆ', 'HAPPY': 'ਖੁਸ਼', 'HEAR': 'ਸੁਣੋ',
    'HEART': 'ਦਿਲ', 'HELLO': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ', 'HELP': 'ਮਦਦ ਕਰੋ', 'HI': 'ਹਾਇ',
    'HIDING': 'ਲੁਕਾਉਣਾ', 'HOW': 'ਕਿਵੇਂ', 'HUNGRY': 'ਭੁੱਖੇ', 'HURT': 'ਦੁੱਖ',
    'I ME MY MINE': 'ਮੈਂ', 'KIND': 'ਦਿਆਲੂ', 'LEAVE': 'ਛੱਡਣਾ', 'LIKE': 'ਪਸੰਦ',
    'LOVE': 'ਪਿਆਰ', 'MEDICINE': 'ਦਵਾਈ', 'MEET': 'ਮਿਲਦਾ', 'NAME': 'ਨਾਮ',
    'NICE': 'ਅਚ਼ਾ', 'NOT': 'ਨਹੀਂ', 'NUMBER': 'ਨੰਬਰ', 'OLD AGE': 'ਬੁਢਾਪਾ',
    'ON THE WAY': 'ਰਸਤੇ ਵਿਚ ਹਾਂ', 'OUTSIDE': 'ਬਾਹਰ', 'PHONE': 'ਫੋਨ', 'PLACE': 'ਸਥਾਨ',
    'PLEASE': 'ਕਿਰਪਾ ਕਰਕੇ', 'POUR': 'ਡੋਲ੍ਹ', 'PREPARE': 'ਤਿਆਰੀ ਕਰੋ', 'PROMISE': 'ਵਾਅਦਾ',
    'REALLY': 'ਸੱਚ ਵਿੱਚ', 'REPEAT': 'ਦੁਹਰਾਓ', 'ROOM': 'ਕਮਰਾ', 'SCHOOL': 'ਸਕੂਲ',
    'SERVE': 'ਸੇਵਾ', 'SHIRT': 'ਸ਼ਰਟ', 'SIKH': 'ਸਿੱਖ', 'SITTING': 'ਬੈਠੇ',
    'SLEEP': 'ਸੌਣਾ', 'SLOWER': 'ਹੌਲੀ', 'SOFTLY': 'ਨਰਮਾਈ ਨਾਲ', 'SOMETHING': 'ਕੁਝ',
    'SOME HOW': 'ਕਿਸੇ ਤਰ੍ਹਾਂ', 'SOME ONE': 'ਕੋਈ', 'SORRY': 'ਮਾਫ ਕਰਨਾ', 'SO MUCH': 'ਬਹੁਤ ਜ਼ਿਆਦਾ',
    'SPEAK': 'ਗੱਲ', 'STOCK': 'ਸਟਾਕ', 'STOP': 'ਰੁਕੋ', 'STUBBORN': 'ਜ਼ਿੱਦੀ',
    'SURE': 'ਯਕੀਨ', 'TAKE CARE': 'ਖਿਆਲ ਰੱਖਣਾ', 'TAKE TIME': 'ਸਮਾਂ ਲਵੋ', 'TALK': 'ਗੱਲ',
    'TELL': 'ਦੱਸੋ', 'THANK': 'ਧੰਨਵਾਦ', 'THAT': 'ਉਹ', 'THINGS': 'ਗੱਲ',
    'THINK': 'ਸੋਚੋ', 'THIRSTY': 'ਪਿਆਸ', 'TIRED': 'ਥੱਕੇ ਹੋਏ', 'TODAY': 'ਅੱਜ',
    'TRAIN': 'ਰੇਲਗੱਡੀ', 'TRUST': 'ਭਰੋਸਾ', 'TRUTH': 'ਸੱਚ', 'TURN ON': 'ਚਾਲੂ ਕਰੋ',
    'UNDERSTAND': 'ਸਮਝ', 'WANT': 'ਚਾਹੁੰਦੇ', 'WATER': 'ਪਾਣੀ', 'WEAR': 'ਪਹਿਨੋ',
    'WELCOME': 'ਜੀ ਆਇਆ ਨੂੰ', 'WHAT': 'ਕੀ', 'WHERE': 'ਕਿੱਥੇ', 'WHO': 'ਕੌਣ',
    'WORRY': 'ਚਿੰਤਾ', 'YOU YOUR': 'ਤੁਸੀਂ',
}

id2label = {
    '0': 'ACCIDENT', '1': 'AEROPLANE', '2': 'AFRAID', '3': 'AGREE', '4': 'ALL',
    '5': 'ANGRY', '6': 'ANYTHING', '7': 'APPRECIATE', '8': 'BABY', '9': 'BAD',
    '10': 'BARK', '11': 'BEAUTIFUL', '12': 'BECOME', '13': 'BED', '14': 'BIG',
    '15': 'BITE', '16': 'BORED', '17': 'BRING', '18': 'BUSY', '19': 'CALCULATOR',
    '20': 'CALL', '21': 'CHAT', '22': 'CLASS', '23': 'COLD', '24': 'COLLEGE',
    '25': 'COMB', '26': 'COME', '27': 'CONGRATULATIONS', '28': 'COST', '29': 'CRYING',
    '30': 'DANCE', '31': 'DARE', '32': 'DIFFERENCE', '33': 'DILEMMA', '34': 'DISAPPOINTED',
    '35': 'DO', '36': 'DOCTOR', '37': 'DONT CARE', '38': 'DRINK', '39': 'ENJOY',
    '40': 'FARM', '41': 'FARMER', '42': 'FAVOUR', '43': 'FEVER', '44': 'FINE',
    '45': 'FOOD', '46': 'FREE', '47': 'FRIEND', '48': 'FROM', '49': 'GO',
    '50': 'GOOD', '51': 'GRATEFUL', '52': 'HAD', '53': 'HAPPENED', '54': 'HAPPY',
    '55': 'HEAR', '56': 'HEART', '57': 'HELLO', '58': 'HELP', '59': 'HI',
    '60': 'HIDING', '61': 'HOW', '62': 'HUNGRY', '63': 'HURT', '64': 'I ME MY MINE',
    '65': 'KIND', '66': 'LEAVE', '67': 'LIKE', '68': 'LOVE', '69': 'MEDICINE',
    '70': 'MEET', '71': 'NAME', '72': 'NICE', '73': 'NOT', '74': 'NUMBER',
    '75': 'OLD AGE', '76': 'ON THE WAY', '77': 'OUTSIDE', '78': 'PHONE', '79': 'PLACE',
    '80': 'PLEASE', '81': 'POUR', '82': 'PREPARE', '83': 'PROMISE', '84': 'REALLY',
    '85': 'REPEAT', '86': 'ROOM', '87': 'SCHOOL', '88': 'SERVE', '89': 'SHIRT',
    '90': 'SIKH', '91': 'SITTING', '92': 'SLEEP', '93': 'SLOWER', '94': 'SOFTLY',
    '95': 'SOMETHING', '96': 'SOME HOW', '97': 'SOME ONE', '98': 'SORRY', '99': 'SO MUCH',
    '100': 'SPEAK', '101': 'STOCK', '102': 'STOP', '103': 'STUBBORN', '104': 'SURE',
    '105': 'TAKE CARE', '106': 'TAKE TIME', '107': 'TALK', '108': 'TELL', '109': 'THANK',
    '110': 'THAT', '111': 'THINGS', '112': 'THINK', '113': 'THIRSTY', '114': 'TIRED',
    '115': 'TODAY', '116': 'TRAIN', '117': 'TRUST', '118': 'TRUTH', '119': 'TURN ON',
    '120': 'UNDERSTAND', '121': 'WANT', '122': 'WATER', '123': 'WEAR', '124': 'WELCOME',
    '125': 'WHAT', '126': 'WHERE', '127': 'WHO', '128': 'WORRY', '129': 'YOU YOUR',
}


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════

for _k, _v in {
    "latest_image": None, "show_camera": False, "current_image": None,
    "landmark_image": None, "predicted_label": None, "punjabi_text": None,
    "confidence": None, "audio_file": None, "last_upload": None,
    "keypoint_count": 0, "hand_detected": False,
    "pose_detected": False, "face_detected": False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════════════════
#  MEDIAPIPE HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def draw_styled_landmarks(image_rgb, results):
    annotated = image_rgb.copy()
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )
    return annotated


def extract_keypoints(image_rgb):
    results = holistic.process(image_rgb)
    annotated = draw_styled_landmarks(image_rgb, results)
    keypoints = []
    for attr in ("pose_landmarks", "left_hand_landmarks", "right_hand_landmarks"):
        lms = getattr(results, attr)
        if lms:
            keypoints.extend((lm.x, lm.y, lm.z) for lm in lms.landmark)
    detection_info = {
        "face":       results.face_landmarks is not None,
        "pose":       results.pose_landmarks is not None,
        "left_hand":  results.left_hand_landmarks is not None,
        "right_hand": results.right_hand_landmarks is not None,
    }
    return annotated, keypoints, detection_info


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def perform_inference(pil_image: Image.Image, threshold: float = 0.3):
    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image)
        annotated_np, keypoints, detection_info = extract_keypoints(image_np)

        inputs = processor(images=Image.fromarray(annotated_np), return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_prob, top_idx = torch.max(probs, dim=1)
            confidence = top_prob.item()
            predicted_index = top_idx.item()

        if confidence < threshold:
            return ("Not Recognized", "ਪਛਾਣਿਆ ਨਹੀਂ ਗਿਆ",
                    confidence, annotated_np, keypoints, detection_info)

        label   = id2label.get(str(predicted_index), "Unknown")
        punjabi = punjabi_translation.get(label, label)
        return label, punjabi, confidence, annotated_np, keypoints, detection_info

    except Exception as e:
        return "Error", str(e), 0.0, np.array(pil_image), [], {}


# ═══════════════════════════════════════════════════════════════════════════
#  AUDIO GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_audio(text: str):
    """
    Returns raw MP3 bytes for st.audio(), or None on failure.
    Every failure path shows the REAL error in the UI.
    """
    if not text or not text.strip():
        st.error("No Punjabi text to synthesise.")
        return None

    try:
        from gtts import gTTS  # noqa
    except ImportError:
        st.error("gtts not installed. Add  to requirements.txt and redeploy.")
        return None

    try:
        data = synthesize_punjabi(text)
        return data
    except Exception as e:
        # Show the full real error — gTTS network errors, SSL issues etc.
        import traceback
        st.error(f"TTS failed: {type(e).__name__}: {e}")
        st.code(traceback.format_exc(), language="")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(pil_image: Image.Image):
    st.session_state.current_image = pil_image
    st.session_state.audio_file    = None

    with st.spinner("🔍 Extracting landmarks & running inference..."):
        label, punjabi, conf, annotated_np, keypoints, detection_info = \
            perform_inference(pil_image)

    st.session_state.predicted_label = label
    st.session_state.punjabi_text    = punjabi
    st.session_state.confidence      = conf
    st.session_state.landmark_image  = Image.fromarray(annotated_np)
    st.session_state.keypoint_count  = len(keypoints)
    st.session_state.face_detected   = detection_info.get("face", False)
    st.session_state.pose_detected   = detection_info.get("pose", False)
    st.session_state.hand_detected   = (
        detection_info.get("left_hand", False) or
        detection_info.get("right_hand", False)
    )


# ═══════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="ISL to Punjabi Translator", layout="centered")
st.title("🖐️ Sanket2Shabd")
st.write("Upload an image or capture from webcam to translate Indian Sign Language to Punjabi.")

col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Capture Image"):
        st.session_state.show_camera    = True
        st.session_state.current_image  = None
        st.session_state.predicted_label = None
        st.session_state.audio_file     = None
with col2:
    if st.session_state.show_camera:
        if st.button("❌ Cancel Capture"):
            st.session_state.show_camera = False
            st.rerun()

# -- File upload --
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    if st.session_state.last_upload != uploaded_file.name:
        st.session_state.last_upload = uploaded_file.name
        run_pipeline(Image.open(uploaded_file).convert("RGB"))

# -- Webcam capture --
elif st.session_state.show_camera:
    captured = st.camera_input("Take a Picture")
    if captured is not None:
        st.session_state.latest_image = captured
        st.session_state.show_camera  = False
        st.rerun()

if st.session_state.latest_image is not None:
    run_pipeline(Image.open(st.session_state.latest_image).convert("RGB"))
    st.session_state.latest_image = None

# -- Results --
if st.session_state.current_image is not None:

    col_orig, col_lm = st.columns(2)
    with col_orig:
        st.subheader("📷 Original")
        st.image(st.session_state.current_image, use_container_width=True)
    with col_lm:
        st.subheader("🦴 Landmarks")
        if st.session_state.landmark_image is not None:
            st.image(st.session_state.landmark_image, use_container_width=True)

    badges = [
        ("✅" if st.session_state.face_detected else "❌") + " Face",
        ("✅" if st.session_state.pose_detected else "❌") + " Pose",
        ("✅" if st.session_state.hand_detected else "❌") + " Hand(s)",
    ]
    st.caption(
        f"Detected: {' | '.join(badges)} | "
        f"Keypoints: {st.session_state.keypoint_count}"
    )

    st.divider()

    label   = st.session_state.predicted_label
    punjabi = st.session_state.punjabi_text
    conf    = st.session_state.confidence

    if label == "Not Recognized":
        st.error("❌ Sign not recognized (low confidence)")
        st.info("Try a clearer image with better lighting and a visible hand gesture.")
    elif label == "Error":
        st.error(f"An error occurred: {punjabi}")
    else:
        st.success(f"✅ Predicted: **{label}** (confidence: {conf:.1%})")
        st.info(f"ਪੰਜਾਬੀ: {punjabi}")

    if st.button("🔊 Generate Audio"):
        with st.spinner("Generating audio..."):
            audio_bytes = generate_audio(st.session_state.punjabi_text)
            if audio_bytes:
                st.session_state.audio_file = audio_bytes   # store raw bytes

    if st.session_state.audio_file:
        # Pass bytes directly — avoids all /tmp file-path & permission issues
        st.audio(st.session_state.audio_file, format="audio/mp3")
