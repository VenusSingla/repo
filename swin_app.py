import streamlit as st
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SwinForImageClassification
from transformers import VitsModel, AutoTokenizer
from scipy.io.wavfile import write
from huggingface_hub import login
import cv2
import mediapipe as mp

# ------------------ MediaPipe Holistic Setup ------------------
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

@st.cache_resource
def load_holistic():
    return mp_holistic.Holistic(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

holistic = load_holistic()

# ------------------ Load ML Models ------------------
@st.cache_resource
def load_models():
    hf_token = st.secrets.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)
    try:
        model = SwinForImageClassification.from_pretrained(
            "vsingla/Swin_transformer", token=hf_token
        )
        processor = AutoImageProcessor.from_pretrained(
            "vsingla/Swin_transformer", token=hf_token
        )
        model_speech = VitsModel.from_pretrained(
            "facebook/mms-tts-pan", token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/mms-tts-pan", token=hf_token
        )
        return model, processor, model_speech, tokenizer
    except OSError as e:
        st.error(f"❌ Failed to load models.\n\nDetails: {str(e)}")
        st.stop()

model, processor, model_speech, tokenizer = load_models()

# ------------------ Punjabi Translation Map ------------------
punjabi_translation = {
    'ACCIDENT': 'ਹਾਦਸਾ', 'AEROPLANE': 'ਹਵਾਈ ਜਹਾਜ਼', 'AFRAID': 'ਡਰ', 'AGREE': 'ਸਹਿਮਤ',
    'ALL': 'ਸਾਰੇ', 'ANGRY': 'ਗੁੱਸਾ', 'ANYTHING': 'ਕੁਝ ਵੀ', 'APPRECIATE': 'ਸਰਾਹਨਾ',
    'BABY': 'ਬੱਚਾ', 'BAD': 'ਬੁਰਾ', 'BARK': 'ਭੌਂਕਣਾ', 'BEAUTIFUL': 'ਸੁੰਦਰ',
    'BECOME': 'ਬਣ', 'BED': 'ਬਿਸਤਰੇ', 'BIG': 'ਵੱਡਾ', 'BITE': 'ਦੰਦੀ',
    'BORED': 'ਬੋਰ', 'BRING': 'ਲਿਆਓ', 'BUSY': 'ਰੁੱਝੇ ਹੋਏ', 'CALCULATOR': 'ਕੈਲਕੁਲੇਟਰ',
    'CALL': 'ਕਾਲ ਕਰੋ', 'CHAT': 'ਗੱਲਬਾਤ', 'CLASS': 'ਜਮਾਤ', 'COLD': 'ਠੰਡਾ',
    'COLLEGE': 'ਕਾਲਜ', 'COMB': 'ਕੰਘਾ', 'COME': 'ਆਉਣਾ', 'CONGRATULATIONS': 'ਵਧਾਈਆਂ',
    'COST': 'ਲਾਗਤ', 'CRYING': 'ਰੋਣਾ', 'DANCE': 'ਡਾਂਸ', 'DARE': 'ਹਿੰਮਤ',
    'DIFFERENCE': 'ਅੰਤਰ', 'DILEMMA': 'ਦੁਬਿਧਾ', 'DISAPPOINTED': 'ਨਿਰਾਸ਼', 'DO': 'ਕਰਨਾ',
    'DOCTOR': 'ਡਾਕਟਰ', 'DONT CARE': 'ਫਰਕ ਨਹੀਂ ਪੈਂਦਾ', 'DRINK': 'ਪੀਓ', 'ENJOY': 'ਅਨੰਦ ਲਓ',
    'FARM': 'ਖੇਤ', 'FARMER': 'ਕਿਸਾਨ', 'FAVOUR': 'ਪਾਰਟੀ ਪੱਖ', 'FEVER': 'ਬੁਖ਼ਾਰ',
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
    'SERVE': 'ਸੇਵਾ', 'SHIRT': 'ਸ਼ਰਟ', 'SIKH': 'ਸਿੱਖ', 'SITTING': 'ਬੈਠਣਾ',
    'SLEEP': 'ਸੌਣਾ', 'SLOWER': 'ਧੀਰੇ', 'SOFTLY': 'ਮਿੱਠੇ', 'SOMETHING': 'ਕੁਝ',
    'SOME HOW': 'ਕੁਝ', 'SOME ONE': 'ਕੋਈ', 'SORRY': 'ਮਾਫ ਕਰਨਾ', 'SO MUCH': 'ਬਹੁਤ ਜ਼ਿਆਦਾ',
    'SPEAK': 'ਗੱਲ', 'STOCK': 'ਸਟਾਕ', 'STOP': 'ਰੁਕੋ', 'STUBBORN': 'ਜ਼ਿੱਦੀ',
    'SURE': 'ਯਕੀਨ', 'TAKE CARE': 'ਆਪਣਾ ਖਿਆਲ ਰੱਖਣਾ', 'TAKE TIME': 'ਸਮਾਂ ਲਵੋ', 'TALK': 'ਗੱਲ',
    'TELL': 'ਦੱਸੋ', 'THANK': 'ਧੰਨਵਾਦ', 'THAT': 'ਕਿ', 'THINGS': 'ਗੱਲ',
    'THINK': 'ਸੋਚੋ', 'THIRSTY': 'ਪਿਆਸ', 'TIRED': 'ਥੱਕੇ ਹੋਏ', 'TODAY': 'ਅੱਜ',
    'TRAIN': 'ਰੇਲਗੱਡੀ', 'TRUST': 'ਭਰੋਸਾ', 'TRUTH': 'ਸੱਚ', 'TURN ON': 'ਚਾਲੂ ਕਰੋ',
    'UNDERSTAND': 'ਸਮਝ', 'WANT': 'ਚਾਹੁੰਦੇ', 'WATER': 'ਪਾਣੀ', 'WEAR': 'ਪਹਿਨੋ',
    'WELCOME': 'ਜੀ ਆਇਆ ਨੂੰ', 'WHAT': 'ਕੀ', 'WHERE': 'ਕਿੱਥੇ', 'WHO': 'ਕੌਣ',
    'WORRY': 'ਚਿੰਤਾ', 'YOU YOUR': 'ਤੁਸੀਂ'
}

# ------------------ Label Mapping ------------------
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
    '125': 'WHAT', '126': 'WHERE', '127': 'WHO', '128': 'WORRY', '129': 'YOU YOUR'
}

# ------------------ Session State Init ------------------
for key, default in {
    "latest_image": None,
    "show_camera": False,
    "current_image": None,
    "landmark_image": None,
    "predicted_label": None,
    "punjabi_text": None,
    "confidence": None,
    "audio_file": None,
    "last_upload": None,
    "keypoint_count": 0,
    "hand_detected": False,
    "pose_detected": False,
    "face_detected": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ------------------ Draw Landmarks ------------------
def draw_styled_landmarks(image_rgb, results):
    """Draw face, pose, left hand, right hand landmarks on image."""
    annotated = image_rgb.copy()

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )
    return annotated

# ------------------ Extract Keypoints ------------------
def extract_keypoints(image_rgb):
    """
    Run MediaPipe Holistic on an RGB numpy array.
    Returns annotated image, keypoints list, and detection flags.
    """
    results = holistic.process(image_rgb)
    annotated = draw_styled_landmarks(image_rgb, results)

    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append((lm.x, lm.y, lm.z))
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.append((lm.x, lm.y, lm.z))
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.append((lm.x, lm.y, lm.z))

    detection_info = {
        "face": results.face_landmarks is not None,
        "pose": results.pose_landmarks is not None,
        "left_hand": results.left_hand_landmarks is not None,
        "right_hand": results.right_hand_landmarks is not None,
    }

    return annotated, keypoints, detection_info

# ------------------ Inference ------------------
def perform_inference(pil_image: Image.Image, threshold=0.3):
    try:
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # Convert PIL → numpy RGB for MediaPipe
        image_np = np.array(pil_image)

        # Step 1: Extract landmarks with MediaPipe Holistic
        annotated_np, keypoints, detection_info = extract_keypoints(image_np)

        # Step 2: Convert annotated image back to PIL for Swin input
        # Using the landmark-annotated image gives the model richer spatial info
        annotated_pil = Image.fromarray(annotated_np)

        # Step 3: Run Swin Transformer classification
        inputs = processor(images=annotated_pil, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_probs, predicted_index = torch.max(predictions, dim=1)
            predicted_index = predicted_index.item()
            confidence = predicted_probs.item()

        if confidence < threshold:
            return "Not Recognized", "ਪਛਾਣਿਆ ਨਹੀਂ ਗਿਆ", confidence, annotated_np, keypoints, detection_info
        else:
            predicted_label = id2label.get(str(predicted_index), "Unknown")
            punjabi_text = punjabi_translation.get(predicted_label, predicted_label)
            return predicted_label, punjabi_text, confidence, annotated_np, keypoints, detection_info

    except Exception as e:
        return "Error", str(e), 0.0, np.array(pil_image), [], {}

# ------------------ Audio Generation ------------------
def generate_audio(text):
    try:
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = model_speech(**inputs)
        
        # VITS returns waveform of shape [1, 1, samples] or [1, samples]
        waveform = output.waveform
        
        # Safely squeeze all extra dimensions → 1D array
        waveform_np = waveform.squeeze().cpu().numpy()
        
        # If still 2D for some reason, take first channel
        if waveform_np.ndim == 2:
            waveform_np = waveform_np[0]
        
        # Normalize to [-1, 1] to avoid clipping/distortion
        max_val = np.abs(waveform_np).max()
        if max_val > 0:
            waveform_np = waveform_np / max_val
        
        # Scale to int16 range with slight headroom to avoid clipping
        waveform_int16 = (waveform_np * 32000).astype(np.int16)
        
        sampling_rate = model_speech.config.sampling_rate
        audio_filename = "/tmp/output_audio.wav"
        write(audio_filename, sampling_rate, waveform_int16)
        
        return audio_filename

    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
        return None

# ------------------ Run Pipeline on New Image ------------------
def run_pipeline(pil_image: Image.Image):
    """Run full pipeline and store all results in session_state."""
    st.session_state.current_image = pil_image
    st.session_state.audio_file = None

    with st.spinner("🔍 Extracting landmarks & running inference..."):
        label, punjabi, conf, annotated_np, keypoints, detection_info = perform_inference(pil_image)

    st.session_state.predicted_label = label
    st.session_state.punjabi_text = punjabi
    st.session_state.confidence = conf
    st.session_state.landmark_image = Image.fromarray(annotated_np)
    st.session_state.keypoint_count = len(keypoints)
    st.session_state.face_detected = detection_info.get("face", False)
    st.session_state.pose_detected = detection_info.get("pose", False)
    st.session_state.hand_detected = (
        detection_info.get("left_hand", False) or detection_info.get("right_hand", False)
    )

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="ISL to Punjabi Translator", layout="centered")
st.title("🖐️ Sanket2Shabd")
st.write("Upload an image or capture from webcam to translate Indian Sign Language to Punjabi.")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📷 Capture Image"):
        st.session_state.show_camera = True
        st.session_state.current_image = None
        st.session_state.predicted_label = None
        st.session_state.audio_file = None
with col2:
    if st.session_state.show_camera:
        if st.button("❌ Cancel Capture"):
            st.session_state.show_camera = False
            st.rerun()

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    if st.session_state.last_upload != uploaded_file.name:
        st.session_state.last_upload = uploaded_file.name
        pil_image = Image.open(uploaded_file).convert('RGB')
        run_pipeline(pil_image)

# ------------------ Camera Capture ------------------
elif st.session_state.show_camera:
    captured = st.camera_input("Take a Picture")
    if captured is not None:
        st.session_state.latest_image = captured
        st.session_state.show_camera = False
        st.rerun()

if st.session_state.latest_image is not None:
    pil_image = Image.open(st.session_state.latest_image).convert('RGB')
    st.session_state.latest_image = None
    run_pipeline(pil_image)

# ------------------ Display Results ------------------
if st.session_state.current_image is not None:

    col_orig, col_lm = st.columns(2)
    with col_orig:
        st.subheader("📷 Original")
        st.image(st.session_state.current_image, use_container_width=True)
    with col_lm:
        st.subheader("🦴 Landmarks")
        if st.session_state.landmark_image is not None:
            st.image(st.session_state.landmark_image, use_container_width=True)

    # Detection status badges
    badges = []
    if st.session_state.face_detected:
        badges.append("✅ Face")
    else:
        badges.append("❌ Face")
    if st.session_state.pose_detected:
        badges.append("✅ Pose")
    else:
        badges.append("❌ Pose")
    if st.session_state.hand_detected:
        badges.append("✅ Hand(s)")
    else:
        badges.append("❌ Hand(s)")

    st.caption(f"Detected: {' | '.join(badges)} | Keypoints: {st.session_state.keypoint_count}")

    # Prediction results
    label = st.session_state.predicted_label
    punjabi = st.session_state.punjabi_text
    conf = st.session_state.confidence

    st.divider()

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
            audio_file = generate_audio(st.session_state.punjabi_text)
            if audio_file:
                st.session_state.audio_file = audio_file
    
    if st.session_state.audio_file:
        st.audio(st.session_state.audio_file, format="audio/wav")
