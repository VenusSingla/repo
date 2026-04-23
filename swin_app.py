import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, SwinForImageClassification
from transformers import VitsModel, AutoTokenizer
from scipy.io.wavfile import write
import io

import os
from huggingface_hub import login

@st.cache_resource
def load_models():
    # Authenticate with HuggingFace using secret token
    hf_token = st.secrets.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)

    try:
        model = SwinForImageClassification.from_pretrained(
            "vsingla/Swin_transformer",
            token=hf_token
        )
        processor = AutoImageProcessor.from_pretrained(
            "vsingla/Swin_transformer",
            token=hf_token
        )
        model_speech = VitsModel.from_pretrained(
            "facebook/mms-tts-pan",
            token=hf_token
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "facebook/mms-tts-pan",
            token=hf_token
        )
        return model, processor, model_speech, tokenizer

    except OSError as e:
        st.error(
            "❌ Failed to load models. Possible causes:\n"
            "1. The model repo `vsingla/Swin_transformer` is private — add HF_TOKEN to Streamlit secrets\n"
            "2. The repo name/path is incorrect\n"
            "3. No internet access on this deployment\n\n"
            f"Details: {str(e)}"
        )
        st.stop()

model, processor, model_speech, tokenizer = load_models()
punjabi_translation =  {
    'ACCIDENT': 'ਹਾਦਸਾ','AEROPLANE': 'ਹਵਾਈ ਜਹਾਜ਼', 'AFRAID': 'ਡਰ', 'AGREE': 'ਸਹਿਮਤ', 'ALL': 'ਸਾਰੇ', 'ANGRY': 'ਗੁੱਸਾ', 'ANYTHING': 'ਕੁਝ ਵੀ',
    'APPRECIATE': 'ਸਰਾਹਨਾ', 'BABY': 'ਬੇਬੀ', 'BAD': 'ਬੁਰਾ', 'BARK': 'ਸੱਕ', 'BEAUTIFUL': 'ਸੁੰਦਰ', 'BECOME': 'ਬਣ','BED': 'ਬਿਸਤਰੇ',
    'BIG': 'ਵੱਡਾ', 'BITE': 'ਦੰਦੀ', 'BORED': 'ਬੋਰ', 'BRING': 'ਲਿਆਓ', 'BUSY': 'ਰੁੱਝੇ ਹੋਏ', 'CALCULATOR': 'ਕੈਲਕੁਲੇਟਰ', 'CALL': 'ਕਾਲ ਕਰੋ',
    'CHAT': 'ਗੱਲਬਾਤ', 'CLASS': 'ਕਲਾਸ',  'COLD': 'ਠੰਡਾ','COLLEGE': 'ਕਾਲਜ',
    'COMB': 'ਕੰਘਾ',
    'COME': 'ਆਉਣਾ',
    'CONGRATULATIONS': 'ਵਧਾਈਆਂ',
    'COST': 'ਲਾਗਤ',
    'CRYING': 'ਰੋਣਾ',
    'DANCE': 'ਡਾਂਸ',
    'DARE': 'ਹਿੰਮਤ',
    'DIFFERENCE': 'ਅੰਤਰ',
    'DILEMMA': 'ਦੁਬਿਧਾ',
    'DISAPPOINTED': 'ਨਿਰਾਸ਼',
    'DO': 'ਕਰਨਾ',
    'DOCTOR': 'ਡਾਕਟਰ',
    'DONT CARE': 'ਫਰਕ ਨਹੀਂ ਪੈਂਦਾ',
    'DRINK': 'ਪੀਓ',
    'ENJOY': 'ਅਨੰਦ ਲਓ',
    'FARM': 'ਖੇਤ',
    'FARMER': 'ਕਿਸਾਨ',
    'FAVOUR': 'ਪਾਰਟੀ ਪੱਖ',
    'FEVER': 'ਬੁਖ਼ਾਰ',
    'FINE': 'ਠੀਕ ਹੈ',
    'FOOD': 'ਭੋਜਨ',
    'FREE': 'ਮੁਫਤ',
    'FRIEND': 'ਦੋਸਤ',
    'FROM': 'ਤੋਂ',
    'GO': 'ਜਾਓ',
    'GOOD': 'ਚੰਗਾ',
    'GRATEFUL': 'ਸ਼ੁਕਰਗੁਜ਼ਾਰ',
    'HAD': 'ਸੀ',
    'HAPPENED': 'ਹੋਇਆ',
    'HAPPY': 'ਖੁਸ਼',
    'HEAR': 'ਸੁਣੋ',
    'HEART': 'ਦਿਲ',
    'HELLO': 'ਸਤ ਸ੍ਰੀ ਅਕਾਲ',
    'HELP': 'ਮਦਦ ਕਰੋ',
    'HI': 'ਹਾਇ',
    'HIDING': 'ਲੁਕਾਉਣਾ',
    'HOW': 'ਕਿਵੇਂ',
    'HUNGRY': 'ਭੁੱਖੇ',
    'HURT': 'ਦੁੱਖ',
    'I': 'ਮੈਂ',
    'ME': 'ਮੈਨੂੰ',
    'MY':'ਮੇਰਾ',
    'MINE':'ਮੇਰਾ',
    'KIND': 'ਦਿਆਲੂ',
    'LEAVE': 'ਛੱਡਣਾ',
    'LIKE': 'ਪਸੰਦ',
    'LOVE': 'ਪਿਆਰ',
    'MEDICINE': 'ਦਵਾਈ',
    'MEET': 'ਮਿਲਦਾ',
    'NAME': 'ਨਾਮ',
    'NICE': 'ਅਚ਼ਾ',
    'NOT': 'ਨਹੀਂ',
    'NUMBER': 'ਨੰਬਰ',
    'OLD AGE': 'ਬੁਢਾਪਾ',
    'ON THE WAY': 'ਰਸਤੇ ਵਿਚ ਹਾਂ',
    'OUTSIDE': 'ਬਾਹਰ',
    'PHONE': 'ਫੋਨ',
    'PLACE': 'ਸਥਾਨ',
    'PLEASE': 'ਕਿਰਪਾ ਕਰਕੇ',
    'POUR': 'ਡੋਲ੍ਹ',
    'PREPARE': 'ਤਿਆਰੀ ਕਰੋ',
    'PROMISE': 'ਵਾਅਦਾ',
    'REALLY': 'ਸੱਚ ਵਿੱਚ',
    'REPEAT': 'ਦੁਹਰਾਓ',
    'ROOM': 'ਕਮਰਾ',
    'SCHOOL': 'ਸਕੂਲ',
    'SERVE': 'ਸੇਵਾ',
    'SHIRT': 'ਸ਼ਰਟ',
    'SIKH': 'ਸਿੱਖ',
    'SITTING': 'ਬੈਠਣਾ',
    'SLEEP': 'ਸੌਣਾ',
    'SLOWER': 'ਧੀਰੇ',
    'SOFTLY': 'ਮਿੱਠੇ',
    'SOMETHING': 'ਕੁਝ',
    'SOME HOW': 'ਕੁਝ',
    'SOME ONE': 'ਕੋਈ',
    'SORRY': 'ਮਾਫ ਕਰਨਾ',
    'SO MUCH': 'ਬਹੁਤ ਜ਼ਿਆਦਾ',
    'SPEAK': 'ਗੱਲ',
    'STOCK': 'ਸਟਾਕ',
    'STOP': 'ਰੁਕੋ',
    'STUBBORN': 'ਜ਼ਿੱਦੀ',
    'SURE': 'ਯਕੀਨ',
    'TAKE CARE': 'ਆਪਣਾ ਖਿਆਲ ਰੱਖਣਾ',
    'TAKE TIME': 'ਸਮਾਂ ਲਵੋ',
    'TALK': 'ਗੱਲ',
    'TELL': 'ਦੱਸੋ',
    'THANK': 'ਧੰਨਵਾਦ',
    'THAT': 'ਕਿ',
    'THINGS': 'ਗੱਲ',
    'THINK': 'ਸੋਚੋ',
    'THIRSTY': 'ਪਿਆਸ',
    'TIRED': 'ਥੱਕੇ ਹੋਏ',
    'TODAY': 'ਅੱਜ',
    'TRAIN': 'ਰੇਲਗੱਡੀ',
    'TRUST': 'ਭਰੋਸਾ',
    'TRUTH': 'ਸੱਚ',
    'TURN ON': 'ਚਾਲੂ ਕਰੋ',
    'UNDERSTAND': 'ਸਮਝ',
    'WANT': 'ਚਾਹੁੰਦੇ',
    'WATER': 'ਪਾਣੀ',
    'WEAR': 'ਪਹਿਨੋ',
    'WELCOME': 'ਜੀ ਆਇਆ ਨੂੰ',
    'WHAT': 'ਕੀ',
    'WHERE': 'ਕਿੱਥੇ',
    'WHO': 'ਕੌਣ',
    'WORRY': 'ਚਿੰਤਾ',
    'YOU': 'ਤੁਸੀਂ',
    'YOUR':'ਤੁਹਾਡਾ'
}
# Label mapping
id2label = {
    '0': 'ACCIDENT', '1': 'AEROPLANE', '2': 'AFRAID', '3': 'AGREE', '4': 'ALL', '5': 'ANGRY', '6': 'ANYTHING',
    '7': 'APPRECIATE', '8': 'BABY', '9': 'BAD', '10': 'BARK', '11': 'BEAUTIFUL', '12': 'BECOME', '13': 'BED',
    '14': 'BIG', '15': 'BITE', '16': 'BORED', '17': 'BRING', '18': 'BUSY', '19': 'CALCULATOR', '20': 'CALL',
    '21': 'CHAT', '22': 'CLASS', '23': 'COLD', '24': 'COLLEGE', '25': 'COMB', '26': 'COME', '27': 'CONGRATULATIONS',
    '28': 'COST', '29': 'CRYING', '30': 'DANCE', '31': 'DARE', '32': 'DIFFERENCE', '33': 'DILEMMA', '34': 'DISAPPOINTED',
    '35': 'DO', '36': 'DOCTOR', '37': 'DONT_CARE', '38': 'DRINK', '39': 'ENJOY', '40': 'FARM', '41': 'FARMER',
    '42': 'FAVOUR', '43': 'FEVER', '44': 'FINE', '45': 'FOOD', '46': 'FREE', '47': 'FRIEND', '48': 'FROM', '49': 'GO',
    '50': 'GOOD', '51': 'GRATEFUL', '52': 'HAD', '53': 'HAPPENED', '54': 'HAPPY', '55': 'HEAR', '56': 'HEART',
    '57': 'HELLO', '58': 'HELP', '59': 'HI', '60': 'HIDING', '61': 'HOW', '62': 'HUNGRY', '63': 'HURT', '64': 'I_ME_MY_MINE',
    '65': 'KIND', '66': 'LEAVE', '67': 'LIKE', '68': 'LOVE', '69': 'MEDICINE', '70': 'MEET', '71': 'NAME', '72': 'NICE',
    '73': 'NOT', '74': 'NUMBER', '75': 'OLD_AGE', '76': 'ON_THE_WAY', '77': 'OUTSIDE', '78': 'PHONE', '79': 'PLACE',
    '80': 'PLEASE', '81': 'POUR', '82': 'PREPARE', '83': 'PROMISE', '84': 'REALLY', '85': 'REPEAT', '86': 'ROOM',
    '87': 'SCHOOL', '88': 'SERVE', '89': 'SHIRT', '90': 'SIKH', '91': 'SITTING', '92': 'SLEEP', '93': 'SLOWER',
    '94': 'SOFTLY', '95': 'SOMETHING', '96': 'SOME_HOW', '97': 'SOME_ONE', '98': 'SORRY', '99': 'SO_MUCH', '100': 'SPEAK',
    '101': 'STOCK', '102': 'STOP', '103': 'STUBBORN', '104': 'SURE', '105': 'TAKE_CARE', '106': 'TAKE_TIME', '107': 'TALK',
    '108': 'TELL', '109': 'THANK', '110': 'THAT', '111': 'THINGS', '112': 'THINK', '113': 'THIRSTY', '114': 'TIRED',
    '115': 'TODAY', '116': 'TRAIN', '117': 'TRUST', '118': 'TRUTH', '119': 'TURN_ON', '120': 'UNDERSTAND', '121': 'WANT',
    '122': 'WATER', '123': 'WEAR', '124': 'WELCOME', '125': 'WHAT', '126': 'WHERE', '127': 'WHO', '128': 'WORRY', '129': 'YOU_YOUR'
}

# Function to draw landmarks on image
def draw_landmarks(image, landmarks):
    draw = ImageDraw.Draw(image)
    for (x, y) in landmarks:
        # Draw a red circle at each landmark point
        draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill='red', outline='red')
    return image

# Inference function
# Inference function
async def perform_inference(image, threshold=0.3):  # Threshold can be tuned
    try:
        # Ensure the image is in RGB mode (convert if necessary)
        if image.mode != 'RGB':
            print(f"Image mode before conversion: {image.mode}")  # Debugging log
            image = image.convert('RGB')
            print(f"Image mode after conversion: {image.mode}")  # Debugging log

        # Process the image for inference
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_probs, predicted_index = torch.max(predictions, dim=1)
            predicted_index = predicted_index.item()
            confidence = predicted_probs.item()
        
        if confidence < threshold:
            # Return a tuple with None for landmarks when not recognized
            return "Not Recognized", "ਪਛਾਣਿਆ ਨਹੀਂ ਗਿਆ", []
        else:
            predicted_label = id2label.get(str(predicted_index), "Unknown")
            # NEW
            label_key = predicted_label.replace('_', ' ')          # DONT_CARE → DONT CARE
            punjabi_text = punjabi_translation.get(label_key, predicted_label)  # fallback to English if missing
            landmarks = [(100, 150), (200, 250), (300, 350)]
            return predicted_label, punjabi_text, landmarks
    except Exception as e:
        print(f"Error during inference: {str(e)}")  # Log the error
        return "Error", str(e), []  # Return an empty list for landmarks in case of error

# Function to generate audio
def generate_audio(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model_speech(**inputs).waveform

    waveform_np = output.cpu().numpy()
    sampling_rate = model_speech.config.sampling_rate

    # Save as WAV file
    audio_filename = "output_audio.wav"
    write(audio_filename, sampling_rate, waveform_np.T)  # Fix: Using scipy to save

    return audio_filename  # Return file path

# ------------------ Streamlit UI ------------------

st.set_page_config(page_title="ISL to Punjabi Translator", layout="centered")
st.title("🖐️ Sanket2Shabd")
st.write("Upload an image or capture from webcam")

# Session states
if "latest_image" not in st.session_state:
    st.session_state.latest_image = None
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# Camera + Cancel buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📷 Capture Image"):
        st.session_state.show_camera = True
with col2:
    if st.session_state.show_camera:
        if st.button("❌ Cancel Capture"):
            st.rerun()

# Upload input
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_DIMENSION = 2000  # Max image dimension (you can adjust this)
generate_speech_disabled = True
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        # Check image size (in MB) and resize if necessary
        file_size = uploaded_file.size  # in bytes
        if file_size > MAX_FILE_SIZE:
            st.warning(f"Image is too large. Resizing it to fit within the size limit of 200MB.")
            # Resize image to fit within the max file size limit
            image = image.resize((MAX_DIMENSION, MAX_DIMENSION), Image.ANTIALIAS)
        
        # Ensure image is in RGB mode before processing
        if image.mode != 'RGB':
            image = image.convert('RGB')
        st.image(image, caption="Processed Image", use_container_width=True)
        # Perform inference once
        predicted_label, punjabi_translation, landmarks = asyncio.run(perform_inference(image))
        if landmarks:  # Check if landmarks are present
            image_with_landmarks = draw_landmarks(image.copy(), landmarks)
            st.image(image_with_landmarks, caption="Image with Landmarks", use_container_width=True)
        else:
            st.warning("No landmarks to display.")

        if predicted_label == "Not Recognized":
            st.error("Not recognized sign")
            st.info(f"Punjabi Translation: {punjabi_translation}")
            generate_speech_disabled = True
        else:
            st.success(f"Predicted: {predicted_label}")
            st.info(f"Punjabi Translation: {punjabi_translation}")
            generate_speech_disabled = False
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        print(f"Error: {str(e)}") 
    
    st.session_state.latest_image = uploaded_file

elif st.session_state.show_camera:
    capture_image = st.camera_input("Take a Picture")
    if capture_image is not None:
        st.session_state.latest_image = capture_image
        st.session_state.show_camera = False

# Process latest image if available
if st.session_state.latest_image is not None:
    try:
        # Important: properly open the uploaded/captured file as image
        image = Image.open(st.session_state.latest_image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        st.image(image, caption="Processed Image", use_container_width=True)

        # Perform inference
        predicted_label, punjabi_translation, landmarks = asyncio.run(perform_inference(image))

        if landmarks:
            image_with_landmarks = draw_landmarks(image.copy(), landmarks)
            st.image(image_with_landmarks, caption="Image with Landmarks", use_container_width=True)

        if predicted_label == "Not Recognized":
            st.error("Not recognized sign")
            st.info(f"Punjabi Translation: {punjabi_translation}")
            generate_speech_disabled = True
        else:
            st.success(f"Predicted: {predicted_label}")
            st.info(f"Punjabi Translation: {punjabi_translation}")
            generate_speech_disabled = False

    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        print(f"Error: {str(e)}")

    st.session_state.latest_image = None  # Reset after processing
