import streamlit as st
import cv2
import numpy as np
import torch
import asyncio
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor, VitsModel, AutoTokenizer
from googletrans import Translator
import tempfile

# Load models
model_path = "https://huggingface.co/vsingla/isl_trainer/"
model = ViTForImageClassification.from_pretrained(model_path)
processor = ViTFeatureExtractor.from_pretrained(model_path)
model_speech = VitsModel.from_pretrained("facebook/mms-tts-pan")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pan")
translator = Translator()

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
async def perform_inference(image):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_index = torch.argmax(predictions, dim=1).item()
    
    predicted_label = id2label.get(str(predicted_index), "Unknown")
    translation = await translator.translate(predicted_label, src='en', dest='pa')
    return predicted_label, translation.text
from scipy.io.wavfile import write

def generate_audio(text):
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model_speech(**inputs).waveform

    # Convert PyTorch tensor to NumPy
    waveform_np = output.numpy()

    # Define sampling rate
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

# Upload input
uploaded_file = st.file_uploader("📁 Upload Image", type=["jpg", "png", "jpeg"])

# Camera + Cancel buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("📷 Capture Image"):
        st.session_state.show_camera = True
with col2:
    if st.session_state.show_camera:
        if st.button("❌ Cancel Capture"):
            st.session_state.show_camera = False
            st.rerun()

# Camera input
if st.session_state.show_camera:
    capture_image = st.camera_input("Take a Picture")
    if capture_image is not None:
        st.session_state.latest_image = capture_image
        st.session_state.show_camera = False

# Save uploaded image
if uploaded_file is not None:
    st.session_state.latest_image = uploaded_file



# Process the latest image
if st.session_state.latest_image is not None:
    image = Image.open(st.session_state.latest_image)
    st.image(image, caption="Processed Image", use_column_width=True)
    # Perform inference
    predicted_label, punjabi_translation = asyncio.run(perform_inference(image))
    st.success(f"Predicted: {predicted_label}")
    st.info(f"Punjabi Translation: {punjabi_translation}")

    if st.button("Generate Speech"):
        audio_file = generate_audio(punjabi_translation)
        st.audio(audio_file, format="audio/wav")
        st.success("Audio generated successfully!")
