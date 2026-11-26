import os
import tempfile
from gtts import gTTS
import streamlit as st

from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

# -----------------------------------
# CONFIG
# -----------------------------------

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

#MODEL_NAME = "llama-3.3-70b-versatile"

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Voice AI Assistant",
    page_icon="🎤",
    layout="centered"
)

st.title("🎤 Voice AI Assistant")
st.write("Speak a question and receive a voice response.")

# -----------------------------------
# INIT SESSION
# -----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# LOAD GROQ
# -----------------------------------

llm = ChatGroq(
   groq_api_key=GROQ_API_KEY,
   model="openai/gpt-oss-120b",
   temperature=0.3
)

# -----------------------------------
# LOAD WHISPER
# -----------------------------------

@st.cache_resource
def load_whisper():
    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

whisper_model = load_whisper()

# -----------------------------------
# SPEECH TO TEXT
# -----------------------------------

def transcribe_audio(audio_bytes):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".webm"
    ) as f:

        f.write(audio_bytes)
        temp_audio = f.name

    segments, _ = whisper_model.transcribe(temp_audio)

    text = ""

    for seg in segments:
        text += seg.text

    os.remove(temp_audio)

    return text.strip()

# -----------------------------------
# TEXT TO SPEECH
# -----------------------------------

def text_to_speech(text):

    tts = gTTS(
        text=text,
        lang="en"
    )

    output_file = "response.mp3"

    tts.save(output_file)

    return output_file

# -----------------------------------
# CHAT HISTORY
# -----------------------------------

st.subheader("Conversation")

for msg in st.session_state.messages:

    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -----------------------------------
# MICROPHONE
# -----------------------------------

audio = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    key="voice_recorder"
)

# -----------------------------------
# PROCESS AUDIO
# -----------------------------------

if audio:

    with st.spinner("Transcribing voice..."):

        user_text = transcribe_audio(audio["bytes"])

    st.chat_message("user").write(user_text)

    st.session_state.messages.append(
        HumanMessage(content=user_text)
    )

    with st.spinner("Thinking..."):

        response = llm.invoke(
            st.session_state.messages
        )

        answer = response.content

    st.session_state.messages.append(
        AIMessage(content=answer)
    )

    st.chat_message("assistant").write(answer)

    with st.spinner("Generating Voice..."):

        audio_file = text_to_speech(answer)

    st.audio(audio_file)

# -----------------------------------
# TEXT INPUT OPTION
# -----------------------------------

st.divider()

text_question = st.chat_input(
    "Ask something..."
)

if text_question:

    st.chat_message("user").write(text_question)

    st.session_state.messages.append(
        HumanMessage(content=text_question)
    )

    response = llm.invoke(
        st.session_state.messages
    )

    answer = response.content

    st.session_state.messages.append(
        AIMessage(content=answer)
    )

    st.chat_message("assistant").write(answer)

    audio_file = text_to_speech(answer)

    st.audio(audio_file)
