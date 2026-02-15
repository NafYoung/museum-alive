import streamlit as st
import os
import asyncio
import edge_tts
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Museum Alive", page_icon="🏛️")
st.title("🏛️ Museum Alive: Let Artifacts Speak")
st.caption("✨ Cloud Edition: Artifact Storytelling & Voice Synthesis")

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.warning("⚠️ Please set DEEPSEEK_API_KEY in .env")
    else:
        st.success("✅ API Key Loaded")
    
    st.markdown("---")
    st.info("ℹ️ **Note:** This is the Cloud version. AI Vision is disabled to ensure fast performance. Please manually enter the artifact name.")

# Initialize DeepSeek Client
try:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
except Exception as e:
    st.error(f"Failed to initialize AI client: {e}")

# --- Core Functions ---

async def generate_audio(text, output_file="output.mp3"):
    """Generate audio using Edge-TTS (Free)"""
    try:
        # Use a high-quality Chinese voice
        communicate = edge_tts.Communicate(text, "zh-CN-YunxiNeural")
        await communicate.save(output_file)
    except Exception as e:
        st.error(f"Audio generation failed: {e}")

def get_artifact_story(artifact_name):
    """Ask DeepSeek to roleplay based on artifact name"""
    prompt = f"""
    The user is looking at a museum artifact named: "{artifact_name}".
    
    Please roleplay as this artifact.
    1. Start with a captivating hook.
    2. Describe your history and significance in the first person ("I").
    3. Keep it engaging, educational, and under 150 words.
    4. Language: Chinese (Mandarin).
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a sentient museum artifact with a distinct personality."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"I am unable to speak right now... (Error: {str(e)})"

# --- Main UI ---

# 1. Image Upload (Display Only)
uploaded_file = st.file_uploader("📸 Upload Artifact Photo (Optional)", type=["jpg", "png", "jpeg"])
if uploaded_file:
    st.image(uploaded_file, caption="Artifact Preview", use_container_width=True)

# 2. Input
artifact_name = st.text_input("💡 What is this artifact?", placeholder="e.g., Bronze Mask of Sanxingdui / 三星堆青铜面具")

# 3. Action
if artifact_name and st.button("Let it Speak 🗣️"):
    if not api_key:
        st.error("Please configure your API Key first!")
    else:
        with st.spinner("The artifact is waking up..."):
            # A. Generate Story
            story = get_artifact_story(artifact_name)
            
            # Display Story
            st.markdown("### 📜 The Artifact Says:")
            # Use a nice container for the text
            st.info(story)
            
            # B. Generate & Play Audio
            output_file = "artifact_voice.mp3"
            asyncio.run(generate_audio(story, output_file))
            
            if os.path.exists(output_file):
                st.audio(output_file)
                st.success("🎉 Voice generated successfully!")
