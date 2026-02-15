import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Museum Alive", page_icon="🏛️")

# Title and Description
st.title("🏛️ Museum Alive: Let Artifacts Speak")
st.write("Upload a photo of an artifact, and AI will bring it to life.")

# Sidebar for Settings
with st.sidebar:
    st.header("Settings")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or "your-key-here" in api_key:
        st.warning("⚠️ Please set your DEEPSEEK_API_KEY in the .env file.")
    else:
        st.success("✅ API Key Loaded")

import asyncio
import edge_tts
from openai import OpenAI

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Initialize DeepSeek Client
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Sidebar Settings
with st.sidebar:
    st.header("Settings")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        st.warning("⚠️ Please set DEEPSEEK_API_KEY in .env")
    else:
        st.success("✅ API Key Loaded")
    
    # Vision Toggle (Default OFF to save Cloud resources)
    use_vision = st.toggle("Enable AI Vision (Experimental)", value=False, help="Turn this on ONLY if running locally. Streamlit Cloud may crash due to memory limits.")

# Initialize Vision Model (Lazy Load)
@st.cache_resource
def load_vision_model():
    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

# Main Content
st.write("Upload a photo of an artifact, and AI will bring it to life.")
uploaded_file = st.file_uploader("📸 给他拍张照 (或上传图片)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="已上传文物", use_container_width=True)
    
    artifact_description = ""
    
    # Logic Branch: Vision vs Manual
    if use_vision:
        if st.button("👁️ AI 识图并说话"):
            with st.spinner("正在加载视觉模型 (第一次可能很慢)..."):
                try:
                    vision_model, vision_tokenizer = load_vision_model()
                    image = Image.open(uploaded_file)
                    with st.spinner("AI 正在观察文物..."):
                        enc_image = vision_model.encode_image(image)
                        artifact_description = vision_model.answer_question(enc_image, "Describe this artifact in detail.", vision_tokenizer)
                        st.info(f"👀 AI 看到的：{artifact_description}")
                except Exception as e:
                    st.error(f"视觉模型加载失败 (可能是内存不足): {e}")
                    st.stop()
    else:
        artifact_description = st.text_input("💡 (省流版) 告诉我它的名字/特征：", placeholder="比如：三星堆青铜面具")
        if artifact_description and st.button("让它说话 🗣️"):
             pass # Trigger next block

    # Generate Story & Audio (Common Logic)
    if artifact_description:
         with st.spinner("正在唤醒沉睡的灵魂..."):
            # 2. Generate Story
            story = get_artifact_story(artifact_description)
            st.markdown(f"### 📜 文物的自述")
            st.write(story)
            
            # 3. Generate Audio
            output_file = "artifact_voice.mp3"
            asyncio.run(generate_audio(story, output_file))
            
            # 4. Play Audio
            st.audio(output_file)
            if not use_vision:
                st.success("🎉 唤醒成功！(这是省流版，本地开启 Vision 可体验全自动)")
