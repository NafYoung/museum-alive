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

# Initialize Vision Model (Moondream) - Cached to avoid reloading
@st.cache_resource
def load_vision_model():
    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer

vision_model, vision_tokenizer = load_vision_model()

async def generate_audio(text, output_file="output.mp3"):
    """Generate audio using Edge-TTS (Free)"""
    communicate = edge_tts.Communicate(text, "zh-CN-YunxiNeural")
    await communicate.save(output_file)

def analyze_image(image):
    """Use Moondream to describe the image"""
    enc_image = vision_model.encode_image(image)
    # Prompt Moondream to describe the artifact
    description = vision_model.answer_question(enc_image, "Describe this artifact in detail.", vision_tokenizer)
    return description

def get_artifact_story(artifact_description):
    """Ask DeepSeek to roleplay based on visual description"""
    prompt = f"""
    我给你看了一张文物的图片，它的特征是：{artifact_description}。
    
    请你根据这个描述，猜猜你可能是谁（如果特征很明显），或者就作为一个神秘的古物。
    
    请用第一人称（“我”）做一个自我介绍。
    
    要求：
    1. 既然是“让文物说话”，语气要符合你的身份。
    2. 不要只讲枯燥的数据，要讲你的感受。
    3. 篇幅控制在 150 字以内。
    4. 开头要吸引人。
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个博物馆里的文物，富有性格和情感。"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"哎呀，我看不清自己... ({str(e)})"

# Main Content
uploaded_file = st.file_uploader("📸 给他拍张照 (或上传图片)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="已上传文物", use_container_width=True)
    
    if st.button("让它说话 🗣️"):
        if not api_key:
            st.error("请先在 .env 文件中配置 DEEPSEEK_API_KEY")
        else:
            with st.spinner("正在观察这个文物 (AI 识图中)..."):
                # 0. Load Image
                image = Image.open(uploaded_file)
                
                # 1. Vision Analysis
                description = analyze_image(image)
                st.info(f"👀 我看到的：{description}")
                
                with st.spinner("正在唤醒沉睡的灵魂..."):
                    # 2. Generate Story
                    story = get_artifact_story(description)
                    st.markdown(f"### 📜 文物的自述")
                    st.write(story)
                    
                    # 3. Generate Audio
                    output_file = "artifact_voice.mp3"
                    asyncio.run(generate_audio(story, output_file))
                    
                    # 4. Play Audio
                    st.audio(output_file)
                    st.success("🎉 唤醒成功！")
