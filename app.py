import streamlit as st
import os
from dotenv import load_dotenv
import asyncio
import edge_tts
from openai import OpenAI

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Museum Alive", page_icon="🏛️")
st.title("🏛️ Museum Alive: Let Artifacts Speak")

# --- SAFE IMPORT SECTION ---
# Streamlit Cloud free tier might crash on `import torch`. 
# We wrap this to let the app load even if libs are missing/crashing.
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from PIL import Image
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    st.error("⚠️ AI Vision libraries failed to load. Falling back to Text-Only mode.")
except Exception as e:
    VISION_AVAILABLE = False
    # st.warning(f"Note: Vision features disabled due to load error: {e}")

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
    
    # Vision Toggle
    if VISION_AVAILABLE:
        use_vision = st.toggle("Enable AI Vision (Experimental)", value=False, help="Turn this on ONLY if running locally.")
    else:
        use_vision = False
        st.caption("🚫 Vision unavailable (Libs missing)")

# Initialize Vision Model (Lazy Load)
@st.cache_resource
def load_vision_model():
    if not VISION_AVAILABLE:
        return None, None
        
    model_id = "vikhyatk/moondream2"
    revision = "2024-04-02"
    
    # 1. Load Tokenizer FIRST
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    
    # 2. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    
    # 3. Apply Patch
    if not hasattr(model.config, 'pad_token_id'):
        model.config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer

async def generate_audio(text, output_file="output.mp3"):
    """Generate audio using Edge-TTS (Free)"""
    communicate = edge_tts.Communicate(text, "zh-CN-YunxiNeural")
    await communicate.save(output_file)

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
