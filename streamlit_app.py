import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ✅ Page Config
st.set_page_config(page_title="Emoji Math Solver", page_icon="🧮", layout="centered")

# ✅ Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0f1117;
        color: white;
    }
    .stButton>button {
        color: #fff;
        background-color: #4CAF50;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        border: none;
    }
    footer {
        visibility: hidden;
    }
    .footer-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: center;
        color: white;
        padding: 10px;
    }
    .footer-container:hover:after {
        content: " Hassan Haseen & Sameen Muzaffar";
        color: #ff4b4b;
        display: block;
        font-size: 14px;
        margin-top: 4px;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Footer
st.markdown('<div class="footer-container">Created with ❤️ by Team CodeRunners</div>', unsafe_allow_html=True)

# ✅ Title + Intro
st.title("🧮 Emoji Math Solver")
st.subheader("Solve fun math problems written in emojis!")

# ✅ Load Model (with caching)
@st.cache_resource
def load_model():
    model_id = "hassanhaseen/tinyllama-emoji-math-merged"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

pipe = load_model()

# ✅ Input Area
user_input = st.text_input("🔢 Enter an Emoji Math Problem:", placeholder="e.g., 🔟 ➕ 5️⃣ = ❓")

# ✅ Generate Button
if st.button("🔍 Solve"):
    if user_input.strip() == "":
        st.warning("Please enter a math problem using emojis!")
    else:
        with st.spinner("Thinking... 🤔"):
            prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
            output = pipe(prompt, max_length=100, do_sample=True, top_p=0.9, temperature=0.5)
            answer = output[0]['generated_text'].split("<|assistant|>")[-1].strip()

        # ✅ Reveal Answer
        with st.expander("✅ Click to Reveal Answer"):
            st.success(f"🎉 **Answer:** {answer}")

        # ✅ Feedback
        st.markdown("---")
        st.markdown("### 🙌 How was the answer?")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.button("⭐ 1", key="1")
        with col2:
            st.button("⭐ 2", key="2")
        with col3:
            st.button("⭐ 3", key="3")
        with col4:
            st.button("⭐ 4", key="4")
        with col5:
            st.button("⭐ 5", key="5")
