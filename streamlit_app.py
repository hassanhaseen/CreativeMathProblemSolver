# ✅ Streamlit App for Emoji Math Solver
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ✅ Page Config
st.set_page_config(page_title="🤖 Emoji Math Solver", page_icon="🧠")

# ✅ Header
st.markdown("<h1 style='text-align: center;'>🤖 Emoji Math Solver</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Solve emoji-based math equations with AI!</p>", unsafe_allow_html=True)

# ✅ Sidebar Footer
with st.sidebar:
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>Created with ❤️ by <span title='Hassan Haseen & Sameen Muzaffar'>Team CodeRunners</span></p>",
        unsafe_allow_html=True
    )

# ✅ Load Model and Tokenizer (cached for speed)
@st.cache_resource
def load_model():
    model_id = "hassanhaseen/deepseek-coder-emoji-math-lora"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

pipe = load_model()

# ✅ Input Area
user_input = st.text_input("🔢 Enter an Emoji Math Problem:", placeholder="🍎 + 🍎 + 🍎 = 12")

# ✅ Spinner and Generate Button
if st.button("🔍 Solve"):
    if user_input:
        with st.spinner("Thinking hard... 🤔"):
            prompt = f"<|system|>You are an expert emoji math solver.<|user|>Solve this: {user_input}<|assistant|>"
            response = pipe(prompt, max_length=256, do_sample=True, temperature=0.3)[0]["generated_text"]
            
            # Extract response
            if "<|assistant|>" in response:
                answer = response.split("<|assistant|>")[-1].strip()
            else:
                answer = response.strip()

            # ✅ Display Reveal-Answer Style
            with st.expander("📖 Click to Reveal Answer"):
                st.success(answer)
    else:
        st.warning("Please enter a math problem first!")

# ✅ Styling (Optional CSS tweaks)
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)
