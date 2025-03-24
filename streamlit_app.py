import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ✅ Streamlit Page Config
st.set_page_config(page_title="Emoji Math Solver", page_icon="🧠", layout="centered")

# ✅ Custom Style (no config.toml)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        background-color: #fefefe;
    }
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #f8f8f2 !important;
        border: 1px solid #f63366;
        font-size: 18px !important;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #f63366;
        color: white;
        font-size: 18px;
        border-radius: 8px;
        padding: 10px 24px;
        margin-top: 12px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #e10050;
        color: #ffffff;
        transform: scale(1.02);
    }
    .title-style {
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.25rem;
    }
    .subtitle-style {
        text-align: center;
        font-size: 1.2rem;
        color: #444;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 Title
st.markdown('<div class="title-style">🧠 Emoji Math Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">AI-powered fun for emoji-based math riddles! 🤖➕🍎➗🚗</div>', unsafe_allow_html=True)

# 📦 Load model
with st.spinner("🔄 Loading model from Hugging Face... please wait"):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("hassanhaseen/emoji-math-distilgpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 📝 Input Box
prompt = st.text_area("🔢 Your Emoji Math Problem (start with `Problem:`)", value="Problem: 🍎 + 🍎 + 🍎 = 15", height=140)

# 🧮 Solve Button
if st.button("🧠 Solve My Emoji Problem"):
    if not prompt.strip().lower().startswith("problem:"):
        st.error("⚠️ Make sure your input starts with `Problem:`")
    else:
        with st.spinner("🧠 Solving..."):
            result = generator(
                prompt,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                truncation=True,
                pad_token_id=tokenizer.eos_token_id
            )
            output = result[0]["generated_text"]
            if "Solution:" in output:
                solution = output.split("Solution:")[-1].strip()
            else:
                solution = "⚠️ Couldn't generate a valid solution."

        st.success("✅ Here's your solution!")
        st.code(solution)
