import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 🌈 Streamlit Page Config
st.set_page_config(
    page_title="Emoji Math Solver",
    page_icon="🧠",
    layout="centered"
)

# 🎨 Custom CSS Styling
st.markdown("""
    <style>
    .main {
        background-color: #fdfdfd;
    }
    .stTextArea textarea {
        font-size: 18px;
        line-height: 1.6;
    }
    .stButton button {
        background-color: #F63366;
        color: white;
        font-size: 18px;
        padding: 0.6em 1.2em;
        border-radius: 8px;
        margin-top: 10px;
    }
    .stButton button:hover {
        background-color: #e10050;
        color: white;
    }
    .emoji-title {
        font-size: 40px;
        text-align: center;
        margin-bottom: 0.2em;
    }
    .emoji-subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# 🧠 App Header
st.markdown('<div class="emoji-title">🧠 Emoji Math Solver</div>', unsafe_allow_html=True)
st.markdown('<div class="emoji-subtitle">AI-powered fun for emoji-based math riddles! 🤖➕🍎➗🚗</div>', unsafe_allow_html=True)

# 📦 Load model + tokenizer
with st.spinner("🔄 Loading model from Hugging Face... please wait"):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("hassanhaseen/emoji-math-distilgpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✍️ User Input
prompt = st.text_area(
    "📝 Enter your Emoji Math Problem below (must start with `Problem:`):",
    value="Problem: 🍎 + 🍎 + 🍎 = 15",
    height=150
)

# 🧠 Solve Button
if st.button("🧮 Solve My Emoji Problem"):
    if not prompt.strip().lower().startswith("problem:"):
        st.error("⚠️ Please make sure your input starts with `Problem:`")
    else:
        with st.spinner("🔍 Thinking... solving your problem!"):
            result = generator(
                prompt,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                truncation=True,
                pad_token_id=tokenizer.eos_token_id
            )
            full_output = result[0]["generated_text"]
            if "Solution:" in full_output:
                solution = full_output.split("Solution:")[-1].strip()
            else:
                solution = "⚠️ Hmm... I couldn’t detect a solution. Try a different input?"

        # ✅ Display Result
        st.success("✅ Solution Found!")
        st.markdown(f"### 🧾 `{solution}`")

# 👣 Footer
st.markdown("""
---
Made with ❤️ by [Hassan Haseen](https://github.com/hassanhaseen)  
Built using 🤗 Transformers and deployed via 🚀 Streamlit Cloud
""")
