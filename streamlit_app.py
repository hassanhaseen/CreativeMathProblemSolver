import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time

# ✅ Page Config
st.set_page_config(
    page_title="Emoji Math Solver 🧠",
    page_icon="🧮",
    layout="centered"
)

# ✅ Sidebar Settings
st.sidebar.header("⚙️ App Settings")
temperature = st.sidebar.slider("Creativity (Temperature)", 0.3, 1.0, 0.8)

# ✅ Title & Subtitle
st.title("🧠 Emoji Math Solver")
st.caption("AI-powered fun for emoji-based math riddles! 🍎➕🍕➗🚗")

# ✅ Load model/tokenizer with spinner
@st.cache_resource
def load_model():
    with st.spinner("Loading your emoji genius 🤓..."):
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model = AutoModelForCausalLM.from_pretrained("hassanhaseen/emoji-math-distilgpt2")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

generator = load_model()

# ✅ User Input Section
st.subheader("🧩 Enter Your Emoji Math Riddle")
user_input = st.text_area(
    "Start with `Problem:` (Example: `Problem: 🍎 + 🍎 + 🍎 = 15`)",
    value="Problem: 🍎 + 🍎 + 🍎 = 15",
    height=140
)

# ✅ Generate Button
if st.button("🧮 Solve It!"):
    if not user_input.strip().lower().startswith("problem:"):
        st.warning("⚠️ Your input must start with `Problem:`")
    else:
        with st.spinner("Solving your riddle..."):
            result = generator(
                user_input,
                max_length=128,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=temperature,
                truncation=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            full_output = result[0]["generated_text"]
            solution = full_output.split("Solution:")[-1].strip() if "Solution:" in full_output else "⚠️ Couldn't generate a solution."

        time.sleep(1)  # just for effect
        st.success("✅ Here's your solution:")
        st.code(solution)

# ✅ Footer
st.markdown("---")
st.markdown(
    """
    <style>
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888888;
        }
        .footer span {
            position: relative;
            cursor: pointer;
            color: #FF4B4B;
        }
        .footer span::after {
            content: "Hassan Haseen";
            position: absolute;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            background-color: #333;
            color: #fff;
            padding: 5px 10px;
            border-radius: 8px;
            white-space: nowrap;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
            font-size: 12px;
        }
        .footer span:hover::after {
            opacity: 1;
        }
    </style>

    <div class='footer'>
        Created with ❤️ by <span>Team CodeRunners</span>
    </div>
    """,
    unsafe_allow_html=True
)
