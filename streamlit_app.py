import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import time

# Page config
st.set_page_config(
    page_title="🧮 Emoji Math Solver",
    page_icon="🧠",
    layout="centered"
)

# Title
st.title("🧮 Emoji Math Solver")
st.caption("Powered by TinyLlama + Team CodeRunners")

# Sidebar Controls
st.sidebar.header("⚙️ Settings")
temperature = st.sidebar.slider("Creativity (Temperature)", 0.1, 1.0, 0.5)

# Spinner loader emoji
loader_emojis = ["🧐", "🤖", "🔧", "⚙️", "💡", "🔢"]

# Error ratings
error_ratings = [
    "99% accurate, 1% fun",
    "100% math wizardry",
    "80% genius, 20% emoji master",
    "Mathified with style 😎",
    "90% brainpower, 10% sass"
]

# ✅ Load Merged Model + Tokenizer
@st.cache_resource
def load_model():
    with st.spinner("Loading Quantized TinyLlama Model... " + random.choice(loader_emojis)):

        # Always load tokenizer from the BASE model repo
        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

        # Load YOUR quantized model (weights only)
        model = AutoModelForCausalLM.from_pretrained(
            "hassanhaseen/TinyLlama-EmojiMathSolver-Quantized",
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.bfloat16
        )

        return tokenizer, model





tokenizer, model = load_model()

# ✅ Generate Answer Function
def solve_emoji_math(problem):
    prompt = f"<|startoftext|>Problem: {problem} Solution:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_length=100,
        temperature=temperature,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        num_return_sequences=1
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Solution:" in decoded_output:
        solution = decoded_output.split("Solution:")[-1].strip()
    else:
        solution = "Oops! Couldn't figure it out 🤷‍♂️"

    return solution

# ✅ Text Input
st.subheader("🔍 Enter Your Emoji Math Problem")
user_input = st.text_area("Paste your emoji math equation here... (e.g., 🍎 + 🍎 + 🍎 = 12)")

# ✅ Solve Button
if st.button("🛠️ Solve It!"):
    if not user_input.strip():
        st.warning("Please enter a valid emoji equation!")
    else:
        with st.spinner("Crunching numbers " + random.choice(loader_emojis)):
            time.sleep(1)  # Small delay for UX
            result = solve_emoji_math(user_input)

        st.success("✅ Problem Solved!")
        st.markdown(f"**🧮 Error Rating:** {random.choice(error_ratings)}")

        # Reveal Solution
        with st.expander("Click to reveal the solution:"):
            st.info(result)

# ✅ Footer with Hover
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
            content: "Hassan Haseen & Sameen Muzaffar";
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
