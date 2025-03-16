import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# ✅ Page Configuration
st.set_page_config(page_title="Creative Math Problem Solver 🤖➕", page_icon="🧮", layout="centered")

# ✅ Title and Description
st.title("🤖 Creative Math Problem Solver")
st.subheader("Enter an Emoji-based Math Problem and get your solution instantly!")

# ✅ Loader Spinner while model loads
with st.spinner('Loading model... (first time may take a few seconds)'):
    @st.cache_resource(show_spinner=False)
    def load_model():
        model_name = "hassanhaseen/tinyllama-emoji-math-merged"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe

    pipe = load_model()

st.success('Model Loaded!')

# ✅ Input Area
user_input = st.text_input("🔢 Enter an Emoji Math Problem:", placeholder="e.g. 🍎🍎 + 🍎 = ❓")

if user_input:
    with st.spinner('Generating answer...'):
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)

        # Extract generated text after <|assistant|> token
        raw_output = output[0]['generated_text']
        response = raw_output.split("<|assistant|>")[-1].strip()

        st.markdown("### ✅ Problem")
        st.info(user_input)

        # Reveal Answer Box
        with st.expander("📝 Click to reveal the answer"):
            st.success(response)

# ✅ Footer with hover effect
st.markdown("""
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            font-size: 14px;
            color: #888;
        }
        .footer span:hover {
            color: #f63366;
        }
    </style>
    <div class="footer">
        Created with ❤️ by <span title='Hassan Haseen & Sameen Muzaffar'>Team CodeRunners</span>
    </div>
""", unsafe_allow_html=True)
