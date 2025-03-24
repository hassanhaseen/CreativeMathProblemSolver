import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="Emoji Math Solver", page_icon="🧮", layout="centered")

st.title("🧠 Emoji Math Solver")
st.markdown("Let the AI solve your emoji-based math problems! Try something like `Problem: 🍎 + 🍎 + 🍎 = 15`")

# 🧠 Load model and tokenizer safely
with st.spinner("Loading model from Hugging Face 🤗... Please wait."):
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")  # ✅ FIXED tokenizer source
    model = AutoModelForCausalLM.from_pretrained("hassanhaseen/emoji-math-distilgpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = st.text_area("📥 Enter your Emoji Math Problem (must start with `Problem:`):", value="Problem: 🍎 + 🍎 + 🍎 = 15")

if st.button("🧠 Solve"):
    if not prompt.strip().startswith("Problem:"):
        st.error("Please make sure your input starts with `Problem:`")
    else:
        with st.spinner("Generating solution..."):
            result = generator(prompt, max_length=128, num_return_sequences=1, do_sample=True)
            full_output = result[0]["generated_text"]
            if "Solution:" in full_output:
                solution = full_output.split("Solution:")[-1].strip()
            else:
                solution = "Couldn't find a solution in the output."

            st.success("✅ Solution")
            st.code(solution, language="text")
