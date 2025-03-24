import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# 🎯 Setup page
st.set_page_config(page_title="Emoji Math Solver", page_icon="🧮", layout="centered")

# 🎨 Title & Instructions
st.title("🧠 Emoji Math Solver")
st.markdown("Let the AI solve your emoji-based math problems! Just type a problem like `Problem: 🍎 + 🍎 + 🍎 = 15` and hit solve.")

# 🔁 Load model and tokenizer (no cache to avoid cloud issues)
@st.spinner("Loading model from Hugging Face 🤗... Please wait."):
    tokenizer = AutoTokenizer.from_pretrained("hassanhaseen/emoji-math-distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("hassanhaseen/emoji-math-distilgpt2")
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# ✍️ Text input
prompt = st.text_area("📥 Enter your Emoji Math Problem (must start with `Problem:`):", value="Problem: 🍎 + 🍎 + 🍎 = 15")

# 🧠 Solve Button
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

            # ✅ Display result
            st.success("✅ Solution")
            st.code(solution, language="text")
