import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="Creative Math Problem Solver", page_icon="🧮")

st.title("🧠 Creative Math Problem Solver")
st.caption("Generate creative solutions for emoji math problems using TinyLlama!")

# ✅ Cache the model and tokenizer loading for performance
@st.cache_resource(show_spinner="Loading model... (⏳)")
def load_model():
    # Model name (merged fine-tuned model)
    model_name = "hassanhaseen/tinyllama-emoji-math-merged"

    # ✅ Use base tokenizer (assuming no added tokens)
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    # ✅ Load the fine-tuned model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )

    # ✅ Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,   # Limit response length
        temperature=0.5,      # Creativity level (adjust if needed)
        top_p=0.95            # Nucleus sampling for diversity
    )
    return pipe

# ✅ Load model
pipe = load_model()
st.success("✅ Model Loaded Successfully!")

# ✅ User input for math problem
user_input = st.text_area("🔢 Enter an emoji math problem:", height=150)

if st.button("💡 Generate Solution"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a math problem!")
    else:
        # ✅ Format the prompt (matching fine-tuned training format)
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

        # ✅ Generate response
        with st.spinner("Generating creative solution..."):
            response = pipe(prompt)

        # ✅ Extract and display response
        answer = response[0]["generated_text"].split("<|assistant|>\n")[-1].strip()
        st.subheader("✨ Solution")
        st.write(answer)

st.caption("Made with ❤️ using TinyLlama")
