import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ Page Config
st.set_page_config(page_title="Creative Math Problem Solver", page_icon="🧮")
st.title("🧮 Creative Math Problem Solver")
st.markdown("Enter an **Emoji Math Problem** and get the solution!")

# ✅ Cache Model Loading
@st.cache_resource(show_spinner="Loading model... (This may take a while ⏳)")
def load_model():
    model_name = "hassanhaseen/tinyllama-emoji-math-merged"  # ✅ Your merged model repo

    # Load tokenizer and model (CPU-friendly)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",     # Use auto to prevent CPU conflicts
        device_map="auto"       # Runs on CPU in Streamlit Cloud
    )

    # Create a pipeline for generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.3,
        top_p=0.9
    )

    return pipe

# ✅ Load the model
pipe = load_model()
st.success("✅ Model Loaded Successfully!")

# ✅ User Input
user_input = st.text_input("🔢 Enter an Emoji Math Problem:", placeholder="Example: 🐱 + 🐱 = ?")

# ✅ Generate Answer
if st.button("Solve") and user_input:
    with st.spinner("Solving your emoji math problem..."):
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        result = pipe(prompt)[0]["generated_text"]

        # Post-process response (extract answer)
        answer = result.split("<|assistant|>\n")[-1].strip()

        st.markdown(f"### ✅ Answer:\n{answer}")
