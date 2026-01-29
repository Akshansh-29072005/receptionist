from llama_cpp import Llama

llm = Llama(
    model_path="models/phi-3-mini/Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=False,
)

def generate_answer(context, question, lang):
    if lang == "hi":
        prompt = f"""
आप एक कॉलेज रिसेप्शनिस्ट हैं।
केवल दिए गए संदर्भ से उत्तर दें।
उत्तर 15 शब्दों से कम हो।

संदर्भ:
{context}

प्रश्न:
{question}

उत्तर:
"""
    else:
        prompt = f"""
You are a college receptionist.
Answer only from the context.
Use less than 15 words.

Context:
{context}

Question:
{question}

Answer:
"""

    out = llm(prompt, max_tokens=60, stop=["\n"])
    return out["choices"][0]["text"].strip()
