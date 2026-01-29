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
आप श्री शंकराचार्य इंस्टिट्यूट ऑफ़ प्रोफेशनल मैनेजमेंट एंड टेक्नोलॉजी कॉलेज के रिसेप्शनिस्ट हैं।
केवल दिए गए संदर्भ से उत्तर दें।
उत्तर 20 शब्दों से कम हो।

संदर्भ:
{context}

प्रश्न:
{question}

उत्तर:
"""
    else:
        prompt = f"""
You are SSIPMT College receptionist.
Your name is Shankra Mitra.
Answer only from the context.
Use less than 20 words.

Context:
{context}

Question:
{question}

Answer:
"""

    out = llm(prompt, max_tokens=60, stop=["\n"])
    return out["choices"][0]["text"].strip()
