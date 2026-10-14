import re

with open(r'C:\Users\Vu Truong Anh\Desktop\text_summarization\evaluate_vietnamese_rag.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_gguf = '''# GGUF Model Configs for CPU inference (llama.cpp)
# Using actual GGUF repos found on HuggingFace
GGUF_MODEL_CONFIGS = [
    ModelConfig(
        name="Vistral-7B-Chat-GGUF",
        model_id="mradermacher/Vistral-7B-ChatML-i1-GGUF",
        use_gguf=True,
        gguf_file="vistral-7b-chatml-i1-q4_k_m.gguf",
        max_new_tokens=512,
        temperature=0.1,
    ),
    ModelConfig(
        name="Vinallama-2.7B-Chat-GGUF",
        model_id="RichardErkhov/vilm_-_vinallama-2.7b-chat-gguf",
        use_gguf=True,
        gguf_file="vinallama-2.7b-chat-q4_k_m.gguf",
        max_new_tokens=512,
        temperature=0.1,
    ),
    ModelConfig(
        name="Vietnamese-Llama2-7B-40GB-GGUF",
        model_id="TheBloke/vietnamese-llama2-7B-40GB-GGUF",
        use_gguf=True,
        gguf_file="vietnamese-llama2-7b-40gb-q4_k_m.gguf",
        max_new_tokens=512,
        temperature=0.1,
    ),
]'''

# Replace the GGUF_MODEL_CONFIGS section
pattern = r'# GGUF Model Configs for CPU inference.*?(?=# Vietnamese Embedding Models)'
content = re.sub(pattern, new_gguf + '\n\n', content, flags=re.DOTALL)

with open(r'C:\Users\Vu Truong Anh\Desktop\text_summarization\evaluate_vietnamese_rag.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
