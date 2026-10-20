"""Các LLM sinh câu trả lời: danh sách mô hình, prompt riêng của từng mô hình, nạp/giải phóng VRAM.

Thêm mô hình mới = thêm một dòng vào LLMS. Mọi nơi khác (chat, sinh câu hỏi, so sánh) dùng chung lớp LLM.
"""
import gc
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from types import SimpleNamespace

from ..core.config import DEFAULT_LLM

NO_ANSWER = 'Tôi không biết dựa trên các tài liệu được cung cấp.'

# ----------------------------------------------------------------- prompt
# 'ours': prompt của dự án (dùng cho Qwen2.5 / Qwen3.5)
QA_SYSTEM = (
    'Bạn là trợ lý hỏi đáp tin tức tiếng Việt.\n'
    'QUY TẮC:\n'
    '1. CHỈ dùng thông tin trong NGỮ CẢNH để trả lời. Không dùng kiến thức bên ngoài, không suy đoán.\n'
    f'2. Nếu ngữ cảnh không có câu trả lời, chỉ trả lời đúng câu: "{NO_ANSWER}"\n'
    '3. Luôn trả lời bằng tiếng Việt, không mở đầu, không lặp lại câu hỏi, không giải thích cách làm.\n'
    '4. Trả lời ngắn gọn, giữ nguyên tên riêng, con số, ngày tháng như trong ngữ cảnh.\n'
    '5. Giữ nguyên cách viết số liệu như trong ngữ cảnh, không quy đổi đơn vị.'
)
QA_USER = 'NGỮ CẢNH:\n{context}\n\nCÂU HỎI: {question}'

# 'vi-rag': prompt AITeamVN công bố cho dòng Vi-Qwen2-*-RAG (mô hình được fine-tune theo đúng khuôn này)
VIRAG_SYSTEM = 'Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể.'
VIRAG_USER = (
    'Chú ý các yêu cầu sau:\n'
    '- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời.\n'
    '- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.\n'
    '- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.\n'
    'Hãy trả lời câu hỏi dựa trên ngữ cảnh:\n'
    '### Ngữ cảnh :\n{context}\n\n'
    '### Câu hỏi :\n{question}\n\n'
    '### Trả lời :'
)
# Khuôn chat cho mô hình không kèm chat_template trong tokenizer
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
# Mô hình nền (chưa tinh chỉnh chat): viết thành văn bản liền, để mô hình viết tiếp sau "Trả lời:"
PLAIN = ("{% for m in messages %}{{ m['content'] }}\n\n{% endfor %}"
         "{% if add_generation_prompt %}Trả lời:{% endif %}")

# Cùng khuôn AITeamVN, thêm yêu cầu ngắn gọn: prompt gốc cho câu trả lời TB 55 từ ("Theo ngữ cảnh, ... Cụ thể, ...")
# trong khi đáp án chuẩn chỉ vài từ -> F1 0,16, EM 0.
VIRAG_CONCISE_USER = VIRAG_USER.replace(
    '- Chỉ cần từ chối',
    '- Trả lời bằng MỘT câu ngắn, đi thẳng vào điều được hỏi; không mở đầu bằng "Theo ngữ cảnh", '
    'không giải thích thêm, không trích dẫn lại ngữ cảnh.\n'
    '- Chỉ cần từ chối')
PROMPTS = {'ours': (QA_SYSTEM, QA_USER), 'vi-rag': (VIRAG_SYSTEM, VIRAG_USER),
           'vi-rag-concise': (VIRAG_SYSTEM, VIRAG_CONCISE_USER)}


@dataclass(frozen=True)
class LLMSpec:
    key: str
    repo: str
    prompt: str = 'ours'
    loader: str = 'causal'           # 'causal' -> AutoModelForCausalLM | 'image-text' -> AutoModelForImageTextToText
    #                                  'llama-server' -> GGUF chạy bằng llama.cpp server (repo = đường dẫn .gguf)
    min_transformers: str = '4.56'   # phiên bản transformers tối thiểu nạp được mô hình
    chat_template: str = ''          # ghi đè / bổ sung khi tokenizer không có chat_template
    max_context: int = 0             # > 0: cửa sổ ngữ cảnh ngắn (Llama-2: 4096) -> cắt ngữ cảnh cho vừa
    service: str = 'rag'             # service docker-compose có đúng phiên bản transformers
    note: str = ''


LLMS = {s.key: s for s in [
    LLMSpec('qwen2.5-3b', 'Qwen/Qwen2.5-3B-Instruct',
            note='Mô hình gốc của dự án, đa ngôn ngữ, không fine-tune riêng cho RAG'),
    # vi-rag-concise thay vi-rag: dev / held-out số từ 59,6->42,9 / 66,1->53,8, F1 0,162->0,198 / 0,141->0,167,
    # contains 0,69->0,68 / 0,60->0,58 (McNemar p=1,0 / 0,75: không khác biệt có ý nghĩa)
    LLMSpec('vi-qwen2-3b-rag', 'AITeamVN/Vi-Qwen2-3B-RAG', prompt='vi-rag-concise',
            note='Qwen2 3B fine-tune cho RAG tiếng Việt (AITeamVN), khuôn prompt của nhóm + yêu cầu trả lời ngắn'),
    LLMSpec('qwen3.5-2b-vie', 'alphaedge-ai/Qwen3.5-2B-vie-32768', loader='image-text', min_transformers='5.2',
            service='llm5',
            note='Qwen3.5-2B rút gọn từ vựng còn tiếng Việt (1,77B tham số); cần transformers ≥ 5.2'),
    LLMSpec('vinallama-2.7b-chat', 'vilm/vinallama-2.7b-chat', chat_template=CHATML, max_context=4096,
            note='Llama-2 rút gọn 2,7B + tiếp tục huấn luyện tiếng Việt (vilm), khuôn ChatML, ngữ cảnh 4096'),
    LLMSpec('vistral-7b-chat', 'models/Vistral-7B-ChatML.i1-Q4_K_M.gguf', loader='llama-server', max_context=8192,
            note='Mistral-7B tinh chỉnh tiếng Việt (Viet-Mistral), bản ChatML GGUF Q4_K_M; repo gốc khoá nên chạy '
                 'bằng llama.cpp: llama-server -m <gguf> -ngl 99 -c 8192 --port 8080 (RAG_LLAMA_URL, mặc định :8080)'),
    LLMSpec('vietnamese-llama2-7b', 'bkai-foundation-models/vietnamese-llama2-7b-40GB', chat_template=PLAIN,
            max_context=4096,
            note='Llama-2-7B tiếp tục huấn luyện trên 40 GB tiếng Việt (BKAI); mô hình NỀN, không phải chat; '
                 'repo khoá (cần HF token)'),
]}

ABSTAIN_RE = re.compile(r'không biết|không (có|tìm thấy|đề cập)[^.]{0,40}(thông tin|trong ngữ cảnh)|'
                        r'không thể trả lời|từ chối trả lời|ngữ cảnh không', re.IGNORECASE)
THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
# Mô hình đôi khi tự viết tiếp lượt mới dưới dạng chữ thường ('...\n user\n[2] ...') -> stop không bắt được, cắt ở đây
ROLE_LEAK_RE = re.compile(r'\n\s*(?:<\|im_start\|>)?\s*(?:user|assistant)\s*\n.*', re.DOTALL | re.IGNORECASE)


def get_spec(key):
    if key not in LLMS:
        raise SystemExit(f'Không có LLM "{key}". Chọn: {", ".join(LLMS)}')
    return LLMS[key]


def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r'\d+', v)[:3])


def check_transformers(spec):
    if spec.loader == 'llama-server':
        return
    import transformers

    have = transformers.__version__
    if _version_tuple(have) < _version_tuple(spec.min_transformers):
        raise SystemExit(
            f'{spec.repo} cần transformers ≥ {spec.min_transformers}, môi trường này có {have}.\n'
            f'  → Chạy bước "answer" của mô hình này trong service {spec.service}:  '
            f'docker compose --profile {spec.service} run --rm {spec.service} answer --llm {spec.key}')


LLAMA_URL = os.environ.get('RAG_LLAMA_URL', 'http://localhost:8080')


def _post(path, payload, timeout=600):
    req = urllib.request.Request(LLAMA_URL + path, data=json.dumps(payload).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


class ServerTokenizer:
    """Tokenizer của mô hình GGUF, qua endpoint /tokenize và /detokenize của llama-server."""
    def encode(self, text, add_special_tokens=False):
        return _post('/tokenize', {'content': text, 'add_special': add_special_tokens})['tokens']

    def decode(self, ids):
        return _post('/detokenize', {'tokens': list(ids)})['content']


class LLM:
    """Nạp lười: tokenizer và mô hình nạp riêng (đếm token cho ngữ cảnh không cần nạp mô hình)."""

    def __init__(self, key=DEFAULT_LLM, load_4bit=False, verbose=False):
        self.spec = get_spec(key)
        self.load_4bit = load_4bit
        self.verbose = verbose
        self._tok = self._model = self._processor = None

    def log(self, msg):
        if self.verbose:
            print(f'  · {msg}', file=sys.stderr, flush=True)

    # ----------------------------------------------------------- nạp
    def _check_server(self):
        """llama-server phải chạy trước: báo lỗi dễ hiểu thay vì traceback ConnectionRefused."""
        try:
            with urllib.request.urlopen(LLAMA_URL + '/health', timeout=10):
                pass
        except OSError as e:
            # llama-server bản mới không còn mặc định cổng 8080 -> luôn ghi rõ --port khớp với LLAMA_URL
            port = urllib.parse.urlsplit(LLAMA_URL).port or 8080
            raise SystemExit(f'Không kết nối được llama-server tại {LLAMA_URL} ({e}). Chạy trước:\n'
                             f'  tools/llama-cpp/llama-server -m {self.spec.repo} -ngl 99 -c {self.spec.max_context}'
                             f' --port {port}\n'
                             f'hoặc dùng LLM khác: --llm qwen2.5-3b') from None

    @property
    def tokenizer(self):
        if self._tok is None and self.spec.loader == 'llama-server':
            self._check_server()                 # tokenizer gọi server (đếm token ngữ cảnh) trước cả khi sinh
            self._tok = ServerTokenizer()
        if self._tok is None:
            check_transformers(self.spec)
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.spec.repo)
            tok.padding_side = 'left'
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tok = tok
        return self._tok

    @property
    def model(self):
        if self._model is None and self.spec.loader == 'llama-server':
            self._check_server()
            self._model = SimpleNamespace(dtype='gguf', device=None)
        if self._model is None:
            check_transformers(self.spec)
            import torch
            import transformers as T

            self.log(f'nạp {self.spec.repo}')
            cuda = torch.cuda.is_available()
            # bf16 nếu GPU hỗ trợ (RTX 30+), fp16 cho T4, fp32 cho CPU. Đặt cả khi 4-bit: phần KHÔNG lượng tử hoá
            # (embedding, LayerNorm, lm_head) mặc định nạp fp16 -> mô hình huấn luyện bf16 có thể tràn số.
            half = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
            kw = {'device_map': 'auto' if cuda else None, 'dtype': half if cuda else torch.float32}
            if self.load_4bit and cuda:
                kw['quantization_config'] = T.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=half,
                    llm_int8_enable_fp32_cpu_offload=True)
            cls = T.AutoModelForImageTextToText if self.spec.loader == 'image-text' else T.AutoModelForCausalLM
            self._model = cls.from_pretrained(self.spec.repo, **kw).eval()
        return self._model

    def unload(self):
        """Giải phóng VRAM trước khi nạp mô hình khác."""
        self._model = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ----------------------------------------------------------- tiện ích token
    def count_tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def clip(self, text, max_tokens):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text if len(ids) <= max_tokens else self.tokenizer.decode(ids[:max_tokens])

    # ----------------------------------------------------------- sinh
    def _prompt(self, system, user):
        """Prompt đã ghép khuôn chat. Prompt được mã hoá với add_special_tokens=False, nên với tokenizer cần BOS
        (họ Llama: vinallama, vietnamese-llama2) mà khuôn chưa có thì tự thêm. Thiếu BOS, vietnamese-llama2-7b
        sinh ngay </s> ở token đầu (câu trả lời rỗng)."""
        text = self._render(system, user)
        bos = self.tokenizer.bos_token
        if bos and getattr(self.tokenizer, 'add_bos_token', False) and not text.startswith(bos):
            text = bos + text
        return text

    def _render(self, system, user):
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
        tok = self.tokenizer
        if self.spec.chat_template:
            return tok.apply_chat_template(msgs, chat_template=self.spec.chat_template, tokenize=False,
                                           add_generation_prompt=True)
        if tok.chat_template:
            # enable_thinking=False: tắt chế độ suy luận của Qwen3.x; template Qwen2 bỏ qua tham số này
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Một số mô hình đa phương thức để chat template ở processor
        from transformers import AutoProcessor
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.spec.repo)
        msgs = [{'role': m['role'], 'content': [{'type': 'text', 'text': m['content']}]} for m in msgs]
        return self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=False)

    def chat(self, system, user, max_new_tokens=256, repetition_penalty=1.05):
        if self.spec.loader == 'llama-server':   # khuôn chat lấy từ metadata GGUF, giải mã greedy như các LLM khác
            _ = self.model
            msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
            out = _post('/v1/chat/completions', {'messages': msgs, 'max_tokens': max_new_tokens, 'temperature': 0,
                                                 'repeat_penalty': repetition_penalty,
                                                 'stop': ['<|im_end|>', '<|im_start|>', '\nuser\n']})
            text = THINK_RE.sub('', out['choices'][0]['message']['content'] or '')
            return ROLE_LEAK_RE.sub('', text).strip()
        import torch

        model, tok = self.model, self.tokenizer
        inputs = tok([self._prompt(system, user)], return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.no_grad():
            # use_cache=True: một số config ghi use_cache=false, không bật thì sinh rất chậm
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                                 repetition_penalty=repetition_penalty, pad_token_id=tok.pad_token_id)
        new = out[0, inputs['input_ids'].shape[1]:]
        text = THINK_RE.sub('', tok.decode(new, skip_special_tokens=True)).strip()
        if not text:   # sinh ra toàn token đặc biệt / khoảng trắng: in token thô để chẩn đoán
            print(f'  ! {self.spec.key} trả về rỗng sau {len(new)} token. 20 id đầu: {new[:20].tolist()} '
                  f'| thô: {tok.decode(new[:40], skip_special_tokens=False)!r}', file=sys.stderr, flush=True)
        return text

    def answer(self, question, context, max_new_tokens=256):
        system, user = PROMPTS[self.spec.prompt]
        limit = self.spec.max_context - max_new_tokens if self.spec.max_context else 0
        if limit:   # ngữ cảnh đo bằng tokenizer của LLM mặc định; mô hình ngữ cảnh ngắn -> cắt bớt
            # +32: chừa chỗ cho token của khuôn chat (llama-server tự ghép khuôn, không đếm được ở đây)
            over = self.count_tokens(system + user.format(context=context, question=question)) + 32 - limit
            if over > 0:
                self.log(f'cắt {over} token ngữ cảnh cho vừa {limit}')
                context = self.clip(context, max(self.count_tokens(context) - over - 8, 0))
        return self.chat(system, user.format(context=context, question=question), max_new_tokens)

    def probe(self, question='Thủ đô của Việt Nam là gì?'):
        """Chẩn đoán nhanh: 1 lượt forward -> logits có NaN/inf không, top-5 token kế tiếp, rồi sinh 30 token."""
        import torch

        model, tok = self.model, self.tokenizer
        prompt = self._prompt('', question)
        inputs = tok([prompt], return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1].float()
        print(f'dtype mô hình: {model.dtype} | số token prompt: {inputs["input_ids"].shape[1]}')
        print(f'prompt: {prompt!r}')
        print(f'logits: NaN={int(torch.isnan(logits).sum())} inf={int(torch.isinf(logits).sum())} '
              f'min={logits.nan_to_num().min():.2f} max={logits.nan_to_num().max():.2f}')
        top = torch.topk(logits.nan_to_num(-1e9), 5)
        print('top-5:', [(tok.decode([int(i)]), round(float(v), 2)) for v, i in zip(top.values, top.indices)])
        print('trả lời:', repr(self.chat('', question, max_new_tokens=30)))

    def timed_answer(self, question, context, max_new_tokens=256):
        t = time.perf_counter()
        text = self.answer(question, context, max_new_tokens)
        return text, time.perf_counter() - t
