"""Các LLM sinh câu trả lời: danh sách mô hình, prompt riêng của từng mô hình, nạp/giải phóng VRAM.

Thêm mô hình mới = thêm một dòng vào LLMS. Mọi nơi khác (chat, sinh câu hỏi, so sánh) dùng chung lớp LLM.
"""
import gc
import re
import sys
import time
from dataclasses import dataclass

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
# 'phogpt': chat template của PhoGPT chỉ có lượt người dùng ("### Câu hỏi: ... ### Trả lời:"), không có vai system
# -> đưa quy tắc vào chung lượt người dùng, theo khuôn hỏi đáp trên văn bản mà VinAI dùng làm ví dụ.
PHOGPT_USER = (
    'Dựa vào văn bản sau đây:\n{context}\n\n'
    'Hãy trả lời câu hỏi: {question}\n'
    'Chỉ dùng thông tin trong văn bản, trả lời ngắn gọn, giữ nguyên tên riêng, con số, ngày tháng. '
    f'Nếu văn bản không có câu trả lời, chỉ trả lời: "{NO_ANSWER}"'
)
PROMPTS = {'ours': (QA_SYSTEM, QA_USER), 'vi-rag': (VIRAG_SYSTEM, VIRAG_USER), 'phogpt': ('', PHOGPT_USER)}


@dataclass(frozen=True)
class LLMSpec:
    key: str
    repo: str
    prompt: str = 'ours'
    loader: str = 'causal'           # 'causal' -> AutoModelForCausalLM | 'image-text' -> AutoModelForImageTextToText
    min_transformers: str = '4.56'   # khoảng transformers nạp được mô hình: [min, max)
    max_transformers: str = ''
    trust_remote_code: bool = False  # mô hình tự mang code kiến trúc (PhoGPT: MPT của MosaicML)
    service: str = 'rag'             # service docker-compose có đúng phiên bản transformers
    note: str = ''


LLMS = {s.key: s for s in [
    LLMSpec('qwen2.5-3b', 'Qwen/Qwen2.5-3B-Instruct',
            note='Mô hình gốc của dự án, đa ngôn ngữ, không fine-tune riêng cho RAG'),
    LLMSpec('vi-qwen2-3b-rag', 'AITeamVN/Vi-Qwen2-3B-RAG', prompt='vi-rag',
            note='Qwen2 3B fine-tune cho RAG tiếng Việt (AITeamVN), dùng prompt gốc của nhóm'),
    LLMSpec('qwen3.5-2b-vie', 'alphaedge-ai/Qwen3.5-2B-vie-32768', loader='image-text', min_transformers='5.2',
            service='llm5',
            note='Qwen3.5-2B rút gọn từ vựng còn tiếng Việt (1,77B tham số); cần transformers ≥ 5.2'),
    LLMSpec('phogpt-4b-chat', 'vinai/PhoGPT-4B-Chat', prompt='phogpt', min_transformers='4.40',
            max_transformers='4.45', trust_remote_code=True, service='phogpt',
            note='VinAI, 3,7B tham số huấn luyện từ đầu cho tiếng Việt, ngữ cảnh 8192; code MPT riêng chỉ chạy '
                 'với transformers 4.4x (dùng lớp RoPE của Llama đã bị xoá ở bản mới)'),
]}
DEFAULT_LLM = 'qwen2.5-3b'

ABSTAIN_RE = re.compile(r'không biết|không (có|tìm thấy|đề cập)[^.]{0,40}(thông tin|trong ngữ cảnh)|'
                        r'không thể trả lời|từ chối trả lời|ngữ cảnh không', re.I)
THINK_RE = re.compile(r'<think>.*?</think>', re.S)


def get_spec(key):
    if key not in LLMS:
        raise SystemExit(f'Không có LLM "{key}". Chọn: {", ".join(LLMS)}')
    return LLMS[key]


def _version_tuple(v):
    return tuple(int(x) for x in re.findall(r'\d+', v)[:3])


def check_transformers(spec):
    import transformers

    have = transformers.__version__
    v = _version_tuple(have)
    too_old = v < _version_tuple(spec.min_transformers)
    too_new = bool(spec.max_transformers) and v >= _version_tuple(spec.max_transformers)
    if too_old or too_new:
        need = f'≥ {spec.min_transformers}' + (f', < {spec.max_transformers}' if spec.max_transformers else '')
        raise SystemExit(
            f'{spec.repo} cần transformers {need}, môi trường này có {have}.\n'
            f'  → Chạy bước "answer" của mô hình này trong service {spec.service}:  '
            f'docker compose --profile {spec.service} run --rm {spec.service} answer --llm {spec.key}')


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
    @property
    def tokenizer(self):
        if self._tok is None:
            check_transformers(self.spec)
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.spec.repo, trust_remote_code=self.spec.trust_remote_code)
            tok.padding_side = 'left'
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tok = tok
        return self._tok

    @property
    def model(self):
        if self._model is None:
            check_transformers(self.spec)
            import torch
            import transformers as T

            self.log(f'nạp {self.spec.repo}')
            cuda = torch.cuda.is_available()
            kw = {'device_map': 'auto' if cuda else None, 'trust_remote_code': self.spec.trust_remote_code}
            # bf16 nếu GPU hỗ trợ (RTX 30+), fp16 cho T4, fp32 cho CPU.
            # Đặt cả khi 4-bit: phần KHÔNG lượng tử hoá (embedding, LayerNorm, lm_head) mặc định nạp fp16,
            # mà PhoGPT/MPT huấn luyện bf16 -> fp16 tràn số -> logits NaN -> sinh toàn token 0 (<unk>).
            half = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
            kw['dtype' if _version_tuple(T.__version__) >= (4, 56) else 'torch_dtype'] = half if cuda else torch.float32
            if self.load_4bit and cuda:
                kw['quantization_config'] = T.BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=half)
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
        msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
        tok = self.tokenizer
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
        import torch

        model, tok = self.model, self.tokenizer
        inputs = tok([self._prompt(system, user)], return_tensors='pt', add_special_tokens=False).to(model.device)
        with torch.no_grad():
            # use_cache=True: PhoGPT ghi use_cache=false trong config, không bật thì sinh rất chậm
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
