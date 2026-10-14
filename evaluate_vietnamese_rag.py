#!/usr/bin/env python3
"""
Vietnamese LLM RAG Evaluation Script
Tests top 3 Vietnamese LLMs for RAG performance.
Supports both HuggingFace models (GPU with 4-bit quantization) and GGUF models (llama.cpp).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Try importing required libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import numpy as np
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: uv pip install torch transformers sentence-transformers accelerate bitsandbytes")
    sys.exit(1)

# Try importing llama-cpp-python for GGUF models
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Note: llama-cpp-python not installed. GGUF models won't work. Install with: uv pip install llama-cpp-python")


@dataclass
class EvaluationResult:
    model_name: str
    question: str
    context: str
    generated_answer: str
    ground_truth: str
    latency_ms: float
    tokens_per_second: float
    faithfulness_score: float = 0.0
    relevancy_score: float = 0.0
    fluency_score: float = 0.0


@dataclass
class ModelConfig:
    name: str
    model_id: str
    tokenizer_id: Optional[str] = None
    use_gguf: bool = False
    gguf_file: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    load_4bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9


# Top 3 Vietnamese Models for RAG (HuggingFace - GPU with 4-bit quantization)
MODEL_CONFIGS = [
    ModelConfig(
        name="Vinallama-2.7B-Chat",
        model_id="vilm/vinallama-2.7b-chat",
        device_map="auto",
        torch_dtype="bfloat16",
        load_4bit=True,
        max_new_tokens=256,
        temperature=0.1,
    ),
    ModelConfig(
        name="Vistral-7B-Chat",
        model_id="Viet-Mistral/Vistral-7B-Chat",
        device_map="auto",
        torch_dtype="bfloat16",
        load_4bit=True,
        max_new_tokens=256,
        temperature=0.1,
    ),
    ModelConfig(
        name="Vietnamese-Llama2-7B-40GB",
        model_id="bkai-foundation-models/vietnamese-llama2-7b-40GB",
        device_map="auto",
        torch_dtype="bfloat16",
        load_4bit=True,
        max_new_tokens=256,
        temperature=0.1,
    ),
]

# GGUF Model Configs for CPU / llama.cpp inference
GGUF_MODEL_CONFIGS = [
    ModelConfig(
        name="Vistral-7B-Chat-GGUF",
        model_id="mradermacher/Vistral-7B-ChatML-i1-GGUF",
        use_gguf=True,
        gguf_file="Vistral-7B-ChatML.i1-Q4_K_M.gguf",
        max_new_tokens=256,
        temperature=0.1,
    ),
    ModelConfig(
        name="Vinallama-2.7B-Chat-GGUF",
        model_id="mradermacher/CHATBOT_VINALLAMA-GGUF",
        use_gguf=True,
        gguf_file="CHATBOT_VINALLAMA.Q4_K_M.gguf",
        max_new_tokens=256,
        temperature=0.1,
    ),
    ModelConfig(
        name="Vietnamese-Llama2-7B-40GB-GGUF",
        model_id="TheBloke/vietnamese-llama2-7B-40GB-GGUF",
        use_gguf=True,
        gguf_file="vietnamese-llama2-7b-40gb.Q4_K_M.gguf",
        max_new_tokens=256,
        temperature=0.1,
    ),
]

# Vietnamese Embedding Models
EMBEDDING_MODELS = {
    "bkai-biencoder": "bkai-foundation-models/vietnamese-bi-encoder",
    "aiteam-embedding": "AITeamVN/Vietnamese_Embedding",
    "dangvantuan-embedding": "dangvantuan/vietnamese-document-embedding",
}

# Reranker
RERANKER_MODEL = "AITeamVN/Vietnamese_Reranker"


# Vietnamese Test Dataset
VIETNAMESE_TEST_DATA = [
    {
        "question": "Thủ đô của Việt Nam là gì?",
        "context": "Hà Nội là thủ đô của Cộng hòa Xã hội Chủ nghĩa Việt Nam. Thành phố nằm ở phía Bắc nước, có diện tích 3.359,82 km² và dân số khoảng 8 triệu người (2023). Hà Nội là trung tâm chính trị, văn hóa, khoa học và công nghệ của cả nước.",
        "ground_truth": "Hà Nội là thủ đô của Việt Nam."
    },
    {
        "question": "Việt Nam có bao nhiêu tỉnh thành?",
        "context": "Hiện nay Việt Nam có 63 đơn vị hành chính cấp tỉnh, bao gồm 57 tỉnh và 6 thành phố trực thuộc trung ương: Hà Nội, Hồ Chí Minh, Hải Phòng, Đà Nẵng, Cần Thơ và Huế (từ năm 2025).",
        "ground_truth": "Việt Nam có 63 tỉnh thành (57 tỉnh và 6 thành phố trực thuộc trung ương)."
    },
    {
        "question": "Đặc điểm khí hậu của Việt Nam?",
        "context": "Việt Nam nằm trong khu vực khí hậu nhiệt đới gió mùa, có 2 mùa rõ rệt: mùa khô (tháng 11- tháng 4) và mùa mưa (tháng 5- tháng 10). Nhiệt độ trung niên năm khoảng 22-27°C. Khí hậu có sự khác biệt lớn giữa các vùng: Bắc có 4 mùa, Nam có 2 mùa, miền Trung có khí hậu chuyển tiếp.",
        "ground_truth": "Việt Nam có khí hậu nhiệt đới gió mùa với 2 mùa rõ rệt: mùa khô và mùa mưa."
    },
    {
        "question": "Sông nào dài nhất ở Việt Nam?",
        "context": "Sông Mekong (tiếng Việt gọi là Sông Cửu Long) là con sông dài nhất chảy qua Việt Nam với đoạn chảy qua lãnh thổ Việt Nam khoảng 220-250 km. Tuy nhiên, nếu tính toàn bộ chiều dài từ nguồn, sông Hồng có đoạn chảy qua Việt Nam khoảng 510 km, dài hơn đoạn sông Mekong ở Việt Nam.",
        "ground_truth": "Sông Hồng là sông dài nhất chảy qua Việt Nam (510 km), sông Mekong dài hơn toàn bộ nhưng đoạn qua Việt Nam ngắn hơn."
    },
    {
        "question": "Tết Nguyên Đán là dịp lễ gì?",
        "context": "Tết Nguyên Đán (Tết Ta, Tết Cổ truyền) là dịp lễ quan trọng nhất, lớn nhất trong năm của người Việt Nam. Tết đánh dấu sự chuyển giao giữa năm cũ và năm mới theo âm lịch, thường rơi vào cuối tháng 1 hoặc đầu tháng 2 dương lịch. Người dân thường về sum họp gia đình, cúng ông Táo, ăn Tết, mừng tuổi, lì xì.",
        "ground_truth": "Tết Nguyên Đán là dịp lễ lớn nhất, quan trọng nhất trong năm của người Việt Nam, đánh dấu năm mới âm lịch."
    },
    {
        "question": "Phổ biến nhất về ẩm thực Việt Nam?",
        "context": "Phở là món ăn đại diện nhất của ẩm thực Việt Nam, được UNESCO công nhận là di sản văn hóa phi vật thể. Ngoài ra còn có bún chả, bánh mì, gỏi cuốn, bún bò Huế, cơm tấm, bánh xèo. Mỗi vùng miền có đặc trưng: Bắc - phở, bún chả; Trung - bún bò Huế, mì Quảng; Nam - hủ tiếu, cơm tấm.",
        "ground_truth": "Phở là món ăn đại diện nhất Việt Nam, ngoài ra có bún chả, bánh mì, bún bò Huế, cơm tấm tùy vùng miền."
    },
    {
        "question": "Vịnh Hạ Long ở đâu?",
        "context": "Vịnh Hạ Long nằm ở tỉnh Quảng Ninh, phía Đông Bắc Việt Nam. Được UNESCO công nhận là Di sản Thiên nhiên Thế giới năm 1994 và 2000. Vịnh có 1.969 hòn đảo đá vôi trỗi lên từ mặt nước, tạo nên khung cảnh tuyệt đẹp. Du khách thường đi tàu du lịch, leo núi Titop, thăm hang Sửng Sốt.",
        "ground_truth": "Vịnh Hạ Long nằm ở tỉnh Quảng Ninh, được UNESCO công nhận Di sản Thiên nhiên Thế giới."
    },
    {
        "question": "Ngôn ngữ chính thức của Việt Nam?",
        "context": "Tiếng Việt là ngôn ngữ chính thức và phổ thông của Việt Nam, sử dụng chữ Quốc ngữ (bảng chữ cái La tinh) từ đầu thế kỷ 20. Có 6 thanh điệu: ngang, huyền, sắc, hỏi, ngã, nặng. Ngoài ra các dân tộc thiểu số có ngôn ngữ riêng (Tày, Nùng, Mông, Dao, Khmer, Chăm...).",
        "ground_truth": "Tiếng Việt là ngôn ngữ chính thức, dùng chữ Quốc ngữ, có 6 thanh điệu."
    },
    {
        "question": "Đơn vị tiền tệ Việt Nam?",
        "context": "Đồng Việt Nam (VND, ký hiệu: ₫) là đơn vị tiền tệ chính thức của Việt Nam do Ngân hàng Nhà nước phát hành. Hiện có mệnh giá: 1.000đ, 2.000đ, 5.000đ, 10.000đ, 20.000đ, 50.000đ, 100.000đ, 200.000đ, 500.000đ (giấy) và các đồng xu (hiếm dùng).",
        "ground_truth": "Đồng Việt Nam (VND, ₫) là đơn vị tiền tệ chính thức, mệnh giá từ 1.000đ đến 500.000đ."
    },
    {
        "question": "Hệ thống giáo dục Việt Nam?",
        "context": "Giáo dục Việt Nam gồm: Mầm non (3-5 tuổi), Tiểu học (5 năm, lớp 1-5), THCS (4 năm, lớp 6-9), THPT (3 năm, lớp 10-12), Đại học/Cao đẳng (4-6 năm). Có cả giáo dục thường xuyên, chính quy và tài năng. Tỷ lệ biết chữ > 98%. Kỳ thi THPT Quốc gia là cửa ngõ vào đại học.",
        "ground_truth": "Giáo dục VN: Mầm non -> Tiểu học (5n) -> THCS (4n) -> THPT (3n) -> ĐH/CĐ (4-6n). Tỷ lệ biết chữ >98%."
    },
]


class VietnameseRAGEvaluator:
    def __init__(self, model_config: ModelConfig, embedding_model_name: str = "bkai-biencoder"):
        self.model_config = model_config
        self.embedding_model_name = embedding_model_name
        self.llm_pipeline = None
        self.embedding_model = None
        self.reranker = None
        self.results = []
        self._doc_embs = None
        self._doc_corpus = None
    
    def load_models(self):
        print(f"\n{'='*60}")
        print(f"Loading models for: {self.model_config.name}")
        print(f"{'='*60}")
        
        # Load LLM - check if GGUF model
        if self.model_config.use_gguf:
            return self._load_gguf_model()
        else:
            return self._load_hf_model()
    
    def _load_hf_model(self):
        """Load HuggingFace model (GPU with optional 4-bit quantization)"""
        print(f"Loading HF LLM: {self.model_config.model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.tokenizer_id or self.model_config.model_id,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            cuda = torch.cuda.is_available()
            half = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
            
            kw = {
                "device_map": self.model_config.device_map if cuda else None,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            if self.model_config.load_4bit and cuda:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=half
                )
            else:
                kw["torch_dtype"] = half if cuda else torch.float32

            model = AutoModelForCausalLM.from_pretrained(
                self.model_config.model_id,
                **kw
            )
            
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
            print("[OK] HF LLM loaded successfully")
        except Exception as e:
            print(f"[FAIL] Failed to load HF LLM: {e}")
            return False
        
        # Load Embedding Model
        emb_repo = EMBEDDING_MODELS[self.embedding_model_name]
        print(f"Loading embedding model: {emb_repo}...")
        try:
            self.embedding_model = SentenceTransformer(
                emb_repo,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("[OK] Embedding model loaded")
        except Exception as e:
            print(f"[FAIL] Failed to load embedding: {e}")
            return False
        
        # Load Reranker
        print(f"Loading reranker: {RERANKER_MODEL}...")
        try:
            self.reranker = CrossEncoder(
                RERANKER_MODEL,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("[OK] Reranker loaded")
        except Exception as e:
            print(f"[WARN] Reranker load failed (optional): {e}")
            self.reranker = None
        
        return True
    
    def _load_gguf_model(self):
        """Load GGUF model via llama-cpp-python"""
        if not LLAMA_CPP_AVAILABLE:
            print("[FAIL] llama-cpp-python not installed. Install with: uv pip install llama-cpp-python")
            return False
        
        print(f"Loading GGUF LLM: {self.model_config.model_id}...")
        try:
            local_path = Path("./models") / (self.model_config.gguf_file or "")
            if local_path.is_file():
                model_path = str(local_path)
                print(f"  Using existing local file: {model_path}")
            else:
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id=self.model_config.model_id,
                    filename=self.model_config.gguf_file,
                    local_dir="./models",
                )
                print(f"  Downloaded to: {model_path}")
            
            self.llm_pipeline = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                n_gpu_layers=0,
                verbose=False
            )
            print("[OK] GGUF LLM loaded successfully")
            
        except Exception as e:
            print(f"[FAIL] Failed to load GGUF LLM: {e}")
            return False
        
        # Load Embedding Model
        emb_repo = EMBEDDING_MODELS[self.embedding_model_name]
        print(f"Loading embedding model: {emb_repo}...")
        try:
            self.embedding_model = SentenceTransformer(
                emb_repo,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("[OK] Embedding model loaded")
        except Exception as e:
            print(f"[FAIL] Failed to load embedding: {e}")
            return False
        
        # Load Reranker
        print(f"Loading reranker: {RERANKER_MODEL}...")
        try:
            self.reranker = CrossEncoder(
                RERANKER_MODEL,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("[OK] Reranker loaded")
        except Exception as e:
            print(f"[WARN] Reranker load failed (optional): {e}")
            self.reranker = None
        
        return True
    
    def precompute_document_embeddings(self, documents: List[str]):
        """Precompute embeddings for the document corpus once"""
        self._doc_corpus = documents
        self._doc_embs = self.embedding_model.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def retrieve_context(self, question: str, documents: List[str], top_k: int = 5) -> List[str]:
        """Retrieve relevant context using embedding + reranking"""
        if not documents:
            return []
        
        if self._doc_embs is None or self._doc_corpus != documents:
            self.precompute_document_embeddings(documents)
        
        # Encode question
        q_emb = self.embedding_model.encode([question], convert_to_numpy=True)
        
        # Cosine similarity
        similarities = np.dot(self._doc_embs, q_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Rerank if available
        if self.reranker:
            try:
                pairs = [[question, documents[i]] for i in top_indices]
                rerank_scores = self.reranker.predict(pairs)
                rerank_indices = np.argsort(rerank_scores)[::-1]
                top_indices = top_indices[rerank_indices]
            except Exception as e:
                pass
        
        return [documents[i] for i in top_indices[:3]]  # Top 3 after rerank
    
    def generate_answer(self, question: str, context: str) -> tuple:
        """Generate answer using LLM with context (supports both HF and GGUF)"""
        prompt = f"""Dựa trên ngữ cảnh dưới đây, hãy trả lời câu hỏi bằng tiếng Việt một cách chính xác và ngắn gọn.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời:"""
        
        start_time = time.time()
        
        if self.model_config.use_gguf:
            # GGUF inference via llama.cpp
            output = self.llm_pipeline(
                prompt,
                max_tokens=self.model_config.max_new_tokens,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                stop=["Ngữ cảnh:", "Câu hỏi:", "Trả lời:", "\n\nNgữ cảnh:", "\n\nCâu hỏi:"],
                echo=False
            )
            generated = output['choices'][0]['text'].strip()
        else:
            # HF pipeline inference
            tok = self.llm_pipeline.tokenizer
            pad_id = tok.pad_token_id or tok.eos_token_id
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=self.model_config.max_new_tokens,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                do_sample=self.model_config.temperature > 0,
                pad_token_id=pad_id,
                return_full_text=False,
            )
            generated = outputs[0]['generated_text'].strip()
            if "Trả lời:" in generated:
                generated = generated.split("Trả lời:")[-1].strip()
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate tokens per second
        if self.model_config.use_gguf:
            tokens = len(self.llm_pipeline.tokenize(generated.encode('utf-8'))) if hasattr(self.llm_pipeline, 'tokenize') else len(generated.split()) * 1.3
        else:
            tokens = len(self.llm_pipeline.tokenizer.encode(generated, add_special_tokens=False))
        tps = tokens / (latency_ms / 1000) if latency_ms > 0 else 0
        
        return generated, latency_ms, tps
    
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """Simple faithfulness check: answer should not contradict context"""
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        return min(overlap * 1.5, 1.0)
    
    def evaluate_relevancy(self, answer: str, question: str) -> float:
        """Simple relevancy: answer should address question keywords"""
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        stopwords = {'là', 'gì', 'của', 'có', 'những', 'các', 'và', 'hoặc', 'trong', 'ở', 'tại', 'về', 'cho', 'với'}
        q_keywords = q_words - stopwords
        a_keywords = a_words - stopwords
        if not q_keywords:
            return 0.5
        overlap = len(q_keywords & a_keywords) / len(q_keywords)
        return min(overlap, 1.0)
    
    def evaluate_fluency(self, answer: str) -> float:
        """Simple Vietnamese fluency check"""
        if not answer or len(answer) < 5:
            return 0.0
        vietnamese_chars = sum(1 for c in answer if ord(c) > 127)
        vietnamese_ratio = vietnamese_chars / len(answer)
        length_score = min(len(answer) / 100, 1.0)
        return (vietnamese_ratio * 0.7 + length_score * 0.3)
    
    def run_evaluation(self, test_data: List[Dict]) -> List[EvaluationResult]:
        print(f"\nRunning evaluation on {len(test_data)} test cases...")
        
        # Prepare document corpus from test data contexts
        documents = list(set([item['context'] for item in test_data]))
        self.precompute_document_embeddings(documents)
        
        for i, item in enumerate(test_data):
            print(f"\n  Test {i+1}/{len(test_data)}: {item['question'][:50]}...")
            
            # Retrieve context
            retrieved = self.retrieve_context(item['question'], documents)
            context = "\n".join(retrieved)
            
            # Generate answer
            answer, latency, tps = self.generate_answer(item['question'], context)
            
            # Evaluate
            faith = self.evaluate_faithfulness(answer, context)
            relev = self.evaluate_relevancy(answer, item['question'])
            fluency = self.evaluate_fluency(answer)
            
            result = EvaluationResult(
                model_name=self.model_config.name,
                question=item['question'],
                context=context[:200] + "...",
                generated_answer=answer,
                ground_truth=item['ground_truth'],
                latency_ms=latency,
                tokens_per_second=tps,
                faithfulness_score=faith,
                relevancy_score=relev,
                fluency_score=fluency,
            )
            
            self.results.append(result)
            print(f"    Answer: {answer[:80]}...")
            print(f"    Latency: {latency:.0f}ms, TPS: {tps:.1f}, Faith: {faith:.2f}, Rel: {relev:.2f}, Flu: {fluency:.2f}")
        
        return self.results
    
    def save_results(self, output_dir: str = "evaluation_results"):
        Path(output_dir).mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_config.name.replace('/', '_')
        filename = f"{output_dir}/{safe_name}_{timestamp}.json"
        
        data = {
            "model": self.model_config.name,
            "model_id": self.model_config.model_id,
            "embedding_model": self.embedding_model_name,
            "timestamp": timestamp,
            "results": [asdict(r) for r in self.results],
            "summary": self.get_summary(),
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to: {filename}")
        return filename
    
    def get_summary(self) -> Dict:
        if not self.results:
            return {}
        
        return {
            "total_tests": len(self.results),
            "avg_latency_ms": float(np.mean([r.latency_ms for r in self.results])),
            "avg_tokens_per_sec": float(np.mean([r.tokens_per_second for r in self.results])),
            "avg_faithfulness": float(np.mean([r.faithfulness_score for r in self.results])),
            "avg_relevancy": float(np.mean([r.relevancy_score for r in self.results])),
            "avg_fluency": float(np.mean([r.fluency_score for r in self.results])),
        }
    
    def print_summary(self):
        summary = self.get_summary()
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.model_config.name}")
        print(f"{'='*60}")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.2f}")
            else:
                print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vietnamese LLMs for RAG")
    parser.add_argument("--models", nargs="+", default=["all"], 
                        choices=["all", "vistral", "vinallama", "vietnamese-llama2", "gguf-all", "gguf-vistral", "gguf-vinallama", "gguf-vietnamese-llama2"],
                        help="Models to evaluate")
    parser.add_argument("--embedding", default="bkai-biencoder",
                        choices=list(EMBEDDING_MODELS.keys()),
                        help="Embedding model to use")
    parser.add_argument("--output-dir", default="evaluation_results",
                        help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                        help="Run quick test with fewer samples (5 samples)")
    
    args = parser.parse_args()
    
    # Select models
    all_configs = MODEL_CONFIGS + GGUF_MODEL_CONFIGS
    selected_models = []
    
    if "all" in args.models:
        selected_models = MODEL_CONFIGS
    elif "gguf-all" in args.models:
        selected_models = GGUF_MODEL_CONFIGS
    else:
        name_map = {
            "vistral": "Vistral-7B-Chat",
            "vinallama": "Vinallama-2.7B-Chat",
            "vietnamese-llama2": "Vietnamese-Llama2-7B-40GB",
            "gguf-vistral": "Vistral-7B-Chat-GGUF",
            "gguf-vinallama": "Vinallama-2.7B-Chat-GGUF",
            "gguf-vietnamese-llama2": "Vietnamese-Llama2-7B-40GB-GGUF",
        }
        for m in args.models:
            target_name = name_map.get(m)
            for cfg in all_configs:
                if cfg.name == target_name:
                    selected_models.append(cfg)
    
    if not selected_models:
        print("No models selected for evaluation.")
        return
    
    # Select test data
    test_data = VIETNAMESE_TEST_DATA[:5] if args.quick else VIETNAMESE_TEST_DATA
    
    print(f"\n{'#'*60}")
    print(f"VIETNAMESE LLM RAG EVALUATION")
    print(f"{'#'*60}")
    print(f"Models to test: {[m.name for m in selected_models]}")
    print(f"Embedding: {args.embedding}")
    print(f"Test cases: {len(test_data)}")
    print(f"Quick mode: {args.quick}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"llama-cpp-python available: {LLAMA_CPP_AVAILABLE}")
    
    all_results = {}
    
    for model_config in selected_models:
        evaluator = VietnameseRAGEvaluator(model_config, args.embedding)
        
        if not evaluator.load_models():
            print(f"Skipping {model_config.name} due to load failure")
            continue
        
        evaluator.run_evaluation(test_data)
        evaluator.print_summary()
        evaluator.save_results(args.output_dir)
        all_results[model_config.name] = evaluator.get_summary()
        
        # Free memory thoroughly
        del evaluator.llm_pipeline
        del evaluator.embedding_model
        if evaluator.reranker:
            del evaluator.reranker
        del evaluator
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print comparison
    print(f"\n{'#'*60}")
    print(f"MODEL COMPARISON")
    print(f"{'#'*60}")
    print(f"{'Model':<35} {'Latency(ms)':<12} {'TPS':<8} {'Faith':<8} {'Rel':<8} {'Flu':<8}")
    print("-" * 85)
    for name, summary in all_results.items():
        print(f"{name:<35} {summary.get('avg_latency_ms',0):<12.0f} "
              f"{summary.get('avg_tokens_per_sec',0):<8.1f} "
              f"{summary.get('avg_faithfulness',0):<8.2f} "
              f"{summary.get('avg_relevancy',0):<8.2f} "
              f"{summary.get('avg_fluency',0):<8.2f}")
    
    # Determine winner
    if all_results:
        best = max(all_results.items(), 
                   key=lambda x: x[1].get('avg_faithfulness',0) + x[1].get('avg_relevancy',0) + x[1].get('avg_fluency',0))
        print(f"\n[WINNER] BEST OVERALL: {best[0]}")
    
    print(f"\nDetailed results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
