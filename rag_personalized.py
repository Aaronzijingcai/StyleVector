from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import numpy as np

# 下载 NLTK 数据
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')

# 数据集
dataset = load_dataset("parquet", data_files={"test": "../data/test-00000-of-00001.parquet"})
test_data = dataset['test']
#test_data = test_data_n.select(range(5))  # 选择前 10 条
input_dataset = Dataset.from_dict({"input": test_data['input'], "profile": test_data['profile']})

# 模型
device = torch.device("cuda:3")
model_address = "/NAS/HuggingFaceModels/Llama-2-7b-chat-hf"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_address)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充 token
model = AutoModelForCausalLM.from_pretrained(
    model_address,
    quantization_config=quantization_config,
    device_map={"": 3}
)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# 构建知识库（完整 chunk）
def build_knowledge_base(profile):
    knowledge_base = []
    for item in profile:
        chunk = (
            f"Generate the review text written by a reviewer who has given an overall rating of \"{item.get('overall', 'N/A')}\" "
            f"for a product with description \"{item.get('description', 'N/A')}\". The summary of the review text is \"{item.get('summary', 'N/A')}\""
        )
        knowledge_base.append({
            "chunk": chunk,
            "reviewText": item.get('reviewText', 'N/A')
        })
    return knowledge_base

def parse_input(input_text):
    try:
        rating_start = input_text.find("overall rating of ") + len("overall rating of ")
        rating_end = input_text.find(" for a product", rating_start)
        rating = input_text[rating_start:rating_end].strip().strip("\"")

        desc_start = input_text.find("product with description '") + len("product with description '")
        desc_end = input_text.find("'. The summary", desc_start)
        description = input_text[desc_start:desc_end].strip()

        summary_start = input_text.find("summary of the review text is '") + len("summary of the review text is '")
        summary_end = input_text.find("'", summary_start)
        summary = input_text[summary_start:summary_end].strip()

        return {"rating": rating, "description": description, "summary": summary}
    except Exception as e:
        print(f"Error parsing input: {e}")
        return {"rating": "N/A", "description": input_text, "summary": "N/A"}

# 检索（完整 input vs 完整 chunk）
def retrieve_from_knowledge_base(input_texts, knowledge_bases, k=2):
    retrieved_results = []
    for input_text, knowledge_base in zip(input_texts, knowledge_bases):
        if not knowledge_base:
            retrieved_results.append([])
            continue
        
        chunks = [item['chunk'] for item in knowledge_base]
        tokenized_chunks = [word_tokenize(chunk.lower()) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        
        # 直接使用完整 input_text 作为查询
        tokenized_query = word_tokenize(input_text.lower())
        scores = bm25.get_scores(tokenized_query)
        
        top_k_indices = np.argsort(scores)[::-1][:min(k, len(knowledge_base))]
        retrieved = [knowledge_base[idx] for idx in top_k_indices]
        retrieved_results.append(retrieved)
    return retrieved_results

# RAG 生成
def generate_batch(batch):
    # 构建知识库
    knowledge_bases = [build_knowledge_base(profile) for profile in batch["profile"]]
    
    # 检索Top2
    retrieved_items = retrieve_from_knowledge_base(batch["input"], knowledge_bases, k=2)
    
    # Aug Prompt
    prompts = []
    for input_text, retrieved in zip(batch["input"], retrieved_items):
        retrieved_context = "\n".join(
            #[f"Previous task: {item['chunk']}\nPrevious review: {item['reviewText']}" for item in retrieved]
            [f"Previous review: {item['reviewText']}" for item in retrieved]
        )
        parsed_input = parse_input(input_text)  # 解析用于任务指令
        augmented_prompt = (
            f"Generate the review text written by a reviewer who has given an overall rating of \"{parsed_input['rating']}\" "
            f"for a product with description \"{parsed_input['description']}\". The summary of the review text is \"{parsed_input['summary']}\"\n\n"
            f"Below are examples of previous reviews by the same reviewer to guide the style:\n"
            f"{retrieved_context}\n"
            #f"Write the review text directly, without adding extra explanations or instructions."
        )
        prompts.append(augmented_prompt)
    
    # 批量生成
    outputs = generator(
        prompts,
        max_new_tokens=350,
        num_return_sequences=1,
        truncation=True,
        do_sample=False,  # 保持贪婪解码
        return_full_text=False
    )
    return {"generated": [output[0]["generated_text"].strip() for output in outputs]}

# 批量处理
batch_size = 16
generated_dataset = input_dataset.map(
    generate_batch,
    batched=True,
    batch_size=batch_size,
    desc="Generating RAG-based reviews"
)

generated_reviews = generated_dataset["generated"]

# 评估
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = [scorer.score(ref, gen)['rougeL'].fmeasure for gen, ref in zip(generated_reviews, test_data['output'])]
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(gen)) for gen, ref in zip(generated_reviews, test_data['output'])]
avg_meteor = sum(meteor_scores) / len(meteor_scores)

print(f"Average ROUGE-L: {avg_rouge_l:.4f}")
print(f"Average METEOR: {avg_meteor:.4f}")