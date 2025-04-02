from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import nltk
import torch
from tqdm import tqdm

# 下载 NLTK 数据
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('punkt_tab')
#print(1)

# 数据集
dataset = load_dataset("parquet", data_files={"test": "../data/test-00000-of-00001.parquet"})
test_data = dataset['test']
input_dataset = Dataset.from_dict({"input": test_data['input']})

device = torch.device("cuda:6")
model_address = "/NAS/HuggingFaceModels/Llama-2-7b-chat-hf"
quantization_config = BitsAndBytesConfig(load_in_8bit=True) 
tokenizer = AutoTokenizer.from_pretrained(model_address)
model = AutoModelForCausalLM.from_pretrained(
    model_address,
    quantization_config=quantization_config,
    device_map={"": 6}  
)
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)



def generate_batch(batch):
    outputs = generator(
        batch["input"],
        max_new_tokens=350,  # 调整为更贴近平均输出长度 (304.54)
        num_return_sequences=1,  # 每个输入得到一个输出序列
        truncation=True,
        do_sample=False,  # Appendix B提到使用贪婪解码
        return_full_text=False  # 只返回生成部分
    )
    return {"generated": [output[0]["generated_text"].strip() for output in outputs]}
#print(5)

# 批量处理
batch_size = 16
generated_dataset = input_dataset.map(
    generate_batch,
    batched=True,
    batch_size=batch_size,
    desc="Generating reviews"
)

generated_reviews = generated_dataset["generated"]

# ROUGE_L
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
rouge_l_scores = [scorer.score(ref, gen)['rougeL'].fmeasure for gen, ref in zip(generated_reviews, test_data['output'])]
avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

# METEOR
meteor_scores = [meteor_score([word_tokenize(ref)], word_tokenize(gen)) for gen, ref in zip(generated_reviews, test_data['output'])]
avg_meteor = sum(meteor_scores) / len(meteor_scores)

output_file = "non_personalized_results.txt"
with open(output_file, "w") as f:
    f.write(f"Average ROUGE-L: {avg_rouge_l:.4f}\n")
    f.write(f"Average METEOR: {avg_meteor:.4f}\n")
