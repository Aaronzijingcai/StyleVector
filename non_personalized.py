from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score

# 加载模型和分词器
model_name = "meta-ai/LLaMA-2-7b-chat-hf"  # 需申请访问权限
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Non-personalized 生成
def generate_non_personalized(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=512, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 测试数据
test_inputs = [row["input"] for row in data if not row["is_history"]]
ground_truths = [row["output"] for row in data if not row["is_history"]]

# 生成结果
non_personalized_outputs = [generate_non_personalized(x) for x in test_inputs]


scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
rouge_l_scores = [scorer.score(gt, pred)["rougeL"].fmeasure 
                  for gt, pred in zip(ground_truths, non_personalized_outputs)]
meteor_scores = [meteor_score([gt.split()], pred.split()) 
                 for gt, pred in zip(ground_truths, non_personalized_outputs)]

print(f"Non-personalized - Avg ROUGE-L: {sum(rouge_l_scores)/len(rouge_l_scores):.4f}")
print(f"Non-personalized - Avg METEOR: {sum(meteor_scores)/len(meteor_scores):.4f}")