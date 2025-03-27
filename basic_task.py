import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置 OpenAI API（需自行获取密钥）
openai.api_key = "your-api-key"

# 加载 LLaMA-2-7B-chat
tokenizer = AutoTokenizer.from_pretrained("meta-ai/LLaMA-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-ai/LLaMA-2-7b-chat-hf", 
                                              output_hidden_states=True, 
                                              device_map="auto")

# Step 1: 生成 Style-Agnostic 响应
def generate_style_agnostic(input_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=512
    )
    return response.choices[0].message["content"]

# Step 2: 提取 Style Vector (Mean Difference)
def extract_style_vector(user_history, layer_idx=15):
    positive_acts, negative_acts = [], []
    for x_i, y_i in user_history:
        # 正向激活 (用户真实响应)
        inputs_p = tokenizer(x_i + " " + y_i, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs_p = model(**inputs_p)
            a_p = outputs_p.hidden_states[layer_idx][0, -1, :]  # 最后一层最后一个 token
        positive_acts.append(a_p)

        # 负向激活 (Style-Agnostic 响应)
        y_hat_i = generate_style_agnostic(x_i)
        inputs_n = tokenizer(x_i + " " + y_hat_i, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs_n = model(**inputs_n)
            a_n = outputs_n.hidden_states[layer_idx][0, -1, :]
        negative_acts.append(a_n)

    # Mean Difference (公式 7)
    s_u = torch.mean(torch.stack(positive_acts) - torch.stack(negative_acts), dim=0)
    return s_u

# Step 3: 激活干预生成
def generate_with_style_vector(input_text, style_vector, layer_idx=15, alpha=1.0):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = list(outputs.hidden_states)
        
        # 干预生成位置 (t >= |x|)
        input_len = inputs["input_ids"].shape[1]
        for t in range(input_len, 512):  # 假设最大长度 512
            hidden_states[layer_idx][0, t, :] += alpha * style_vector
        
        # 继续生成
        outputs = model.generate(**inputs, max_length=512, hidden_states=hidden_states)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 对每个用户处理
style_vector_outputs = []
for user_id in set([row["user_id"] for row in data]):
    # 获取用户历史数据
    user_history = [(row["input"], row["output"]) 
                    for row in data 
                    if row["user_id"] == user_id and row["is_history"]]
    test_input = [row["input"] 
                  for row in data 
                  if row["user_id"] == user_id and not row["is_history"]][0]
    test_gt = [row["output"] 
               for row in data 
               if row["user_id"] == user_id and not row["is_history"]][0]

    # 计算 Style Vector
    s_u = extract_style_vector(user_history)

    # 生成个性化输出
    pred = generate_with_style_vector(test_input, s_u)
    style_vector_outputs.append((test_gt, pred))

# 评估
rouge_l_scores_sv = [scorer.score(gt, pred)["rougeL"].fmeasure 
                     for gt, pred in style_vector_outputs]
meteor_scores_sv = [meteor_score([gt.split()], pred.split()) 
                    for gt, pred in style_vector_outputs]

print(f"StyleVector - Avg ROUGE-L: {sum(rouge_l_scores_sv)/len(rouge_l_scores_sv):.4f}")
print(f"StyleVector - Avg METEOR: {sum(meteor_scores_sv)/len(meteor_scores_sv):.4f}")