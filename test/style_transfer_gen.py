from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
import os
import csv

model_id = "/path/to/LLaMA-Factory/saves/fine_tune"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto", 
    device_map="auto"   
)

input_file = "/path/to/characterbot/example/task3_question.csv"  
output_file = "/path/to/characterbot/example/task3_answer.csv"  

output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"输出目录 {output_dir} 已创建")

with open(input_file, "r", encoding="utf-8") as file:
    csvreader = csv.DictReader(file)
    questions = list(csvreader)

if not questions:
    print("错误：输入文件中没有有效的问题！")
    exit(1)

start_index = 1
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        existing_lines = list(csv.reader(f))
        start_index = len(existing_lines)  
        print(f"已有 {start_index - 1} 条回答，继续生成。")

with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
    
    csvwriter = csv.writer(csvfile)

    if start_index == 1:
        csvwriter.writerow(["编号", "问题", "生成文本", "答案"])

    for idx, row in enumerate(questions[start_index - 1:], start=start_index):
        try:
            question = row.get("问题", "").strip()
            correct_answer = row.get("答案", "").strip()

            messages = [
                {"role": "system", "content": "你是鲁迅。回答问题时，请使用鲁迅的视角、口吻和风格。"},
                {"role": "user", "content": question},
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            outputs = model.generate(
                input_ids,              
                max_new_tokens=256,     
                do_sample=True,        
                temperature=0.7,        
                top_p=0.9,              
            )

            response_ids = outputs[0][input_ids.shape[-1]:]
            generated_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

            print(f"{idx}, {question}, 生成文本: {generated_text}, 正确答案: {correct_answer}")

            csvwriter.writerow([row.get("编号", ""), question, generated_text, correct_answer])

        except Exception as e:
            print(f"生成答案时发生错误，问题为：{row.get('问题', '')}")
            print(f"错误信息：{e}")
            csvwriter.writerow([row.get("编号", ""), row.get("问题", ""), "生成失败", row.get("答案", "")])
            continue

print(f"生成完成，答案已保存到 {output_file} 文件中！")
