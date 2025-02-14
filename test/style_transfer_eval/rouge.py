import os
import pandas as pd
from rouge import Rouge
import jieba
import re

file_path = "/path/to/task3_answer.csv"

rouge = Rouge()

def preprocess_text(text):
    text = re.sub(r"[，。、：；“”\"‘’]", "", text)
    words = jieba.lcut(text)
    return " ".join(words).strip()

def calculate_rouge_1(reference, candidate):
    scores = rouge.get_scores(candidate, reference, avg=True)
    return scores['rouge-1']['f']

if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    data = pd.read_csv(file_path)
    
    rouge_1_scores = []

    for _, row in data.iterrows():
        candidate = preprocess_text(row['生成文本'])
        reference = preprocess_text(row['答案'])
        rouge_1_score = calculate_rouge_1(reference, candidate)
        rouge_1_scores.append(rouge_1_score)

    avg_rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores) if rouge_1_scores else 0

    print("Evaluation Results:")
    print(f"ROUGE-1: {avg_rouge_1:.4f}")
