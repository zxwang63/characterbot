import pandas as pd
import jieba
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def tokenize(text):
    return list(jieba.cut(text.strip()))

def calculate_bleu(df):
    references = [[tokenize(ans)] for ans in df['答案']]
    hypotheses = [tokenize(gen) for gen in df['生成文本']]
    smoother = SmoothingFunction().method1
    return corpus_bleu(references, hypotheses, weights=(0.5, 0.5), smoothing_function=smoother)

df = pd.read_csv(f"/path/to/task3_answer.csv")
bleu_scores = calculate_bleu(df)
print(f"BLEU: {bleu_scores:.4f}")

