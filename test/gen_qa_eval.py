import json
import pandas as pd
import os
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

def review_luxun_responses(context):
    prompt_template = (
        "有一些AI助手正在模拟鲁迅回答问题，现需要对以下模拟鲁迅的AI助手的回答进行评分，评分时仅考虑所提供的鲁迅文章原文：\n"
        "1. 评价回答是否符合所提供的鲁迅原文的语言风格，\n"
        "- 仅考虑语言风格因素，不得考虑核心意思等其他因素。\n"
        "[直接扣分情况]\n"
        "- 使用现代白话文，\n"
        "- 使用非中文，\n"
        "- 使用现代白话中文常用的词汇，而不符合所提供鲁迅原文的风格，\n"
        "- 说教式表达。\n"
        "2. 评价回答是否符合所提供的鲁迅原文的核心意思（包括是否符合所提供的鲁迅原文的事实，以及是否契合所提供的鲁迅原文所表达的思想、情感或立场），\n"
        "- 仅考虑核心意思因素，不得考虑语言风格等其他因素。\n"
        "[直接扣分情况]\n"
        "- 泛泛而谈，与鲁迅原文无关。\n"
        "3. 分别对这两方面进行评分（1-5分），1表示极不符合，5表示非常符合。\n\n"
        "请按照以下格式分别对每个AI助手的回答进行评测，对于每个AI助手的回答只输出4行内容，其中在第2行和第4行仅输出数字评分：\n"
        "第1行：对语言风格的简要评价说明。\n"
        "第2行：语言风格评分（1-5）。\n"
        "第3行：对核心意思的简要评价说明。\n"
        "第4行：核心意思评分（1-5）。\n\n"
        "评分开始\n"
        "[鲁迅原文内容]\n"
        "{background}\n"
        "[用户的问题]\n"
        "{question}\n"
        "[AI助手的回答]\n"
        "{answers}"
    )
    answers_str = "\n".join([f"AI助手{i+1}回答：\n{ans}" for i, ans in enumerate(context["answers"])])
    prompt = prompt_template.format(
        background=context["background"],
        question=context["question"],
        answers=answers_str
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位对鲁迅文学十分熟悉的审稿评分专家。"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        result = response.choices[0].message.content.strip()
        print(result)
        return result
    except Exception as e:
        print(f"Error while reviewing responses: {e}")
        return None

def log_result_to_file(log_file, question_id, ai_id, language_score, meaning_score):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"编号: {question_id}, AI助手: {ai_id}, 语言风格评分: {language_score}, 核心意思评分: {meaning_score}\n")

def process_reviews(question_file, answer_file, log_file):
    questions = pd.read_csv(question_file)
    answers_df = pd.read_csv(answer_file)

    for idx, row in questions.iterrows():
        question_id = row["编号"]
        question_text = row["问题"]
        background = row["原文"]
        relevant_answers = answers_df[answers_df["编号"] == question_id]
        answers_list = relevant_answers["答案"].tolist()

        context = {
            "background": background,
            "question": question_text,
            "answers": answers_list
        }

        result = review_luxun_responses(context)
        if result:
            lines = [line.strip() for line in result.split("\n") if line.strip()]
            for i in range(len(answers_list)):
                try:
                    base_index = i * 5
                    language_score = int(lines[base_index + 2])
                    meaning_score = int(lines[base_index + 4])
                    ai_id = i + 1
                    log_result_to_file(log_file, question_id, ai_id, language_score, meaning_score)
                    print(f"编号: {question_id}, AI助手{ai_id} 语言风格评分: {language_score}, 核心意思评分: {meaning_score}")
                except (ValueError, IndexError) as e:
                    print(f"Error parsing results for Question {question_id}, AI助手{i + 1}: {e}")

def main():
    question_file = "/path/to/characterbot/example/task2_question_text.csv"
    answer_file = "/path/to/characterbot/example/task2_answer.csv"
    log_file = "/path/to/characterbot/example/review_log_test.txt"
    process_reviews(question_file, answer_file, log_file)

if __name__ == "__main__":
    main()
