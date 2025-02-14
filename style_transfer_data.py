# -*- coding: utf-8 -*-

import os
import json
from openai import OpenAI


client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.20a.top/v1"
)

def transform_to_plain_text(article):
    system_message = (
        "你是一个熟悉鲁迅文风和现代白话文的大模型。"
        "接下来会给你一段鲁迅的文章内容，你需要挑选出其中三句有代表性的句子，并将它们逐一转换为现代白话文。"
        "要求：\n"
        "1、挑选的句子应具有鲁迅风格的特点，并能够反映文章的主要思想；\n"
        "2、挑选的句子长度应在100字左右（不低于80字且不超过120字），以保证句子完整且信息充实；\n"
        "3、转换时用流畅的现代白话文表达句子原意，避免过于复杂或晦涩的表达；\n"
        "4、输出结果必须是严格的 JSON 数组格式，例如：\n"
        "[{\"original\": \"原句1\", \"plain\": \"白话文1\"}, {\"original\": \"原句2\", \"plain\": \"白话文2\"}]\n"
        "不要包含其他多余字符或解释。\n"
    )
    user_message = article["doc"]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.4,
            top_p=0.9
        )
        
        transformed_text = response.choices[0].message.content.strip()

        if transformed_text.startswith("```json") and transformed_text.endswith("```"):
            transformed_text = transformed_text[7:-3].strip()

        transformed_data = json.loads(transformed_text)

        return {
            "title": article["title"],
            "transformed_sentences": transformed_data
        }
    except Exception as e:
        print(f"Error while transforming '{article['title']}': {e}")
        return None
    
def process_articles(input_filename, output_filename):

    with open(input_filename, "r", encoding="utf-8") as file:
        articles = json.load(file)

    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            try:
                all_transformed = json.load(f)
            except json.JSONDecodeError:
                all_transformed = []
    else:
        all_transformed = []

    processed_titles = {item["title"] for item in all_transformed}
    
    for article in articles:
        if article["title"] in processed_titles:
            print(f"Article '{article['title']}' already processed, skipping.")
            continue

        result = transform_to_plain_text(article)
        if result:
            all_transformed.append(result)
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(all_transformed, f, ensure_ascii=False, indent=2)
            print(f"Processed article '{article['title']}' and saved transformed sentences.")


def main():
    input_filename = "luxun_essay.json"
    output_filename = "style_transfer.json"
    process_articles(input_filename, output_filename)


if __name__ == "__main__":
    main()
