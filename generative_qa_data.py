import os
import json
from openai import OpenAI


client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.20a.top/v1"
)

def generate_qa_pairs(article):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"这篇名为'{article['title']}'的文章是鲁迅写的。文章内容如下：[{article['doc']}]。请基于这篇文章生成三个问题-答案对。每个问题都应基于文章中的内容，直接询问鲁迅的观点和看法。问题是向鲁迅提问的口气，必须直接使用“你”来提问。在问题中不要提及文章，不要出现“作者”“鲁迅”“本文”“文中”“文章”等字词。如果问题是关于文章特定概念的，就在问题中解释这个概念。所有问题必须严格基于文章内容，不引入任何未提及的内容。回答基于文章内容，以鲁迅的口吻，以鲁迅式语言和独特视角回应，表述尽可能详细、精确、贴近原文的意思，不引入任何未提及的内容，也不得出现‘鲁迅’、‘作者’、‘文章’、‘本文’等字词。输出必须为严格的 JSON 数组格式，不含多余字符、格式或解释，且 JSON 数组中的每个对象仅包含 'question' 和 'answer' 两个键。"
            }
        ]
    )

    qa_pairs = response.choices[0].message.content
    qa_pairs1 = qa_pairs.replace('```json\n', '').replace('\n```', '').strip()

    try:
        data = json.loads(qa_pairs1)
        return {
            "title": article['title'],
            "qa_pairs": data
        }
    except json.JSONDecodeError as e:
        print(f"JSON 解析失败: {e}")
        return None

def process_articles(input_filename, output_filename):

    with open(input_filename, "r", encoding="utf-8") as file:
        articles = [json.loads(line) for line in file]

    os.makedirs("json_output", exist_ok=True)
    output_path = os.path.join("json_output", output_filename)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                all_qa_pairs = json.load(f)
            except json.JSONDecodeError:
                all_qa_pairs = []
    else:
        all_qa_pairs = []

    processed_titles = {item['title'] for item in all_qa_pairs}

    for article in articles:
        if article['title'] in processed_titles:
            print(f"Article '{article['title']}' already processed, skipping.")
            continue

        try:
            qa_data = generate_qa_pairs(article)
            if qa_data:
                all_qa_pairs.append(qa_data)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_qa_pairs, f, ensure_ascii=False, indent=4)
                print(f"Processed article '{article['title']}' and saved results.")
        except Exception as e:
            print(f"Error processing article '{article['title']}': {e}")


def main():
    input_filename = "luxun_essay.json"
    output_filename = "generative_qa.json"
    process_articles(input_filename, output_filename)


if __name__ == "__main__":
    main()
