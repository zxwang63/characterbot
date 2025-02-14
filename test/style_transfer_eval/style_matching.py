import os
import pandas as pd
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com"
)

def check_language_style_consistency(question, generated_text, answer):
    prompt = (
        f"问题: {question}\n"
        f"生成文本: {generated_text}\n"
        f"答案: {answer}\n"
        f"请判断生成文本和答案的语言风格是否一致，1表示一致，0表示不一致。请只输出数字1或0，不得输出其他内容。"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一位精通语言风格分析的专家。"},
                {"role": "user", "content": prompt},
            ],
        )
        result = response.choices[0].message.content.strip()
        print(result)
        return int(result)
    except Exception as e:
        print(f"Error during API call: {e}")
        return 0

def process_file():
    input_file = "/path/to/characterbot/example/task3_answer.csv"
    output_file = "/path/to/result/task3_answer_result.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    data = pd.read_csv(input_file)
    results = []
    consistent_count = 0

    for index, row in data.iterrows():
        question = row['问题']
        generated_text = row['生成文本']
        answer = row['答案']

        consistency = check_language_style_consistency(question, generated_text, answer)
        results.append({"编号": row['编号'], "风格是否一致": consistency})
        consistent_count += consistency

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    total = len(data)
    consistency_ratio = consistent_count / total
    print(f"文件的一致比例: {consistency_ratio:.2%}")

if __name__ == "__main__":
    process_file()
