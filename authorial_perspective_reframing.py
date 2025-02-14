# -*- coding: utf-8 -*-

import os
import json
from openai import OpenAI


client = OpenAI(
    api_key="your-api-key",
    base_url="https://api.20a.top/v1"
)

def generate_summary(article):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        top_p=0.9,
        messages=[
            {
                "role": "system",
                "content": f"请用现代中文（白话文），以第三人称视角复述以下鲁迅文章内容，并准确传达原文信息。要求如下：\n1、观点归属明确：用“鲁迅指出”“鲁迅认为”“鲁迅说”“鲁迅批评”“鲁迅讽刺”或者其他合适的句式标明观点归属为鲁迅。每段复述必须至少一次明确观点归属为鲁迅，有必要的时候必须标明观点归属为鲁迅。\n2、逐句复述：原文的每一句话都需逐句使用流畅的现代中文（白话文）以第三人称视角复述，保持逻辑清晰，确保不遗漏信息，不添加个人解读。\n3、必须使用流畅的现代中文（白话文）表达，不使用文言文句式。\n4、尽可能在复述文章中多出现鲁迅的名字。\n5、输出时仅输出复述文章即可。\n以下是示例输入内容：\n一\n中华民国十五年三月二十五日，就是国立北京女子师范大学为十八日在段祺瑞执政府前遇害的刘和珍杨德群两君开追悼会的那一天，我独在礼堂外徘徊，遇见程君，前来问我道，“先生可曾为刘和珍写了一点什么没有？”我说“没有”。她就正告我，“先生还是写一点罢；刘和珍生前就很爱看先生的文章。”\n这是我知道的，凡我所编辑的期刊，大概是因为往往有始无终之故罢，销行一向就甚为寥落，然而在这样的生活艰难中，毅然预定了《莽原》全年的就有她。我也早觉得有写一点东西的必要了，这虽然于死者毫不相干，但在生者，却大抵只能如此而已。倘使我能够相信真有所谓“在天之灵”，那自然可以得到更大的安慰，——但是，现在，却只能如此而已。\n可是我实在无话可说。我只觉得所住的并非人间。四十多个青年的血，洋溢在我的周围，使我艰于呼吸视听，那里还能有什么言语？长歌当哭，是必须在痛定之后的。而此后几个所谓学者文人的阴险的论调，尤使我觉得悲哀。我已经出离愤怒了。我将深味这非人间的浓黑的悲凉；以我的最大哀痛显示于非人间，使它们快意于我的苦痛，就将这作为后死者的菲薄的祭品，奉献于逝者的灵前。\n以下是示例输出：\n一\n鲁迅回忆了中华民国十五年三月二十五日的情景，那一天，北京女子师范大学为在段祺瑞执政府前遇害的刘和珍和杨德群两位学生举行了追悼会。鲁迅指出，他当时独自在礼堂外徘徊，遇到了程君。程君问他是否为刘和珍写了文章，鲁迅坦言自己尚未动笔。程君直言建议他写点什么，因为刘和珍生前很喜欢鲁迅的文章。\n鲁迅提到，虽然他所编辑的刊物销路一直不好，但刘和珍却毅然在艰难的生活中订阅了全年《莽原》，这让他意识到有必要为她写点什么。鲁迅认为，这虽然对死者无益，但对活着的人却是仅能做到的慰藉。他坦率地表示，如果他能相信“在天之灵”的存在，或许能得到更大的安慰，但现实中，他只能以此表达悼念。\n然而，鲁迅坦承，他实在不知道该说些什么。他感到周围充满了四十多位青年流淌的鲜血，这让他呼吸困难、视听受阻。他自觉此刻身处的已不是“人间”。鲁迅认为，长歌当哭需要等到痛苦稍稍平复之后，而此时此刻，他只感受到深沉的悲凉。鲁迅批评，一些所谓学者和文人的阴险言论更是加深了他的悲哀，使他超越愤怒，沉浸在非人间的黑暗与悲痛之中。\n鲁迅进一步表示，他希望通过自己的最大哀痛来呈现这非人间的深沉悲凉，让那些幸存者快意于他的苦痛。他认为，这或许是他能够献给逝者最微薄的祭品，用以表达自己的深切悼念。"
            },
            {   
                "role": "user", 
                "content": f"{article['text']}"
            }
        ]
        
    )

    text = response.choices[0].message.content.strip()

    return {
        "title": article['title'],
        "text": text
    }

def process_articles(input_filename, output_filename):
    
    failed_articles = []

    with open(input_filename, "r", encoding="utf-8") as file:
        articles = json.load(file)

    os.makedirs("json_output", exist_ok=True)
    output_path = os.path.join("json_output", output_filename)

    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                all_summaries = json.load(f)
            except json.JSONDecodeError:
                all_summaries = []
    else:
        all_summaries = []

    processed_titles = {item['title'] for item in all_summaries}

    for article in articles:
        if article['title'] in processed_titles:
            print(f"Article '{article['title']}' already processed, skipping.")
            continue

        try:
            summary_data = generate_summary(article)
            if summary_data:
                all_summaries.append(summary_data)
                
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_summaries, f, ensure_ascii=False, indent=4)
                print(f"Processed article '{article['title']}' and saved summary.")
        except Exception as e:
            print(f"Error processing article '{article['title']}': {e}")
            failed_articles.append({
                "title": article['title'],
                "error": str(e)
            })

    if failed_articles:
        failed_log_path = os.path.join("json_output", "failed_articles.json")
        with open(failed_log_path, "w", encoding="utf-8") as f:
            json.dump(failed_articles, f, ensure_ascii=False, indent=4)
        print(f"Failed articles logged to {failed_log_path}")


def main():
    input_filename = "luxun_essay.json"
    output_filename = "reframe.json"
    process_articles(input_filename, output_filename)


if __name__ == "__main__":
    main()
