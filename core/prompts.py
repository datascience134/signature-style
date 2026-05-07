keyword_extraction_prompt = '''
You are an OSINT investigator.

Your task is to extract *distinctive authorial fingerprints* from the text below—unusual or identifying keywords and short phrases that can help locate other articles by the same person via web search.

Return exactly {num_keywords} short (1–3 word) distinctive, **author-specific or author-typical** terms.

Prioritize:
- Colloquial language, slang, Singlish, or stylistic quirks
- Emotionally charged or opinionated vocabulary
- Repeated niche phrases or metaphors used by the author
- Rare idioms or invented phrasing
- Region- or subculture-specific acronyms, names, or references

Avoid:
- Technical terms or domain jargon (e.g. “real GDP per capita”, “neoliberalism”) unless used in an unusual way
- Concepts or terms commonly found on Wikipedia or academic websites
- Generic, broad, or emotionally neutral words
- Full sentences or explanations

Think: What would this author say that *few others would?*

Output format:
Return a valid JSON object in the following format:

{{
  "keywords": [
    "keyword1",
    "keyword2",
    "keyword3"
  ]
}}



ARTICLE:
'''

chi_keyword_extraction_prompt = '''
  你是一名开源情报调查专家。

  你的任务是从下面的文本中提取独特的作者行文指纹——那些不寻常或具有辨识度的关键词和简短短语，能够帮助通过网络搜索定位同一作者的其他文章。

  请严格返回 {num_keywords} 个简短（1–3 字）且具有作者个性或典型风格的词语。

  优先提取：

  口语化表达、俚语、方言词（如中文方言、网络黑话、地缘特色用语）

  带有情绪色彩或主观判断的词汇

  作者反复使用的独特比喻、小众短语

  罕见的成语、自创表达或非常规搭配

  特定地区或亚文化圈内的缩略词、人名或指代

  避免提取：

  专业术语或行业行话（例如“GDP增长率”、“新自由主义”），除非用法非常独特

  在维基百科或学术网站上常见的概念或词汇

  泛泛而谈、情感中立的通用词

  完整句子或解释说明

  思考逻辑：这个作者说的什么话，很少会有其他人这样说？

  输出格式：
  返回一个合法的 JSON 对象，格式如下：

  {{
  "keywords": [
  "关键词1",
  "关键词2",
  "关键词3"
  ]
  }}

  文章内容如下：
'''


process_into_list_prompt = '''
You will be given a string that contains a list of items. The list may include newline characters, bullet points (e.g. `-` or `•`), or inconsistent spacing.

Your task is to extract and return a clean **Python list** of strings, with each item stripped of any bullet characters and whitespace.

Return only valid JSON output. For example, given the input:

"- abc  \n - def ghi"

Return:
["abc", "def ghi"]

Input string:
{{input_string}}
'''

chi_process_into_list_prompt = '''
你将收到一个包含列表项的字符串。该列表可能包含换行符、项目符号（例如 `-` 或 `•`）或不一致的空格。

你的任务是从中提取并返回一个干净的 **Python 列表**，其中每个元素都是字符串，
并且要去除任何项目符号和前后空白字符。

只返回合法的 JSON 输出。例如，给定输入：

"- 苹果 \n - 香蕉"

应返回：
["苹果", "香蕉"]

输入字符串：
{{input_string}}
'''

authorship_verification_system_prompt = '''
You are an expert in stylometry and authorship analysis. The user will give you two texts and a detailed task.
After you complete the task step by step, you MUST respond with ONLY a valid JSON object (no markdown fences) with exactly these keys:
- "av_score": a number from 0 to 1 inclusive (0 = low confidence same author, 1 = high confidence same author)
- "av_reason": a concise string summarizing the main evidence for the score
'''

chi_authorship_verification_system_prompt = '''
你是一名笔体学与作者身份分析专家。用户会给你两段文本以及一个详细的任务说明。

在你按步骤完成任务后，你必须只返回一个合法的 JSON 对象（不要使用 markdown 代码块标记），该对象必须包含以下两个字段：
- "av_score": 一个 0 到 1 之间的数字（包含 0 和 1），0 表示低置信度认为同一作者，1 表示高置信度认为同一作者
- "av_reason": 一个简短的字符串，总结得出该分数的主要证据
'''

authorship_verification_user_prompt = '''
Task: On a scale of 0 to 1, with 0 indicating low confidence and 1 indicating high confidence, please provide a general assessment of the likelihood that Text 1 and Text 2 were written by the same author. Your answer should reflect a moderate level of strictness in scoring. Here are some relevant variables to this problem.
1. punctuation style(e.g. hyphen, brackets, colon, comma, parenthesis, quotation mark)
2. special characters style, capitalization style(e.g. Continuous capitalization, capitalizing certain words)
3. acronyms and abbreviations(e.g. Usage of acronyms such as OMG, Abbreviations without punctuation marks such as Mr Rochester vs. Mr. Rochester,Unusual abbreviations such as def vs. definitely)
4. writing style
5. expressions and idioms
6. tone and mood
7. sentence structure
8. any other relevant aspect
First step: Understand the problem, extracting relevant variables and devise a plan to solve the problem. Then, carry out the plan and solve the problem step by step. Finally, show the confidence score.
Text 1: {texta}
Text 2: {textb}
'''

chi_authorship_verification_user_prompt = '''
任务：请在 0 到 1 的范围内，给出一个整体评估，判断文本一与文本二是否由同一作者所写。0 表示低置信度，1 表示高置信度。你的打分应体现适中的严格程度。以下是该任务中一些相关的分析维度：

1. 标点风格（例如：破折号、括号、冒号、逗号、省略号、引号的使用习惯，全角/半角标点的偏好）
2. 特殊字符与数字用法（例如：特殊符号的使用频率、数字格式偏好、中英文标点混用情况）
3. 缩写与简称风格（例如：网络用语缩略如“然并卵”、“u1s1”，人名或机构名的习惯简称）
4. 行文风格（例如：段落长短、口语化 vs 书面化程度、对话体 vs 叙事体）
5. 表达方式与习语（例如：成语使用频率、方言词汇、固定搭配的习惯用法）
6. 语气体裁（例如：讽刺、热情、冷静、调侃、批评等情感基调）
7. 句式结构（例如：长短句交替习惯、排比句使用、倒装结构、“被”字句/“把”字句偏好）
8. 重复用词或句式（例如：特定语气词“啊/哦/嗯”的使用、开头结尾的固定模式）

第一步：理解问题，提取相关分析维度，并制定解决问题的计划。然后，执行计划，逐步分析。最后，给出置信度分数。

文本一：{texta}
文本二：{textb}
'''

article_extraction_system_prompt = """
You are a content quality filter. Analyze this messy markdown scrape.

TASKS:
1. Determine if this is a real article/blog/post that matches the keywords.
2. If YES: Extract the core article text. Remove all menus, ads, and footers.
3. If NO (e.g., it's a login page, a list of unrelated links, or a '403 Forbidden' message): Identify why.

Return ONLY a JSON object with these keys:
{
  "good_quality": boolean,
  "output": "The cleaned article text if good_quality is true, otherwise the specific reason why it failed."
}"""

chi_article_extraction_system_prompt = """
你是一名内容质量过滤器。请分析以下这份格式杂乱的 Markdown 抓取内容。

任务：
1. 判断这是否为一篇与关键词匹配的真实文章、博文或帖子。
2. 如果是：提取核心文章正文。删除所有菜单、广告和页脚内容。
3. 如果不是（例如：是登录页面、不相关的链接列表，或“403 Forbidden”等信息）：说明失败原因。

只返回一个 JSON 对象，包含以下字段：
{
  "good_quality": true/false,
  "output": "如果 good_quality 为 true，则输出清理后的文章正文；否则，输出具体的失败原因"
}
"""