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


process_into_list_prompt = '''
You will be given a string that contains a list of items. The list may include newline characters, bullet points (e.g. `-` or `•`), or inconsistent spacing.

Your task is to extract and return a clean **Python list** of strings, with each item stripped of any bullet characters and whitespace.

Return only valid JSON output. For example, given the input:

'- abc  \n - def ghi'

Return:
["abc", "def ghi"]

Input string:
{{input_string}}
'''


website_ideation_sys_prompt = '''
You are a digital OSINT analyst.

Given an article and a list of keywords that reflect the author's unique style or language, your task is to suggest relevant websites or platforms where searching these keywords might help find **more writings by the same author**.

Output format:
Return a valid JSON object in the following format:

{{
  "sites": [
    "site:reddit.com",
    "site:medium.com",
    "site:twitter.com"
  ]
}}

Focus on:
- Blogs, personal domains, and writing platforms (e.g., Medium, Wordpress, Substack)
- Forums and community sites (e.g., Reddit, HardwareZone)
- Social platforms (e.g., Twitter/X, Facebook)
- Niche communities depending on the article content

Do not explain. Just return the JSON object.
'''


website_ideation_prompt = '''
ARTICLE:

{article}

KEYWORDS:

{keyword_list}
'''

author_checker_prompt = '''
You are an assistant helping verify whether a person authored any content in a screenshot.

Target author: "{author}"

Follow these steps:

1. Read the image text carefully.
2. Check if "{author}" appears **exactly** as the author of any content — such as:
   - A visible username or handle next to a post
   - A byline, reply tag, or attribution explicitly showing authorship

3. Do **not** guess or infer based on similar names. Only confirm if it **exactly matches** "{author}".

Return only:
- "yes" — if "{author}" is clearly shown as an author
- "no" — if not

No extra explanation.
'''



author_content_extraction_prompt = '''
You are a data extraction assistant helping an OSINT analyst identify and structure content written by a specific individual from a screenshot containing text.

Your goal is to extract only the content explicitly authored by the person named "{author}". The text may come from forums, blogs, social media, comment sections, or other platforms.

Follow these steps carefully:

1. **Check if the name "{author}" appears anywhere in the text** — as a username, handle, author tag, or attribution label. Do not infer or assume — only proceed if there is clear textual evidence of this name.

2. If "{author}" is found, identify all the text **directly attributed** to them. This may include:
   - Posts or comments where "{author}" is clearly shown as the author
   - Replies or messages labeled as coming from "{author}"

3. Extract only their content, **excluding anything written by others**.

4. If no content from "{author}" is found, return an **empty list**.

Always return your output as a **valid JSON object** in this exact format:

**If content is found:**
{{
  "content": [
    "First piece of content written by {author}.",
    "Second piece of content written by {author}."
  ]
}}

**If no content is found:**
{{
  "content": []
}}
'''


authorship_verification_system_prompt = '''
You are an expert in stylometry and authorship analysis. The user will give you two texts and a detailed task.
After you complete the task step by step, you MUST respond with ONLY a valid JSON object (no markdown fences) with exactly these keys:
- "av_score": a number from 0 to 1 inclusive (0 = low confidence same author, 1 = high confidence same author)
- "av_reason": a concise string summarizing the main evidence for the score
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

