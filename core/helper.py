import re
import json
import itertools
from typing import Tuple
import streamlit as st
import pandas as pd
import zipfile
import os
import uuid
import requests
from urllib.parse import urlparse
from core import constants
from core import file_handler
from core import prompts
from core.llm_helper import LLMInterface
from firecrawl import Firecrawl
from firecrawl.v2.types import ScrapeOptions


def _strip_outer_quotes(s):
    """Remove matching ASCII double quotes from both ends (handles repeated wrappers)."""
    s = (s or "").strip()
    while len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()
    return s


def _normalize_extracted_keywords_payload(processed):
    """Normalize LLM extraction: keyword strings must not start/end with ASCII \"."""
    if isinstance(processed, dict) and "keywords" in processed:
        inner = processed.get("keywords")
        if isinstance(inner, (list, tuple)):
            return {**processed, "keywords": _unique_keywords_preserve(inner)}
        return processed
    if isinstance(processed, list):
        return _unique_keywords_preserve(processed)
    return processed


def _unique_keywords_preserve(items):
    seen = set()
    out = []
    for x in items:
        x = _strip_outer_quotes((x or "").strip())
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def _parse_keyword_lines(text):
    lines = []
    for line in (text or "").splitlines():
        for part in line.split(","):
            part = part.strip()
            if part:
                lines.append(part)
    return _unique_keywords_preserve(lines)


def _keywords_to_text(kw):
    """Turn extraction output into newline-separated text for the editor."""
    if kw is None:
        return ""
    if isinstance(kw, str):
        return _strip_outer_quotes(kw.strip())
    if isinstance(kw, dict):
        inner = kw.get("keywords")
        if isinstance(inner, (list, tuple)):
            parts = [_strip_outer_quotes(str(k).strip()) for k in inner]
            return "\n".join(p for p in parts if p)
        if isinstance(inner, str) and inner.strip():
            return _strip_outer_quotes(inner.strip())
    if isinstance(kw, (list, tuple)):
        parts = [_strip_outer_quotes(str(k).strip()) for k in kw]
        return "\n".join(p for p in parts if p)
    return _strip_outer_quotes(str(kw).strip())


def _current_keywords_from_session():
    if "keywords_editable" in st.session_state:
        return _parse_keyword_lines(st.session_state["keywords_editable"])
    return list(st.session_state.get("keywords") or [])


def _append_suffix(base_query, suffix_tokens):
    """Append optional suffix tokens to the end of a query (not part of combinations)."""
    base_query = " ".join(base_query.split()).strip()
    if not suffix_tokens:
        return base_query
    suf = " ".join(suffix_tokens).strip()
    if not base_query:
        return suf
    return f"{base_query} {suf}".strip()


def _build_search_queries(base_words, combine_pairs, suffix_tokens):
    """
    Build search queries from base keywords only.
    If combine_pairs is False: one query per keyword.
    If combine_pairs is True: unordered pairs only (or the lone keyword if there is only one).
    suffix_tokens are appended to every query.
    """
    base_words = _unique_keywords_preserve(base_words)
    suffix_tokens = _unique_keywords_preserve(suffix_tokens)
    if not base_words:
        return []

    queries = []
    seen_q = set()

    def add_base(q):
        q = _append_suffix(q, suffix_tokens)
        if not q or q.lower() in seen_q:
            return
        seen_q.add(q.lower())
        queries.append(q)

    if not combine_pairs:
        for w in base_words:
            add_base(w)
        return queries

    if len(base_words) >= 2:
        for a, b in itertools.combinations(base_words, 2):
            add_base(f"{a} {b}")
    else:
        add_base(base_words[0])
    return queries


def _keywords_prefill_checkbox_changed():
    if st.session_state.get("prefill_keywords_editable"):
        st.session_state["keywords_editable"] = _keywords_to_text(st.session_state.get("keywords"))
    else:
        st.session_state["keywords_editable"] = ""


def _firecrawl_item_url_markdown(item):
    """URL and markdown from a Firecrawl v2 search item (Document or SearchResultWeb)."""
    markdown = getattr(item, "markdown", None)
    meta = getattr(item, "metadata", None)
    url = None
    if meta is not None:
        url = getattr(meta, "url", None) or getattr(meta, "source_url", None)
    if not url:
        url = getattr(item, "url", None)
    return url, markdown


def _firecrawl_preview_search_to_rows(preview_queries, limit, api_key):
    """
    For each preview query, run Firecrawl search with markdown scrape.
    Returns rows: search_term, search_result_url, search_result_markdown.
    """
    app = Firecrawl(api_key=api_key)
    rows = []
    scrape_opts = ScrapeOptions(formats=["markdown"])
    for term in preview_queries:
        term = (term or "").strip()
        if not term:
            continue
        try:
            result = app.search(term, limit=limit, scrape_options=scrape_opts)
        except Exception as e:
            rows.append(
                {
                    "search_term": term,
                    "search_result_url": "",
                    "search_result_markdown": f"[search error] {e}",
                }
            )
            continue
        web = getattr(result, "web", None) or []
        if not web:
            rows.append(
                {
                    "search_term": term,
                    "search_result_url": "",
                    "search_result_markdown": "",
                }
            )
            continue
        for item in web:
            url, markdown = _firecrawl_item_url_markdown(item)
            if url and not markdown:
                try:
                    doc = app.scrape(url, formats=["markdown"])
                    markdown = getattr(doc, "markdown", None)
                except Exception as e:
                    markdown = f"[scrape error] {e}"
            rows.append(
                {
                    "search_term": term,
                    "search_result_url": url or "",
                    "search_result_markdown": (markdown or "").strip(),
                }
            )
    return rows


def process_search_result(llm: LLMInterface, messy_markdown: str, search_keywords: str):
    """
    Analyzes markdown and returns a tuple: (good_quality, llm_output).
    """
    if not messy_markdown or len(messy_markdown) < 250:
        return False, (
            "Error: Scrape failed or page returned nearly empty content "
            "(likely a bot block or empty state)."
        )

    system_prompt = """You are a content quality filter. Analyze this messy markdown scrape.

TASKS:
1. Determine if this is a real article/blog/post that matches the keywords.
2. If YES: Extract the core article text. Remove all menus, ads, and footers.
3. If NO (e.g., it's a login page, a list of unrelated links, or a '403 Forbidden' message): Identify why.

Return ONLY a JSON object with these keys:
{
  "good_quality": boolean,
  "output": "The cleaned article text if good_quality is true, otherwise the specific reason why it failed."
}"""

    user_content = f"""Search Keywords used: {search_keywords}

MESSY MARKDOWN:
{messy_markdown[:10000]}"""

    try:
        raw = llm.llm_text(
            system_prompt=system_prompt,
            user_content=user_content,
            response_format={"type": "json_object"},
        )
        try:
            res_json = json.loads(raw)
        except json.JSONDecodeError:
            cleaned = re.sub(
                r"^```(?:json)?\n|```$", "", raw.strip(), flags=re.MULTILINE
            )
            res_json = json.loads(cleaned)
        return res_json.get("good_quality", False), res_json.get("output", "Unknown error")
    except Exception as e:
        return False, f"LLM Processing Error: {str(e)}"


def _enrich_firecrawl_rows_with_llm(llm: LLMInterface, rows: list) -> None:
    """Mutates each row dict with good_quality and llm_output from process_search_result."""
    for row in rows:
        md = row.get("search_result_markdown") or ""
        term = row.get("search_term") or ""
        good, out = process_search_result(llm, md, term)
        row["good_quality"] = good
        row["llm_output"] = out


def _truncate_for_llm(text: str, max_chars: int = 32000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated]"


def authorship_verification_llm(llm: LLMInterface, text_a: str, text_b: str) -> Tuple[float, str]:
    """
    Compare Text 1 (source article) vs Text 2 (candidate excerpt).
    Returns (av_score in [0, 1], av_reason).
    """
    text_a = _truncate_for_llm(text_a)
    text_b = _truncate_for_llm(text_b)
    user_content = prompts.authorship_verification_user_prompt.format(
        texta=text_a, textb=text_b
    )
    raw = llm.llm_text(
        system_prompt=prompts.authorship_verification_system_prompt,
        user_content=user_content,
        response_format={"type": "json_object"},
    )
    try:
        res_json = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = re.sub(
            r"^```(?:json)?\n|```$", "", raw.strip(), flags=re.MULTILINE
        )
        res_json = json.loads(cleaned)
    score = res_json.get("av_score", 0)
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    reason = res_json.get("av_reason", "") or ""
    return score, str(reason)


def keyword_combo_and_search_ui(llm: LLMInterface):
    """After extraction: edit keywords, optional pair combos, suffix text, search each online."""
    if "keywords" not in st.session_state:
        return

    st.divider()
    st.subheader("Choose keywords to search online")
    st.caption(
        "Edit base keywords, optionally combine them in pairs, and append extra text to every query."
    )

    if "keywords_editable" not in st.session_state:
        st.session_state["keywords_editable"] = ""

    st.checkbox(
        "Copy all",
        key="prefill_keywords_editable",
        help="When checked, fills the box from your last extraction. Uncheck to clear the box. If left checked, a new extraction updates the box.",
        on_change=_keywords_prefill_checkbox_changed,
    )

    st.text_area(
        "Keywords (one per line)",
        height=150,
        key="keywords_editable",
    )

    st.text_area(
        "Append to each search query (optional)",
        height=80,
        key="extra_keywords_search",
        placeholder="e.g. constraints like site:reddit.com or 'singapore'",
    )

    base_words = _parse_keyword_lines(st.session_state.get("keywords_editable", ""))
    suffix_tokens = _parse_keyword_lines(st.session_state.get("extra_keywords_search", ""))

    if not base_words:
        st.warning(
            "Add at least one keyword"
        )
    else:
        combine_pairs = st.checkbox(
            "Combine keywords in pairs",
            value=False,
            key="combine_keywords_pairs",
        )

        if st.button("See search terms", key="preview_keyword_queries"):
            preview = _build_search_queries(base_words, combine_pairs, suffix_tokens)
            st.session_state["last_query_preview"] = preview

        if st.session_state.get("last_query_preview"):
            for i, q in enumerate(st.session_state["last_query_preview"], 1):
                st.write(f"{i}. {q}")

            st.divider()
            st.subheader("Search online")
            
            fc_limit = st.number_input(
                "Number of search results taken per search term",
                min_value=1,
                max_value=50,
                value=5,
                key="firecrawl_search_limit",
                help="How many web results to request for each preview line.",
            )
            if st.button("Search", key="firecrawl_run_preview"):
                api_key = constants.FIRECRAWL_API_KEY
                if not api_key:
                    st.error("Add FIRECRAWL_API_KEY to .streamlit/secrets.toml.")
                else:
                    preview = list(st.session_state["last_query_preview"])
                    with st.spinner("Searching with Firecrawl..."):
                        rows = _firecrawl_preview_search_to_rows(
                            preview, int(fc_limit), api_key
                        )
                    with st.spinner("Processing markdown with LLM…"):
                        _enrich_firecrawl_rows_with_llm(llm, rows)
                    if not rows:
                        st.warning("No rows to write.")
                        st.session_state.pop("firecrawl_results_df", None)
                    else:
                        df_fc = pd.DataFrame(rows)[
                            [
                                "good_quality",
                                "search_term",
                                "search_result_url",
                                "search_result_markdown",
                                "llm_output",
                            ]
                        ]
                        cache_dir = os.path.join(os.getcwd(), ".streamlit_cache", "firecrawl")
                        os.makedirs(cache_dir, exist_ok=True)
                        out_name = f"firecrawl_search_{uuid.uuid4().hex[:12]}.csv"
                        out_path = os.path.join(cache_dir, out_name)
                        df_fc.to_csv(out_path, index=False, encoding="utf-8-sig")
                        st.session_state["firecrawl_results_df"] = df_fc
                        st.session_state["firecrawl_csv_out_name"] = out_name
                        st.session_state["firecrawl_csv_out_path"] = out_path
                        st.session_state.pop("firecrawl_av_results_df", None)

            df_cached = st.session_state.get("firecrawl_results_df")
            if df_cached is not None:
                df_fc = df_cached
                out_path = st.session_state.get("firecrawl_csv_out_path") or ""
                out_name = st.session_state.get("firecrawl_csv_out_name") or "firecrawl_search.csv"
                st.success(f"Done! {len(df_fc)} row(s) — saved to `{out_path}`")
                fc_event = st.dataframe(
                    df_fc,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="multi-row",
                    key="firecrawl_results_table",
                )
                sel_rows = list(fc_event.selection.rows)
                text_a = (st.session_state.get("extract_article_text") or "").strip()
                if not text_a:
                    st.warning(
                        "Fill **Paste text here** (above) with the source article — it is used as Text 1 for authorship verification."
                    )
                elif sel_rows:
                    st.caption(
                        f"**{len(sel_rows)}** row(s) selected — Text 2 for each row is **llm_output**."
                    )
                else:
                    st.caption(
                        "Select rows to verify if they are written by the same author as the original text."
                    )
                can_av = bool(sel_rows and text_a)
                if st.button(
                    "Run authorship verification",
                    key="firecrawl_av_run",
                    disabled=not can_av,
                ):
                    av_rows = []
                    with st.spinner("Authorship verification (LLM)…"):
                        for idx in sel_rows:
                            row = df_fc.iloc[int(idx)]
                            text_b = str(row.get("llm_output", "") or "")
                            score, reason = authorship_verification_llm(llm, text_a, text_b)
                            r = row.to_dict()
                            r["av_score"] = score
                            r["av_reason"] = reason
                            av_rows.append(r)
                    df_av = pd.DataFrame(av_rows)
                    base_cols = list(df_fc.columns)
                    extra = [c for c in ("av_score", "av_reason") if c not in base_cols]
                    df_av = df_av[base_cols + extra]
                    st.session_state["firecrawl_av_results_df"] = df_av

                if st.session_state.get("firecrawl_av_results_df") is not None:
                    st.subheader("Authorship verification results")
                    st.dataframe(
                        st.session_state["firecrawl_av_results_df"],
                        use_container_width=True,
                    )

                csv_bytes = df_fc.to_csv(index=False, encoding="utf-8-sig").encode(
                    "utf-8-sig"
                )
                st.download_button(
                    label="Download search results (full)",
                    data=csv_bytes,
                    file_name=out_name,
                    mime="text/csv",
                    key="firecrawl_csv_download",
                )


def extract_keywords(llm: LLMInterface, article: str, num_keywords: int):

    keywords_raw = llm.llm_text(
        system_prompt=prompts.keyword_extraction_prompt.format(num_keywords=num_keywords),
        user_content=article
    )
    keywords_processed = llm.post_process_llm_response(
        processing_prompt=prompts.process_into_list_prompt, 
        response_content=keywords_raw
    )
    return _normalize_extracted_keywords_payload(keywords_processed)

def ideate_websites(llm: LLMInterface, article: str, keywords_processed: list):
    keywords_str = ", ".join(keywords_processed)
    websites_raw = llm.llm_text(
        system_prompt=prompts.website_ideation_sys_prompt,
        user_content=prompts.website_ideation_prompt.format(article=article, keyword_list=keywords_str)
    )
    websites_processed = llm.post_process_llm_response(
        processing_prompt=prompts.process_into_list_prompt, 
        response_content=websites_raw
    )

    return websites_processed

def debug_base64_encoding(base64_str):
    import base64
    from io import BytesIO
    from PIL import Image   

    try:
        # Decode base64
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))

        # Show in Streamlit
        st.image(image, caption="Decoded base64 image", use_column_width=True)

    except Exception as e:
        st.error(f"Failed to decode/display base64 image: {e}")

def author_checker(llm: LLMInterface, author: str, base64_str: str) -> dict:
    """Checks if the author is present in the image and extracts content only if so."""
    
    # Step 1: Check if author appears
    presence_response = llm.llm_image(
        prompt=prompts.author_checker_prompt.format(author=author),
        img_base64=base64_str
    ).strip().lower()

    # print(presence_response)

    if presence_response == "yes":
        # Step 2: Extract content if author is present
        try:
            extraction_response = llm.llm_image(
                prompt=prompts.author_content_extraction_prompt.format(author=author),
                img_base64=base64_str
            )
            try:
                parsed = json.loads(extraction_response)
                return parsed if isinstance(parsed, dict) and "content" in parsed else {"content": []}
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")  
            
        except Exception as e:
            print(f"Error during extraction: {e}")
            return {"content": []}
    else:
        # Author not present, return empty content
        return {"content": []}


def extract_author_content(llm: LLMInterface, author: str, base64_dict: dict):
    content_list = []
    total_images = sum(len(lst) for lst in base64_dict.values())
    progress_text = st.empty()
    progress_bar = st.progress(0)

    processed = 0
    for filename, list_of_base64 in base64_dict.items(): 
        for base64_str in list_of_base64:
            try:
                # Testing
                # debug_base64_encoding(base64_str)

                content_raw = author_checker(llm, author, base64_str)
                content_list.append(content_raw)
            except Exception as e:
                st.error(f"Error during inference: {e}")

            # Update progress
            processed += 1
            progress_text.text(f"Processed {processed} of {total_images} images...")
            progress_bar.progress(processed / total_images)

    # Clear progress indicators after completion (optional)
    progress_text.empty()
    progress_bar.empty()

    input_dicts = []
    input_dicts = [s for s in content_list if isinstance(s, dict)]

    merged_content = []
    for d in input_dicts:
        merged_content.extend(d.get("content", []))

    df = pd.DataFrame({
        "author": [author] * len(merged_content),
        "content": merged_content
    })

    # Remove words that start with 'https://'
    df['content'] = df['content'].apply(
        lambda x: ' '.join(word for word in x.split() if not word.startswith("https://")) if isinstance(x, str) else x
    )

    # Drop NaN/None rows
    df = df[df['content'].notna()] 
    # Drop rows where 'content' is empty or just whitespace
    df = df[df['content'].astype(str).str.strip().astype(bool)] 
    # Drop Duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def extract_from_text(llm, text_input=None):
    num_keywords = st.number_input("Number of keywords to generate:", value=5, min_value=1, max_value=20)

    if text_input is None:
        st.text_area("Paste text here:", key="extract_article_text")
        text_input = st.session_state.get("extract_article_text") or ""

    if text_input and num_keywords and re.search(r'\w', text_input):
        if st.button("Extract keywords"):
            with st.spinner("Running inference..."):
                keywords = extract_keywords(llm=llm, article=text_input, num_keywords=num_keywords)
                st.session_state['keywords'] = keywords
                if st.session_state.get("prefill_keywords_editable", False):
                    st.session_state['keywords_editable'] = _keywords_to_text(keywords)
                else:
                    st.session_state['keywords_editable'] = ""
                st.session_state.pop('sites', None)  # Clear previous sites
                st.session_state.pop('last_query_preview', None)
                st.session_state.pop('firecrawl_results_df', None)
                st.session_state.pop('firecrawl_av_results_df', None)
                st.session_state.pop('firecrawl_csv_out_name', None)
                st.session_state.pop('firecrawl_csv_out_path', None)

    if 'keywords' in st.session_state:
        st.subheader("Extracted keywords:")
        st.write(st.session_state['keywords'])

        # if st.button("Ideate websites to search"):
        #     with st.spinner("Running inference..."):
        #         sites = ideate_websites(
        #             llm=llm,
        #             article=text_input,
        #             keywords_processed=_current_keywords_from_session(),
        #         )
        #         st.session_state['sites'] = sites

    # if 'sites' in st.session_state:
    #     st.subheader("Suggested websites to search:")
    #     st.write(st.session_state['sites'])

def extract_content_from_screenshots(llm, screenshot_files=None):
    uploaded_file = None

    if screenshot_files is None:
        uploaded_file = st.file_uploader(
            "Upload screenshot or zip file of screenshots", 
            type=["png", "jpg", "jpeg", ".webp", "pdf", "zip"],
            key='screenshot_upload'
                                         )
    
    author = st.text_area("Author/ Username (to extract what they wrote):")

    # Only run once per author + file combo
    run_key = f"inference_run_{author}_{hash(str(screenshot_files) + str(uploaded_file))}"

    # Check if content already exists
    if run_key not in st.session_state and (uploaded_file or screenshot_files) and author:
        if screenshot_files:
            base64_dict = file_handler.handle_local_files(screenshot_files)
        else:
            base64_dict = file_handler.handle_uploaded_file(uploaded_file)

        with st.spinner("Running inference..."):
            content_df = extract_author_content(llm=llm, author=author, base64_dict=base64_dict)
        
        # Save to session
        st.session_state[run_key] = content_df
    else:
        content_df = st.session_state.get(run_key)

    if content_df is not None:
        st.subheader("Extracted Author's Content")
        st.dataframe(content_df, use_container_width=True)

        # Convert to CSV
        csv_data = content_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Download CSV",
            data=csv_data.encode('utf-8-sig'),
            file_name="extracted_data.csv",
            mime="text/csv"
        )

        return content_df

def get_screenshots(urls, SCREENSHOTMACHINE_API_KEY_LIST, output_folder):
    screenshot_api = "https://api.screenshotmachine.com"
    key_index = 0
    saved_images = []

    for url in urls:
        while key_index < len(SCREENSHOTMACHINE_API_KEY_LIST):
            current_key = SCREENSHOTMACHINE_API_KEY_LIST[key_index]
            screenshot_params = {
                "key": current_key,
                "url": url,
                "dimension": "1024xfull",
                "device": "desktop",
                "format": "png",
            }

            try:
                response = requests.get(screenshot_api, params=screenshot_params)
                if response.status_code == 200:

                    parsed = urlparse(url)
                    domain = parsed.hostname.replace('.', '_') if parsed.hostname else "unknown"
                    path = parsed.path.strip("/").replace("/", "_")
                    if not path:
                        path = "_"
                    filename = f"{domain}_{path}.png"
                    filepath = os.path.join(output_folder, filename)  # Save inside correct folder

                    with open(filepath, "wb") as f:
                        f.write(response.content)

                    saved_images.append(filepath)
                    st.success(f"Screenshot: '{filename}'")

                    break  # move to next URL

                elif response.status_code in [432, 403, 429]:
                    key_index += 1

                else:
                    print(f"Failed to capture {url} - Status: {response.status_code}")
                    break

            except Exception as e:
                print(f"Error with screenshot for {url}: {e}")
                key_index += 1

        if key_index >= len(SCREENSHOTMACHINE_API_KEY_LIST):
            print("All API keys exhausted.")
            break

    return saved_images

# Use a persistent directory in your project (won't be deleted on rerun)
def get_or_create_screenshot_folder():
    base_folder = os.path.join(os.getcwd(), ".streamlit_cache", "screenshots")
    os.makedirs(base_folder, exist_ok=True)
    return base_folder

def get_screenshot_from_urls():
    st.subheader("Enter URLs (one per line)")
    url_input = st.text_area("Paste the URLs here:", height=200)

    if url_input.strip():
        urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
        st.info(f"Processing {len(urls)} URLs...")

        if "screenshot_result" not in st.session_state or st.session_state.get("last_urls") != urls:
            with st.spinner("Taking screenshots..."):
                output_folder = get_or_create_screenshot_folder()

                # Create a unique subfolder to store this session's screenshots
                unique_folder = os.path.join(output_folder, str(uuid.uuid4()))
                os.makedirs(unique_folder, exist_ok=True)

                # Take screenshots (external API or tool)
                image_paths = get_screenshots(
                    urls,
                    constants.SCREENSHOTMACHINE_API_KEY_LIST,
                    output_folder=unique_folder
                )

                if not image_paths:
                    st.error("No screenshots captured.")
                    return

                # Save to session_state to prevent reprocessing
                st.session_state["screenshot_result"] = {
                    "image_paths": image_paths,
                    "zip_path": os.path.join(unique_folder, "screenshots.zip")
                }
                st.session_state["last_urls"] = urls

                # Create ZIP file only once
                with zipfile.ZipFile(st.session_state["screenshot_result"]["zip_path"], "w") as zipf:
                    for image_file in image_paths:
                        zipf.write(image_file, arcname=os.path.basename(image_file))

        # Show download button using saved result
        image_paths = st.session_state["screenshot_result"]["image_paths"]
        zip_path = st.session_state["screenshot_result"]["zip_path"]

        with open(zip_path, "rb") as f:
            zip_bytes = f.read()

        st.success(f"{len(image_paths)} screenshot(s) captured.")
        st.download_button(
            label="📥 Download Screenshots",
            data=zip_bytes,
            file_name="screenshots.zip",
            mime="application/zip"
        )

# import os
# import requests
# import pandas as pd
# from datetime import datetime

# def run_serper_search(search_terms, SERPER_API_KEY_LIST, max_search_results=10):
#     start = datetime.now()
#     key_index = 0
#     results_list = []

#     headers = {
#         "X-API-KEY": SERPER_API_KEY_LIST[key_index],
#         "Content-Type": "application/json"
#     }

#     for term in search_terms:
#         print(f"\nSearching for: {term}")
        
#         while key_index < len(SERPER_API_KEY_LIST):
#             headers["X-API-KEY"] = SERPER_API_KEY_LIST[key_index]
#             payload = {
#                 "q": term,
#                 "num": max_search_results
#             }

#             try:
#                 response = requests.post(
#                     "https://google.serper.dev/search", 
#                     headers=headers, 
#                     json=payload
#                 )

#                 if response.status_code == 429:
#                     print(f"Key #{key_index + 1} hit its limit (429). Switching to next key...")
#                     key_index += 1
#                     continue

#                 data = response.json()
#                 if "error" in data:
#                     print(f"Error: {data['error']}")
#                     key_index += 1
#                     continue

#                 for result in data.get("organic", []):
#                     results_list.append({
#                         "search_term": term,
#                         "search_result_title": result.get("title"),
#                         "search_result_url": result.get("link"),
#                         "search_result_content": result.get("snippet"),
#                         "search_result_score": None,  # Serper doesn't provide a score
#                     })
#                 break  # Success

#             except Exception as e:
#                 print(f"Exception: {e}")
#                 key_index += 1
#                 continue

#         if key_index >= len(SERPER_API_KEY_LIST):
#             print("All Serper API keys exhausted.")
#             break

#     df = pd.DataFrame(results_list)
#     display(df.sample(min(5, len(df))))
#     end = datetime.now()

#     print("Duration:", end - start)
#     print("Final Serper API key used:", key_index + 1)

#     return df
