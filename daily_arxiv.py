import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Tuple, Dict, Any, List

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)

# ---- URLs ----
PWC_API_BASE = "https://paperswithcode.com/api/v0/papers/"  # /{arxiv_id}
GITHUB_SEARCH_API = "https://api.github.com/search/repositories"
ARXIV_ABS_BASE = "http://arxiv.org/"
HF_MODELS_API = "https://huggingface.co/api/models"


# ---- Sessions ----
def make_session(user_agent: str, retries_total: int = 3) -> requests.Session:
    """
    requests Session + urllib3 Retry.
    Note: redirects are controlled per request via allow_redirects.
    """
    s = requests.Session()
    retry = Retry(
        total=retries_total,
        connect=retries_total,
        read=retries_total,
        status=retries_total,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": user_agent})
    return s


pwc_session = make_session(
    user_agent="llm-agent-daily/1.3 (+https://github.com/eerstar/LLM-Agent-paper-daily)",
    retries_total=2
)
hf_session = make_session(
    user_agent="llm-agent-daily/1.3 (+https://github.com/eerstar/LLM-Agent-paper-daily)",
    retries_total=2
)
gh_session = make_session(
    user_agent="llm-agent-daily/1.3 (+https://github.com/eerstar/LLM-Agent-paper-daily)",
    retries_total=2
)

# Optional tokens to reduce rate limits
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if GITHUB_TOKEN:
    gh_session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
if HF_TOKEN:
    hf_session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})


# ---- Generic helpers ----
def sleep_with_jitter(base_seconds: float) -> None:
    time.sleep(base_seconds + random.random() * 0.6)


def safe_int(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = s.strip()
    return int(s) if s.isdigit() else None


def request_json_with_backoff(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    timeout: float = 8.0,
    allow_redirects: bool = True,
    max_attempts: int = 5,
    purpose: str = ""
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Return (json, err). Handles 429 with Retry-After / exponential backoff.
    If allow_redirects=False, any 3xx returns redirect error to prevent landing on other sites.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, allow_redirects=allow_redirects)

            if not allow_redirects and resp.status_code in (301, 302, 303, 307, 308):
                loc = resp.headers.get("Location", "")
                return None, f"redirect:{resp.status_code} -> {loc}"

            if resp.status_code == 429:
                ra = safe_int(resp.headers.get("Retry-After"))
                if ra is not None:
                    logging.warning(f"{purpose} 429 rate-limited, Retry-After={ra}s, url={resp.url}")
                    time.sleep(ra)
                else:
                    backoff = min(30.0, (2 ** (attempt - 1)))
                    logging.warning(f"{purpose} 429 rate-limited, backoff={backoff:.1f}s, url={resp.url}")
                    sleep_with_jitter(backoff)
                continue

            if resp.status_code < 200 or resp.status_code >= 300:
                return None, f"http:{resp.status_code} url={resp.url}"

            try:
                return resp.json(), None
            except Exception:
                ctype = (resp.headers.get("Content-Type") or "").lower()
                return None, f"json_decode_failed content-type={ctype} url={resp.url}"

        except requests.exceptions.RequestException as e:
            backoff = min(20.0, (2 ** (attempt - 1)))
            logging.warning(f"{purpose} request exception: {e}; backoff={backoff:.1f}s, url={url}")
            sleep_with_jitter(backoff)

    return None, f"failed_after_{max_attempts}_attempts url={url}"


# ---- Config ----
def load_config(config_file: str) -> dict:
    """
    Load YAML config and build arXiv query strings.
    """
    def pretty_filters(**config) -> dict:
        """
        Build per-topic query:
          (term1 OR "multi word term2" OR term3)
        """
        keywords = dict()

        def quote_if_needed(token: str) -> str:
            token = str(token).strip()
            if not token:
                return token
            return f"\"{token}\"" if len(token.split()) > 1 else token

        def parse_filters(filters: list) -> str:
            parts = [quote_if_needed(x) for x in (filters or []) if str(x).strip()]
            if not parts:
                return ""
            return "(" + " OR ".join(parts) + ")"

        for k, v in config.get('keywords', {}).items():
            keywords[k] = parse_filters(v.get('filters', []))
        return keywords

    with open(config_file, 'r', encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['kv'] = pretty_filters(**config)
    logging.info(f'config loaded, topics={list(config.get("kv", {}).keys())}')
    return config


# ---- ArXiv data helpers ----
def get_authors(authors, first_author: bool = False) -> str:
    if not authors:
        return ""
    if not first_author:
        return ", ".join(str(author) for author in authors)
    return str(authors[0])


def sort_papers(papers: dict) -> dict:
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output


# ---- Tokenizer ----
def simple_tokenizer(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.split(r'[^\w-]+', text)
    return [t for t in tokens if t]


# ---- External lookups ----
def get_code_link_github(qword: str, timeout: float = 8.0) -> Optional[str]:
    """
    GitHub search fallback.
    """
    params = {"q": f"{qword}", "sort": "stars", "order": "desc", "per_page": 1}
    try:
        resp = gh_session.get(GITHUB_SEARCH_API, params=params, timeout=timeout)
        if resp.status_code in (403, 429):
            return None
        resp.raise_for_status()
        results = resp.json()
        if results.get("total_count", 0) > 0:
            return results["items"][0].get("html_url")
        return None
    except Exception:
        return None


def fetch_pwc_official_repo(arxiv_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Query PWC API for official repo URL.
    CRITICAL: do NOT follow redirects (avoid being redirected to HF pages).
    """
    url = PWC_API_BASE + arxiv_id
    data, err = request_json_with_backoff(
        pwc_session,
        url,
        timeout=8.0,
        allow_redirects=False,   # 核心修复点：禁止重定向
        max_attempts=4,
        purpose="PWC"
    )
    if data is None:
        return None, err
    try:
        off = data.get("official")
        if off and isinstance(off, dict):
            return off.get("url"), None
    except Exception:
        pass
    return None, "no_official_repo"


def search_huggingface_model(query: str, max_results: int = 2) -> List[Dict[str, str]]:
    params = {"search": query, "limit": max_results}
    data, err = request_json_with_backoff(
        hf_session,
        HF_MODELS_API,
        params=params,
        timeout=8.0,
        allow_redirects=True,
        max_attempts=4,
        purpose="HF"
    )
    if data is None or not isinstance(data, list):
        return []

    q_tokens = set(simple_tokenizer(query.lower()))
    out = []
    for model in data:
        mid = (model.get("modelId") or "").strip()
        if not mid:
            continue
        mid_tokens = set(simple_tokenizer(mid.lower()))
        if q_tokens and (q_tokens & mid_tokens):
            out.append({"title": mid, "url": f"https://huggingface.co/{mid}"})
    return out


def get_hf_model(paper_title: Optional[str]) -> List[Dict[str, str]]:
    if not paper_title:
        return []
    paper_title = paper_title.strip('\r\n ')
    if not paper_title:
        return []

    infos = search_huggingface_model(paper_title, max_results=1)
    if infos:
        return infos

    if ":" in paper_title:
        possible = paper_title.split(":")[0].strip()
        if possible:
            infos = search_huggingface_model(possible, max_results=1)
            if infos:
                return infos
    return []


# ---- Daily papers ----
def get_daily_papers(topic: str, query: str = "slam", max_results: int = 2):
    """
    Return:
      data: {topic: {paper_id: md_row}}
      data_web: {topic: {paper_id: bullet_row}}
    """
    content = dict()
    content_to_web = dict()

    if not query:
        return {topic: content}, {topic: content_to_web}

    search_engine = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    from scripts.harvest import make_client, result_row

    for result in make_client().results(search_engine):
        paper_id = result.get_short_id()
        paper_title = result.title
        paper_first_author = get_authors(result.authors, first_author=True)
        update_time = result.published.date()

        logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

        ver_pos = paper_id.find('v')
        paper_key = paper_id if ver_pos == -1 else paper_id[:ver_pos]

        paper_url = ARXIV_ABS_BASE + 'abs/' + paper_key

        _, content[paper_key] = result_row(result)
        content_to_web[paper_key] = (
            f"- {update_time}, **{paper_title}**, {paper_first_author} et.al., "
            f"Paper: [{paper_url}]({paper_url})\n"
        )

    return {topic: content}, {topic: content_to_web}


# ---- Markdown row parsing (6 columns) ----
def parse_md_row_6cols(row: str) -> Optional[Dict[str, str]]:
    """
    Parse row like:
      |**date**|**title**|authors|[id](pdf_url)|code|model|
    """
    if not row:
        return None
    row = str(row).strip()

    pattern = re.compile(
        r'^\|\*\*(?P<date>.*?)\*\*\|'
        r'\*\*(?P<title>.*?)\*\*\|'
        r'(?P<authors>.*?)\|'
        r'\[(?P<arxiv_id>\d+\.\d+)\]\((?P<pdf_url>[^)]+)\)\|'
        r'(?P<code>.*?)\|'
        r'(?P<model>.*?)\|\s*$'
    )
    m = pattern.match(row)
    if not m:
        return None

    d = m.groupdict()
    for k in d:
        d[k] = (d[k] or "").strip()
    return d


def format_md_row_6cols(date, title, authors, arxiv_id, pdf_url, code_cell, model_cell) -> str:
    return (
        f"|**{date}**|**{title}**|{authors}|"
        f"[{arxiv_id}]({pdf_url})|{code_cell}|{model_cell}|\n"
    )


# ---- Bad-row salvage helpers ----
def _norm_row_text(row: str) -> str:
    row = str(row)
    row = row.replace("\r", " ").replace("\n", " ")
    row = re.sub(r"\s+", " ", row).strip()
    if (row.startswith('"') and row.endswith('"')) or (row.startswith("'") and row.endswith("'")):
        row = row[1:-1].strip()
    return row


def _extract_first_url_after(pattern: str, text: str) -> Optional[str]:
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def salvage_md_row_6cols(paper_id: str, bad_row: str) -> Optional[Dict[str, str]]:
    """
    抢救坏行：
    - 用 paper_id 重建 arxiv_id/pdf_url
    - 尽力恢复 date/title/authors
    - code/model：能抠则抠，否则置 null
    """
    if not paper_id:
        return None

    s = _norm_row_text(bad_row)

    # date: first YYYY-MM-DD
    date = None
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", s)
    if m:
        date = m.group(1)

    # title: prefer structure |**date**|**title**|
    title = None
    m = re.search(r"\|\*\*20\d{2}-\d{2}-\d{2}\*\*\|\*\*(.+?)\*\*\|", s)
    if m:
        title = m.group(1).strip()
    else:
        # fallback: pick a bold segment that is not date/link-ish
        bolds = re.findall(r"\*\*(.+?)\*\*", s)
        bolds = [b.strip() for b in bolds if b.strip()]
        for b in bolds:
            if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", b):
                continue
            if "http://" in b or "https://" in b or "link](" in b:
                continue
            title = b
            break

    # authors: prefer |**date**|**title**|AUTHORS|
    authors = None
    m = re.search(r"\|\*\*20\d{2}-\d{2}-\d{2}\*\*\|\*\*.+?\*\*\|([^|]+)\|", s)
    if m:
        authors = m.group(1).strip()
    else:
        # fallback around et.al
        m = re.search(r"([A-Za-z].{0,120}?et\.?\.?\s*al\.?)", s, flags=re.IGNORECASE)
        if m:
            authors = m.group(1).strip()

    # code/model: preserve existing **[link](...)** if present
    all_links = re.findall(r"\*\*\[link\]\(([^)]+)\)\*\*", s)
    code_url = all_links[0].strip() if len(all_links) >= 1 else None
    model_url = all_links[1].strip() if len(all_links) >= 2 else None

    code_cell = f"**[link]({code_url})**" if code_url else "null"
    model_cell = f"**[link]({model_url})**" if model_url else "null"

    # rebuild from paper_id
    arxiv_id = paper_id
    pdf_url = ARXIV_ABS_BASE + "abs/" + paper_id

    if not (date or title or authors):
        return None

    return {
        "date": date or "unknown",
        "title": title or f"(recovered) {paper_id}",
        "authors": authors or "unknown et.al.",
        "arxiv_id": arxiv_id,
        "pdf_url": pdf_url,
        "code": code_cell,
        "model": model_cell,
    }


# ---- Update code links safely (keep 6 cols + salvage) ----
def update_paper_links(filename: str):
    """
    Weekly update paper links in json file.
    Keep 6 columns intact; only fill Code when it's null.
    If a row is corrupted/unparsable, salvage it by rebuilding PDF from paper_id and recovering title/authors best-effort.
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        json_data = {} if not content else json.loads(content)

    for topic, papers in json_data.items():
        logging.info(f'keywords = {topic}')
        for paper_id, row in list(papers.items()):
            row = str(row)

            parsed = parse_md_row_6cols(row)

            # 兜底：抢救坏行
            if not parsed:
                salvaged = salvage_md_row_6cols(paper_id, row)
                if not salvaged:
                    logging.warning(
                        f"Skip unparsable row (salvage failed): paper_id={paper_id}, head={_norm_row_text(row)[:140]!r}"
                    )
                    continue
                logging.warning(f"Salvaged corrupted row: paper_id={paper_id}")
                parsed = salvaged

            date = parsed["date"]
            title = parsed["title"]
            authors = parsed["authors"]
            arxiv_id_in_row = parsed["arxiv_id"]
            pdf_url = parsed["pdf_url"]
            code_cell = parsed["code"]
            model_cell = parsed["model"]

            arxiv_id_effective = paper_id or arxiv_id_in_row

            # PDF 链接兜底：如果 row 里 pdf_url 已经不可信，就用 paper_id 重建
            if (not pdf_url) or ("arxiv.org/abs/" not in pdf_url):
                pdf_url = ARXIV_ABS_BASE + "abs/" + arxiv_id_effective

            # 仅当 code 为 null 时补链
            if (code_cell or "").strip().lower() == "null":
                repo_url, _ = fetch_pwc_official_repo(arxiv_id_effective)
                if not repo_url:
                    repo_url = get_code_link_github(title) or get_code_link_github(arxiv_id_effective)
                if repo_url:
                    code_cell = f"**[link]({repo_url})**"

            papers[paper_id] = format_md_row_6cols(
                date=date,
                title=title,
                authors=authors,
                arxiv_id=arxiv_id_effective,
                pdf_url=pdf_url,
                code_cell=code_cell if code_cell else "null",
                model_cell=model_cell if model_cell else "null",
            )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f)


# ---- JSON update ----
def update_json_file(filename: str, data_dict: list):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        m = {} if not content else json.loads(content)

    json_data = m.copy()
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword in json_data:
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f)


# ---- JSON -> Markdown ----
def json_to_md(filename,
               md_filename,
               task='',
               to_web=False,
               use_title=True,
               use_tc=True,
               show_badge=True,
               use_b2t=True):

    def pretty_math(s: str) -> str:
        match = re.search(r"\$.*\$", s)
        if match is None:
            return s
        math_start, math_end = match.span()
        space_trail = space_leading = ''
        if s[:math_start] and s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]:
            space_trail = ' '
        if math_end < len(s) and s[math_end:][0] != ' ' and '*' != s[math_end:][0]:
            space_leading = ' '
        return s[:math_start] + f'{space_trail}${match.group()[1:-1].strip()}${space_leading}' + s[math_end:]

    DateNow = str(datetime.date.today()).replace('-', '.')

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        data = {} if not content else json.loads(content)

    with open(md_filename, "w+", encoding="utf-8") as f:
        pass

    with open(md_filename, "a+", encoding="utf-8") as f:
        if use_title and to_web:
            f.write("---\nlayout: default\n---\n\n")

        if show_badge:
            f.write("[![Contributors][contributors-shield]][contributors-url]\n")
            f.write("[![Forks][forks-shield]][forks-url]\n")
            f.write("[![Stargazers][stars-shield]][stars-url]\n")
            f.write("[![Issues][issues-shield]][issues-url]\n\n")

        f.write(("## " if use_title else "> ") + "Updated on " + DateNow + "\n")

        if use_tc:
            f.write("<details>\n")
            f.write("  <summary>Table of Contents</summary>\n")
            f.write("  <ol>\n")
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                kw = keyword.replace(' ', '-')
                f.write(f"    <li><a href=#{kw.lower()}>{keyword}</a></li>\n")
            f.write("  </ol>\n")
            f.write("</details>\n\n")

        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue

            f.write(f"## {keyword}\n\n")

            if use_title:
                if not to_web:
                    f.write("|Publish Date|Title|Authors|PDF|Code|Model|\n|---|---|---|---|---|---|\n")
                else:
                    f.write("| Publish Date | Title | Authors | PDF | Code | Model |\n")
                    f.write("|:---------|:-----------------------|:---------|:------|:------|:------|\n")

            day_content = sort_papers(day_content)
            for _, v in day_content.items():
                if v is not None:
                    f.write(pretty_math(v))
            f.write("\n")

            if use_b2t:
                top_info = f"#Updated on {DateNow}".replace(' ', '-').replace('.', '')
                f.write(f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n")

        if show_badge:
            # NOTE: 原仓库遗留，建议你改成自己的 repo（eerstar/LLM-Agent-paper-daily）
            f.write(
                "[contributors-shield]: https://img.shields.io/github/"
                "contributors/cnlinxi/llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write("[contributors-url]: https://github.com/cnlinxi/llm-arxiv-daily/graphs/contributors\n")
            f.write(
                "[forks-shield]: https://img.shields.io/github/forks/cnlinxi/"
                "llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write("[forks-url]: https://github.com/cnlinxi/llm-arxiv-daily/network/members\n")
            f.write(
                "[stars-shield]: https://img.shields.io/github/stars/cnlinxi/"
                "llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write("[stars-url]: https://github.com/cnlinxi/llm-arxiv-daily/stargazers\n")
            f.write(
                "[issues-shield]: https://img.shields.io/github/issues/cnlinxi/"
                "llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write("[issues-url]: https://github.com/cnlinxi/llm-arxiv-daily/issues\n\n")

    logging.info(f"{task} finished")


# ---- Main demo ----
def demo(**config):
    data_collector = []
    data_collector_web = []

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']

    logging.info(f'Update Paper Link = {config["update_paper_links"]}')

    if not config['update_paper_links']:
        logging.info("GET daily papers begin")
        for topic, keyword in keywords.items():
            if not keyword:
                continue
            logging.info(f"Keyword: {topic}, Query: {keyword}")
            try:
                data, data_web = get_daily_papers(topic, query=keyword, max_results=max_results)
            except arxiv.HTTPError as e:
                logging.error(f"arxiv HTTP error when fetching {topic}: {e}")
                continue
            data_collector.append(data)
            data_collector_web.append(data_web)
        logging.info("GET daily papers end")

    # 1) README
    if publish_readme:
        json_file = config['json_readme_path']
        md_file = config['md_readme_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='Update Readme', show_badge=show_badge)

    # 2) GitHub Pages
    if publish_gitpage:
        json_file = config['json_gitpage_path']
        md_file = config['md_gitpage_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(json_file, md_file, task='Update GitPage', to_web=True,
                   show_badge=show_badge, use_tc=False, use_b2t=False)

    # 3) WeChat
    if publish_wechat:
        json_file = config['json_wechat_path']
        md_file = config['md_wechat_path']
        if config['update_paper_links']:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector_web)
        json_to_md(json_file, md_file, task='Update Wechat', to_web=False,
                   use_title=False, show_badge=show_badge)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.yaml', help='configuration file path')
    parser.add_argument('--update_paper_links', default=False, action="store_true",
                        help='whether to update paper links etc.')
    args = parser.parse_args()

    if args.update_paper_links:
        from scripts.enrich_links import main
        raise SystemExit(main(["--config-path", args.config_path]))
    from scripts.harvest import main
    raise SystemExit(main(["--config-path", args.config_path]))
