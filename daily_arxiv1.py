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
import string
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
    A requests session with urllib3 Retry. Note: redirects are controlled per request via allow_redirects.
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
    user_agent="llm-agent-daily/1.1 (+https://github.com/eerstar/LLM-Agent-paper-daily)",
    retries_total=2
)
hf_session = make_session(
    user_agent="llm-agent-daily/1.1 (+https://github.com/eerstar/LLM-Agent-paper-daily)",
    retries_total=2
)
gh_session = make_session(
    user_agent="llm-agent-daily/1.1 (+https://github.com/eerstar/LLM-Agent-paper-daily)",
    retries_total=2
)

# Optional tokens to reduce rate limit risks
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if GITHUB_TOKEN:
    gh_session.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
if HF_TOKEN:
    hf_session.headers.update({"Authorization": f"Bearer {HF_TOKEN}"})


# ---- Helpers ----
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
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Returns (json_dict, error_string). Never raises for 429; handles Retry-After and exponential backoff.
    If allow_redirects=False, any 3xx is treated as error and returned with the redirect location.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout, allow_redirects=allow_redirects)

            # If we forbid redirects, stop here to avoid landing on other sites (e.g. HF trending).
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

            # Other non-2xx
            if resp.status_code < 200 or resp.status_code >= 300:
                return None, f"http:{resp.status_code} url={resp.url}"

            # Ensure JSON-ish
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" not in ctype and "json" not in ctype:
                # Some endpoints may still return JSON without correct header, try parse once.
                try:
                    return resp.json(), None
                except Exception:
                    return None, f"non_json_response content-type={ctype} url={resp.url}"

            return resp.json(), None

        except requests.exceptions.RequestException as e:
            backoff = min(20.0, (2 ** (attempt - 1)))
            logging.warning(f"{purpose} request exception: {e}; backoff={backoff:.1f}s, url={url}")
            sleep_with_jitter(backoff)

    return None, f"failed_after_{max_attempts}_attempts url={url}"


def load_config(config_file: str) -> dict:
    """
    config_file: input config file path
    return: a dict of configuration
    """

    def pretty_filters(**config) -> dict:
        """
        Build arXiv query string per topic:
          (term1 OR "multi word term2" OR term3)
        IMPORTANT: put spaces around OR, otherwise arXiv query parser often fails.
        """
        keywords = dict()

        def quote_if_needed(token: str) -> str:
            token = str(token).strip()
            if not token:
                return token
            # keep hyphenated tokens, quote only when whitespace exists
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


def simple_tokenizer(text: str) -> List[str]:
    if not text:
        return []
    tokens = re.split(r'[^\w-]+', text)
    return [token for token in tokens if token]


# ---- External lookups ----
def get_code_link_github(qword: str, timeout: float = 8.0) -> Optional[str]:
    """
    GitHub search fallback for code repo.
    """
    query = f"{qword}"
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": 1}
    try:
        resp = gh_session.get(GITHUB_SEARCH_API, params=params, timeout=timeout)
        if resp.status_code == 429:
            return None
        if resp.status_code == 403:
            # GitHub rate limit or forbidden
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
    Critically: do NOT follow redirects.
    """
    url = PWC_API_BASE + arxiv_id
    data, err = request_json_with_backoff(
        pwc_session, url,
        timeout=8.0,
        allow_redirects=False,  # 핵심修复：避免跳到 huggingface trending 等页面
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
        hf_session, HF_MODELS_API,
        params=params,
        timeout=8.0,
        allow_redirects=True,
        max_attempts=4,
        purpose="HF"
    )
    if data is None or not isinstance(data, list):
        return []
    paper_infos = []
    q_tokens = set(simple_tokenizer(query.lower()))
    for model in data:
        mid = (model.get("modelId") or "").strip()
        if not mid:
            continue
        mid_tokens = set(simple_tokenizer(mid.lower()))
        # retain if any token overlaps, not only full query containment
        if q_tokens and (q_tokens & mid_tokens):
            paper_infos.append({"title": mid, "url": f"https://huggingface.co/{mid}"})
    return paper_infos


def get_hf_model(paper_title: Optional[str]) -> List[Dict[str, str]]:
    if not paper_title:
        return []
    paper_title = paper_title.strip('\r\n ')
    if not paper_title:
        return []

    # full title search
    infos = search_huggingface_model(paper_title, max_results=1)
    if infos:
        return infos

    # possible model name before ":"
    if ":" in paper_title:
        possible = paper_title.split(":")[0].strip()
        if possible:
            infos = search_huggingface_model(possible, max_results=1)
            if infos:
                return infos

    return []


# ---- Core pipeline ----
def get_daily_papers(topic: str, query: str = "slam", max_results: int = 2):
    """
    @param topic: str
    @param query: str (arXiv query string)
    @return paper_with_code: dict
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

    for result in search_engine.results():
        paper_id = result.get_short_id()
        paper_title = result.title
        paper_url = result.entry_id
        paper_first_author = get_authors(result.authors, first_author=True)
        update_time = result.updated.date()

        logging.info(f"Time = {update_time} title = {paper_title} author = {paper_first_author}")

        # 2108.09112v1 -> 2108.09112
        ver_pos = paper_id.find('v')
        paper_key = paper_id if ver_pos == -1 else paper_id[:ver_pos]

        paper_url = ARXIV_ABS_BASE + 'abs/' + paper_key

        repo_url = None
        repo_err = None

        # 1) Try PWC official repo (fixed: no redirects)
        repo_url, repo_err = fetch_pwc_official_repo(paper_key)
        if repo_err:
            logging.debug(f"PWC repo not available for {paper_key}: {repo_err}")

        # 2) Fallback to GitHub search if no repo
        if not repo_url:
            repo_url = get_code_link_github(paper_title) or get_code_link_github(paper_key)

        hf_model_infos = get_hf_model(paper_title)
        hf_model_url = hf_model_infos[0]["url"] if hf_model_infos else ""

        if repo_url:
            content_repo_placeholder = f"**[link]({repo_url})**"
            content_web_repo_placeholder = f", Code: **[{repo_url}]({repo_url})**"
        else:
            content_repo_placeholder = "null"
            content_web_repo_placeholder = ""

        if hf_model_url:
            content_hf_model_placeholder = f"**[link]({hf_model_url})**"
            content_web_hf_model_placeholder = f", Model: **[{hf_model_url}]({hf_model_url})**"
        else:
            content_hf_model_placeholder = "null"
            content_web_hf_model_placeholder = ""

        content[paper_key] = (
            f"|**{update_time}**|**{paper_title}**|{paper_first_author} et.al.|"
            f"[{paper_key}]({paper_url})|{content_repo_placeholder}|{content_hf_model_placeholder}|\n"
        )
        content_to_web[paper_key] = (
            f"- {update_time}, **{paper_title}**, {paper_first_author} et.al., "
            f"Paper: [{paper_url}]({paper_url})"
            f"{content_web_repo_placeholder}{content_web_hf_model_placeholder}\n"
        )

    return {topic: content}, {topic: content_to_web}


def update_paper_links(filename: str):
    """
    weekly update paper links in json file
    """

    def parse_arxiv_string(s: str):
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        arxiv_id = parts[4].strip()
        code = parts[5].strip()
        arxiv_id = re.sub(r'v\d+', '', arxiv_id)
        return date, title, authors, arxiv_id, code

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        m = {} if not content else json.loads(content)

    json_data = m.copy()

    for keywords, v in json_data.items():
        logging.info(f'keywords = {keywords}')
        for paper_id, contents in v.items():
            contents = str(contents)
            update_time, paper_title, paper_first_author, paper_url, code_url = parse_arxiv_string(contents)

            # normalize line
            contents = "|{}|{}|{}|{}|{}|\n".format(
                update_time, paper_title, paper_first_author, paper_url, code_url
            )
            json_data[keywords][paper_id] = str(contents)

            valid_link = False if '|null|' in contents else True
            if valid_link:
                continue

            # Try again: PWC official (no redirects) then fallback GH
            repo_url, _ = fetch_pwc_official_repo(paper_id)
            if not repo_url:
                repo_url = get_code_link_github(paper_title) or get_code_link_github(paper_id)

            if repo_url:
                new_cont = contents.replace('|null|', f'|**[link]({repo_url})**|')
                json_data[keywords][paper_id] = str(new_cont)
                logging.info(f'ID = {paper_id}, updated repo link')

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f)


def update_json_file(filename: str, data_dict: list):
    """
    daily update json file using data_dict
    """
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        m = {} if not content else json.loads(content)

    json_data = m.copy()

    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_data, f)


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

    DateNow = datetime.date.today()
    DateNow = str(DateNow).replace('-', '.')

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()
        data = {} if not content else json.loads(content)

    # clean file
    with open(md_filename, "w+", encoding="utf-8") as f:
        pass

    with open(md_filename, "a+", encoding="utf-8") as f:
        if (use_title is True) and (to_web is True):
            f.write("---\nlayout: default\n---\n\n")

        if show_badge is True:
            f.write(f"[![Contributors][contributors-shield]][contributors-url]\n")
            f.write(f"[![Forks][forks-shield]][forks-url]\n")
            f.write(f"[![Stargazers][stars-shield]][stars-url]\n")
            f.write(f"[![Issues][issues-shield]][issues-url]\n\n")

        if use_title is True:
            f.write("## Updated on " + DateNow + "\n")
        else:
            f.write("> Updated on " + DateNow + "\n")

        if use_tc is True:
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

            if use_title is True:
                if to_web is False:
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
                top_info = f"#Updated on {DateNow}"
                top_info = top_info.replace(' ', '-').replace('.', '')
                f.write(f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n")

        if show_badge is True:
            # NOTE: This part is kept as-is from original (it points to cnlinxi repo).
            # You may want to change to your repo.
            f.write(
                f"[contributors-shield]: https://img.shields.io/github/"
                f"contributors/cnlinxi/llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write(f"[contributors-url]: https://github.com/cnlinxi/llm-arxiv-daily/graphs/contributors\n")
            f.write(
                f"[forks-shield]: https://img.shields.io/github/forks/cnlinxi/"
                f"llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write(f"[forks-url]: https://github.com/cnlinxi/llm-arxiv-daily/network/members\n")
            f.write(
                f"[stars-shield]: https://img.shields.io/github/stars/cnlinxi/"
                f"llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write(f"[stars-url]: https://github.com/cnlinxi/llm-arxiv-daily/stargazers\n")
            f.write(
                f"[issues-shield]: https://img.shields.io/github/issues/cnlinxi/"
                f"llm-arxiv-daily.svg?style=for-the-badge\n"
            )
            f.write(f"[issues-url]: https://github.com/cnlinxi/llm-arxiv-daily/issues\n\n")

    logging.info(f"{task} finished")


def demo(**config):
    data_collector = []
    data_collector_web = []

    keywords = config['kv']
    max_results = config['max_results']
    publish_readme = config['publish_readme']
    publish_gitpage = config['publish_gitpage']
    publish_wechat = config['publish_wechat']
    show_badge = config['show_badge']

    b_update = config['update_paper_links']
    logging.info(f'Update Paper Link = {b_update}')

    if config['update_paper_links'] is False:
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

    config = load_config(args.config_path)
    config = {**config, 'update_paper_links': args.update_paper_links}
    demo(**config)
