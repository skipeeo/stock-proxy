# coding: utf-8
"""Google Colab script to collect company data and store in Google Drive."""
import os
import json
import datetime
import requests

try:
    from google.colab import drive  # type: ignore
    IN_COLAB = True
except ImportError:  # pragma: no cover
    IN_COLAB = False

import yfinance as yf

# ----------------------------- Helper Functions -----------------------------

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
DART_API_KEY = os.getenv("DART_API_KEY")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID or "",
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET or "",
}


def call_naver(endpoint: str, query: str, display: int = 100):
    """Query Naver open API for news or blogs."""
    url = (
        f"https://openapi.naver.com/v1/search/{endpoint}.json?"
        f"query={requests.utils.quote(query)}&display={display}"
    )
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json().get("items", [])


def fetch_company_name_from_ticker(ticker: str) -> str:
    """Return company name for ticker using yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return info.get("shortName") or info.get("longName") or ticker
    except Exception:
        return ticker


def fetch_dart_filings(corp_name: str, years: int = 5):
    """Fetch DART filings for the past `years` years."""
    if not DART_API_KEY:
        return []
    end = datetime.date.today()
    start = end.replace(year=end.year - years)
    url = (
        f"https://opendart.fss.or.kr/api/list.json?crtfc_key={DART_API_KEY}"
        f"&corp_name={requests.utils.quote(corp_name)}&bgn_de={start:%Y%m%d}&end_de={end:%Y%m%d}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json().get("list", [])


def fetch_edgar_filings(ticker: str, years: int = 5):
    """Fetch EDGAR filings for US companies via SEC API."""
    try:
        info = yf.Ticker(ticker).info
        cik = str(info.get("cik"))
        if not cik:
            return []
    except Exception:
        return []

    end = datetime.date.today()
    start = end.replace(year=end.year - years)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    r = requests.get(submissions_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200:
        return []
    data = r.json()
    filings = []
    for filing in data.get("filings", {}).get("recent", {}).get("form", []):
        filings.append(filing)
    return filings


def fetch_rapidapi_concall(ticker: str):
    """Fetch conference call transcripts from RapidAPI."""
    if not RAPIDAPI_KEY:
        return []
    url = "https://seeking-alpha.p.rapidapi.com/transcripts/v2/get-list"
    params = {"id": ticker}
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "seeking-alpha.p.rapidapi.com",
    }
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        return []
    return r.json().get("data", [])


def save_to_drive(company: str, data: dict):
    """Save data dict to Google Drive under 기업분석자료/<company>/<date>."""
    if IN_COLAB:
        drive.mount("/content/drive")
        base = "/content/drive/MyDrive/기업분석자료"
    else:
        base = os.path.join(os.getcwd(), "기업분석자료")
    date_folder = datetime.date.today().isoformat()
    folder = os.path.join(base, company, date_folder)
    os.makedirs(folder, exist_ok=True)
    for key, value in data.items():
        path = os.path.join(folder, f"{key}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
    return folder


# ----------------------------- Main Routine -----------------------------

def main():
    ticker_or_name = input("기업명 또는 티커를 입력하세요: ")
    if len(ticker_or_name) <= 6 and ticker_or_name.isalpha():
        company_name = fetch_company_name_from_ticker(ticker_or_name)
        yn = input(f"{company_name} 이 기업이 맞나요? (Y/N): ").strip().lower()
        if yn != "y":
            company_name = input("정확한 기업명을 입력해주세요: ")
            ticker = ticker_or_name
        else:
            ticker = ticker_or_name
    else:
        company_name = ticker_or_name
        ticker = ticker_or_name

    print("뉴스와 공시자료를 수집합니다. 잠시만 기다려주세요...")
    news = call_naver("news", company_name, 100)
    blogs = call_naver("blog", company_name, 50)
    dart = fetch_dart_filings(company_name)
    edgar = fetch_edgar_filings(ticker)
    transcripts = fetch_rapidapi_concall(ticker)

    saved_folder = save_to_drive(company_name, {
        "news": news,
        "blogs": blogs,
        "dart": dart,
        "edgar": edgar,
        "transcripts": transcripts,
    })
    print(f"자료가 저장되었습니다: {saved_folder}")


if __name__ == "__main__":
    main()
