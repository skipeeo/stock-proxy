# Stock Proxy Data Collection

This repository contains a Flask app and a Colab-compatible script to collect
news articles, regulatory filings, and conference call transcripts for a given
company.

## Colab Usage

`colab_collect_data.py` can be run directly in Google Colab.
It will:

1. Ask for a company name or ticker symbol.
2. Confirm the company when a ticker is used.
3. Collect data from:
   - Naver Open API for news and blog posts.
   - DART (Korean) filings.
   - EDGAR (US) filings.
   - Seeking Alpha transcripts via RapidAPI.
4. Save all fetched data to your Google Drive under
   `기업분석자료/<회사명>/<수집일자>/`.

API keys for Naver, DART, and RapidAPI should be provided as environment
variables: `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`, `DART_API_KEY`, and
`RAPIDAPI_KEY`.
