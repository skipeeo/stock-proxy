import os, urllib.parse, requests, json
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
HEADERS = {
    "X-Naver-Client-Id": NAVER_CLIENT_ID,
    "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
}

def call_naver(endpoint, query, display):
    url = (
        f"https://openapi.naver.com/v1/search/{endpoint}.json?"
        f"query={urllib.parse.quote(query)}&display={display}"
    )
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json().get("items", [])

@app.route("/analyze-stock", methods=["GET"])
def analyze_stock():
    q = request.args.get("query")
    num_news  = int(request.args.get("num_news", 100))
    num_blogs = int(request.args.get("num_blogs", 50))

    if not q:
        return jsonify({"error": "query parameter required"}), 400

    news  = call_naver("news",  q, num_news)
    blogs = call_naver("blog",  q, num_blogs)

    return jsonify({
        "queried_term": q,
        "total_news_hits": len(news),
        "total_blog_hits": len(blogs),
        "news_articles": news,
        "blog_posts": blogs
    })

if __name__ == "__main__":
    app.run(debug=True, port=8000)
