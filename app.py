#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Altın (veya başka bir varlık) haberleri için duygu analizi aracı.
- Google News RSS'ten haber çeker
- Başlık (ve opsiyonel içerik) üzerinden VADER veya FinBERT ile sentiment çıkarır
- Özet istatistikleri ve istenirse JSON/CSV çıktısı üretir

Kullanım:
    python app.py --queries "gold market" "gold price" --per-query 10 --model finbert --include-content --out-json out.json
"""

import sys
import json
import csv
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote

import requests
import feedparser
from bs4 import BeautifulSoup

# VADER varsayılan, FinBERT isteğe bağlı
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

# FinBERT'i isteğe bağlı ve "lazy" yükleyeceğiz
_FINBERT_BUNDLE = {
    "tokenizer": None,
    "model": None,
    "labels": ["Positive", "Negative", "Neutral"],
    "loaded": False,
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

@dataclass
class Article:
    title: str
    link: str
    published: str
    content: str


# ----------------------------
# Haber Toplama
# ----------------------------
def fetch_news(query: str, limit: int = 10, sleep_sec: float = 0.0) -> List[Article]:
    """
    Google News RSS üzerinden haberleri çeker.
    """
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}"
    feed = feedparser.parse(rss_url)
    items = feed.entries[:limit]

    articles: List[Article] = []
    for it in items:
        title = getattr(it, "title", "").strip()
        link = getattr(it, "link", "").strip()
        published = getattr(it, "published", "").strip()
        articles.append(Article(title=title, link=link, published=published, content=""))

    if sleep_sec > 0:
        time.sleep(sleep_sec)

    return articles


def fetch_article_content(url: str, timeout: int = 12) -> str:
    """
    Haber sayfasına istek atıp <p> etiketlerinden metni toplar.
    Content paywall/JS render gerektiren sitelerde boş dönebilir.
    """
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
        return content.strip()
    except Exception:
        return ""


# ----------------------------
# Sentiment: VADER
# ----------------------------
class VaderAnalyzer:
    def __init__(self):
        if not _HAS_VADER:
            raise RuntimeError("vaderSentiment kurulu değil. `pip install vaderSentiment`")
        self._analyzer = SentimentIntensityAnalyzer()

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Dönüş: (label, score)
        label: Positive/Negative/Neutral
        score: compound skoru [-1, 1]
        """
        if not text.strip():
            return "Neutral", 0.0
        d = self._analyzer.polarity_scores(text)
        compound = d.get("compound", 0.0)
        if compound > 0.05:
            return "Positive", compound
        elif compound < -0.05:
            return "Negative", compound
        else:
            return "Neutral", compound


# ----------------------------
# Sentiment: FinBERT
# ----------------------------
def _load_finbert():
    """
    FinBERT modelini lazy-load eder.
    Not: torch/transformers kurulu olmalı.
    """
    if _FINBERT_BUNDLE["loaded"]:
        return

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "FinBERT için `transformers` ve `torch` gereklidir. "
            "Kurulum: pip install torch transformers"
        ) from e

    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    _FINBERT_BUNDLE["tokenizer"] = tokenizer
    _FINBERT_BUNDLE["model"] = model
    _FINBERT_BUNDLE["loaded"] = True


class FinBertAnalyzer:
    def __init__(self, max_length: int = 512):
        _load_finbert()
        self.tokenizer = _FINBERT_BUNDLE["tokenizer"]
        self.model = _FINBERT_BUNDLE["model"]
        self.labels = _FINBERT_BUNDLE["labels"]
        self.max_len = max_length

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Dönüş: (label, confidence)
        label: Positive/Negative/Neutral
        confidence: [0,1]
        """
        import torch
        import numpy as np

        if not text.strip():
            return "Neutral", 0.0

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        idx = int(probs.argmax())
        return self.labels[idx], float(probs[idx])


# ----------------------------
# Yardımcılar
# ----------------------------
def analyze_articles(
    articles: List[Article],
    model: str = "vader",
    use_content: bool = False,
) -> List[Dict]:
    """
    Haberleri seçilen modele göre analiz eder ve dict listesi döndürür.
    """
    if model.lower() == "finbert":
        analyzer = FinBertAnalyzer()
    else:
        analyzer = VaderAnalyzer()

    results = []
    for art in articles:
        text = art.title
        if use_content and art.content:
            # Başlık + içerik birlikte de analiz edilebilir; çok uzun metinlerde
            # FinBERT 512 token sınırı nedeniyle truncation devreye girer.
            text = f"{art.title}. {art.content}"

        label, score = analyzer.predict(text)
        results.append(
            {
                "title": art.title,
                "link": art.link,
                "published": art.published,
                "sentiment": label,
                "score": score,
                "used_content": use_content and bool(art.content),
            }
        )
    return results


def summarize(results: List[Dict]) -> Dict[str, float]:
    """
    Sonuçları sayıp yüzde dağılımını döndürür.
    """
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for r in results:
        lbl = r.get("sentiment", "Neutral")
        if lbl in counts:
            counts[lbl] += 1
        else:
            counts["Neutral"] += 1

    total = max(len(results), 1)
    return {
        "total": total,
        "Positive": round(100.0 * counts["Positive"] / total, 2),
        "Negative": round(100.0 * counts["Negative"] / total, 2),
        "Neutral": round(100.0 * counts["Neutral"] / total, 2),
    }


def write_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        # boş CSV
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Google News + (VADER/FinBERT) ile duygu analizi",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--queries",
        nargs="+",
        required=True,
        help='Arama sorguları (ör: "gold market" "gold price")',
    )
    p.add_argument("--per-query", type=int, default=10, help="Sorgu başına haber sayısı")
    p.add_argument("--sleep", type=float, default=0.0, help="Sorgular arasında bekleme (sn)")
    p.add_argument(
        "--include-content",
        action="store_true",
        help="Başlığa ek olarak makale içeriğini de analiz et",
    )
    p.add_argument(
        "--fetch-content",
        action="store_true",
        help="Makale içeriklerini gerçekten indir (performans maliyeti var)",
    )
    p.add_argument(
        "--model",
        choices=["vader", "finbert"],
        default="vader",
        help="Kullanılacak sentiment modeli",
    )
    p.add_argument("--out-json", type=str, default=None, help="Detay sonuçları JSON olarak yaz")
    p.add_argument("--out-csv", type=str, default=None, help="Detay sonuçları CSV olarak yaz")
    p.add_argument("--summary-json", type=str, default=None, help="Özet yüzdeleri JSON olarak yaz")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    all_articles: List[Article] = []

    for q in args.queries:
        print(f"🔎 Fetching: {q}")
        arts = fetch_news(q, limit=args.per_query, sleep_sec=args.sleep)
        # içerik indir
        if args.fetch_content:
            for a in arts:
                a.content = fetch_article_content(a.link)
                # çok agresif olmamak için kısa bir bekleme iyi olur
                time.sleep(0.2)
        all_articles.extend(arts)

    print(f"📰 Toplam haber: {len(all_articles)}")

    results = analyze_articles(
        articles=all_articles,
        model=args.model,
        use_content=args.include_content,
    )
    summary = summarize(results)

    print("\n--- Market Sentiment Summary ---")
    print(f"Total: {summary['total']}")
    print(f"Positive: {summary['Positive']}%")
    print(f"Negative: {summary['Negative']}%")
    print(f"Neutral : {summary['Neutral']}%")

    if args.out_json:
        write_json(args.out_json, {"results": results})
        print(f"💾 JSON yazıldı: {args.out_json}")

    if args.out_csv:
        write_csv(args.out_csv, results)
        print(f"💾 CSV yazıldı: {args.out_csv}")

    if args.summary_json:
        write_json(args.summary_json, summary)
        print(f"💾 Özet JSON yazıldı: {args.summary_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
