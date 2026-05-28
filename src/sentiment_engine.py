# src/sentiment_engine.py

import re
import time
import json
import google.generativeai as genai
import pandas as pd
from bs4 import BeautifulSoup


class SentimentEngine:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def _clean_tweet(self, text: str) -> str:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _analyze_batch(self, tweets: list[str]) -> list[dict]:
        numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tweets))
        prompt = f"""
                Analyze the sentiment of each tweet below about Bitcoin/crypto markets.
                For each tweet return ONLY a JSON array with objects containing:
                - "id": the tweet number (integer)
                - "score": float from -1.0 (extremely negative) to +1.0 (extremely positive)
                - "label": one of "positive", "neutral", "negative"

                Tweets: {numbered}

                Return ONLY the JSON array, no other text.
        """
        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            return json.loads(raw)
        except Exception as e:
            print(f"   [warn] Batch fallito: {e}")
            return [{"id": i+1, "score": 0.0, "label": "neutral"} for i in range(len(tweets))]

    def analyze(
        self,
        df_social: pd.DataFrame,
        batch_size: int = 100,
        requests_per_minute: int = 14,
    ) -> pd.DataFrame:

        df = df_social.copy()
        df["text_clean"] = df["text"].apply(self._clean_tweet)
        df = df[df["text_clean"].str.len() > 20].reset_index(drop=True)

        scores  = [None] * len(df)
        labels  = [None] * len(df)
        delay   = 60.0 / requests_per_minute

        total_batches = (len(df) + batch_size - 1) // batch_size
        print(f"   {len(df):,} tweet → {total_batches} batch da {batch_size}")

        for i in range(0, len(df), batch_size):
            batch_texts  = df["text_clean"].iloc[i:i+batch_size].tolist()
            batch_num    = i // batch_size + 1
            results      = self._analyze_batch(batch_texts)

            for r in results:
                idx = i + r["id"] - 1
                if idx < len(df):
                    scores[idx] = r["score"]
                    labels[idx] = r["label"]

            print(f"   Batch {batch_num}/{total_batches} completato", end="\r")
            time.sleep(delay)

        df["sentiment_score"] = scores
        df["sentiment_label"] = labels

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        weekly = df.resample("W").apply(lambda g: pd.Series({
            "sentiment_score_mean":     g["sentiment_score"].mean(),
            "sentiment_score_weighted": (
                (g["sentiment_score"] * g["user_followers"]).sum()
                / g["user_followers"].sum()
                if g["user_followers"].sum() > 0 else 0
            ),
            "positive_pct": (g["sentiment_label"] == "positive").mean() * 100,
            "negative_pct": (g["sentiment_label"] == "negative").mean() * 100,
            "tweet_count":  len(g),
        }))
        weekly.index.name = "timestamp"
        return weekly