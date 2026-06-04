import re
import time
import json
import os

from google import genai
from google.genai import errors as genai_errors
import pandas as pd
from bs4 import BeautifulSoup

class SentimentEngine:
    def __init__(self, api_key: str, model: str = "gemini-3.1-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.output_dir = "./data"
        os.makedirs(self.output_dir, exist_ok=True)

    def _clean_tweet(self, text: str) -> str:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#(\w+)", r"\1", text)
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _analyze_batch(self, tweets: list[str], max_retries: int = 5) -> list[dict]:
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

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                raw = response.text.strip()
                raw = re.sub(r"```json|```", "", raw).strip()
                return json.loads(raw)

            except json.JSONDecodeError as e:
                print(f"\n   [warn] JSON non valido (tentativo {attempt+1}/{max_retries}): {e}")

            except genai_errors.ClientError as e:
                if e.code == 429:
                    retry_delay = 60
                    try:
                        match = re.search(r"retry in (\d+)", str(e))
                        if match:
                            retry_delay = int(match.group(1)) + 5
                    except Exception:
                        pass
                    print(
                        f"\n   [rate limit] Quota esaurita. "
                        f"Aspetto {retry_delay}s... (tentativo {attempt+1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                elif e.code == 503:
                    retry_delay = 30 * (2 ** attempt)
                    print(
                        f"\n   [503] Server sovraccarico. "
                        f"Aspetto {retry_delay}s... (tentativo {attempt+1}/{max_retries})"
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"\n   [warn] Batch fallito (ClientError {e.code}): {e}")
                    break

            except Exception as e:
                retry_delay = 30 * (2 ** attempt)
                print(
                    f"\n   [warn] Errore inatteso: {e}. "
                    f"Aspetto {retry_delay}s... (tentativo {attempt+1}/{max_retries})"
                )
                time.sleep(retry_delay)

        return [{"id": i + 1, "score": 0.0, "label": "neutral"} for i in range(len(tweets))]

    def analyze(
        self,
        df_social: pd.DataFrame,
        batch_size: int = 100,
        requests_per_minute: int = 14,
        resume: bool = True,
    ) -> pd.DataFrame:
        df = df_social.copy()
        df["text_clean"] = df["text"].apply(self._clean_tweet)
        df = df[df["text_clean"].str.len() > 20].reset_index(drop=True)

        checkpoint_path = os.path.join(self.output_dir, "sentiment_raw_checkpoint.json")

        results_cache: dict[int, dict] = {}
        if resume and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                results_cache = {int(k): v for k, v in json.load(f).items()}
            print(f"   Checkpoint trovato: {len(results_cache)} tweet già analizzati, riprendo da lì.")

        scores = [None] * len(df)
        labels = [None] * len(df)
        delay = 60.0 / requests_per_minute
        total_batches = (len(df) + batch_size - 1) // batch_size

        print(f"   {len(df):,} tweet → {total_batches} batch da {batch_size}")

        for i in range(0, len(df), batch_size):
            batch_num = i // batch_size + 1

            batch_indices = list(range(i, min(i + batch_size, len(df))))
            if all(idx in results_cache for idx in batch_indices):
                for idx in batch_indices:
                    scores[idx] = results_cache[idx]["score"]
                    labels[idx] = results_cache[idx]["label"]
                print(f"   Batch {batch_num}/{total_batches} (da checkpoint)", end="\r")
                continue

            batch_texts = df["text_clean"].iloc[i : i + batch_size].tolist()
            results = self._analyze_batch(batch_texts)

            for r in results:
                idx = i + r["id"] - 1
                if idx < len(df):
                    scores[idx] = r["score"]
                    labels[idx] = r["label"]
                    results_cache[idx] = {"score": r["score"], "label": r["label"]}

            with open(checkpoint_path, "w") as f:
                json.dump(results_cache, f)

            print(f"   Batch {batch_num}/{total_batches} completato", end="\r")
            time.sleep(delay)

        df["sentiment_score"] = scores
        df["sentiment_label"] = labels
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        weekly = df.resample("W").apply(
            lambda g: pd.Series(
                {
                    "sentiment_score_mean": g["sentiment_score"].mean(),
                    "sentiment_score_weighted": (
                        (g["sentiment_score"] * g["user_followers"]).sum()
                        / g["user_followers"].sum()
                        if g["user_followers"].sum() > 0
                        else 0
                    ),
                    "positive_pct": (g["sentiment_label"] == "positive").mean() * 100,
                    "negative_pct": (g["sentiment_label"] == "negative").mean() * 100,
                    "tweet_count": len(g),
                }
            )
        )
        weekly.index.name = "timestamp"

        output_path = os.path.join(self.output_dir, "bitcoin_sentiment.csv")
        weekly.to_csv(output_path)
        print(f"\n   Salvato in {output_path}")

        return weekly