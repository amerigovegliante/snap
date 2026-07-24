import re
import time
import json
import os

import numpy as np
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

        return [{"id": i + 1, "score": 0.0, "label": "neutral", "failed": True} for i in range(len(tweets))]

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
                    if r.get("failed"):
                        scores[idx] = 0.0
                        labels[idx] = "neutral"
                    else:
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
                        else float("nan")
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

    def compute_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(window).mean()
        loss  = (-delta.clip(upper=0)).rolling(window).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def compute_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        ema_fast    = prices.ewm(span=fast,   adjust=False).mean()
        ema_slow    = prices.ewm(span=slow,   adjust=False).mean()
        macd_line   = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({
            "macd":        macd_line,
            "macd_signal": signal_line,
            "macd_hist":   macd_line - signal_line,
        })

    def compute_sharpe(
        self,
        returns: pd.Series,
        window: int = 4,
        risk_free: float = 0.0,
    ) -> pd.Series:
        excess = returns - risk_free / 52
        return (
            excess.rolling(window).mean()
            / excess.rolling(window).std()
            * np.sqrt(52)
        )

    def compute_technical_features(self, df_market: pd.DataFrame) -> pd.DataFrame:
        close   = df_market["Close"]
        returns = close.pct_change().rename("return_w")

        feats = pd.DataFrame(index=df_market.index)
        feats["log_return"]   = np.log1p(returns)
        feats["volatility_w"] = returns.rolling(4).std()
        feats["rsi_14"]       = self.compute_rsi(close)
        feats[["macd", "macd_signal", "macd_hist"]] = self.compute_macd(close)
        feats["sharpe_4w"]    = self.compute_sharpe(returns)
        feats["volume_norm"]  = (
            df_market["Volume"] / df_market["Volume"].rolling(4).mean()
        )
        return feats

    def compute_sentiment_features(
        self,
        df_social: pd.DataFrame,
        freq: str = "W",
        **analyze_kwargs,
    ) -> pd.DataFrame:
        weekly = self.analyze(df_social, **analyze_kwargs)

        full_idx = pd.date_range(
            start=weekly.index.min(),
            end=weekly.index.max(),
            freq=freq,
        )
        weekly = weekly.reindex(full_idx)
        weekly["has_data"]   = (weekly["tweet_count"] > 0).astype(int)
        weekly["tweet_count"] = weekly["tweet_count"].fillna(0)

        cols_to_interpolate = [
            "sentiment_score_mean", "sentiment_score_weighted",
            "positive_pct", "negative_pct",
        ]
        # FIX leakage: l'interpolazione lineare con limit_direction="both" riempiva i buchi
        # usando anche osservazioni FUTURE (guardava avanti nel tempo). Sostituito con un
        # forward-fill puro (causale, limitato a poche settimane), coerente con quanto già
        # fatto manualmente in train.ipynb per gli stessi dati.
        weekly[cols_to_interpolate] = (
            weekly[cols_to_interpolate].ffill(limit=4).fillna(0)
        )
        weekly.index.name = "timestamp"

        output_path = os.path.join(self.output_dir, "bitcoin_sentiment.csv")
        weekly.to_csv(output_path)
        print(f"   CSV aggiornato con gap filling → {output_path}")

        return weekly

    def compute_sentiment_momentum(
        self,
        df_sentiment: pd.DataFrame,
        window: int = 4,
        col: str = "sentiment_score_mean",
    ) -> pd.Series:
        """Differenza tra il sentiment della settimana corrente e la media mobile
        delle `window` settimane precedenti (esclusa quella corrente).

        Causale per costruzione: `.shift(1)` esclude la settimana corrente dalla
        baseline prima di applicare `.rolling(window)`, quindi ogni valore usa
        solo dati strettamente passati rispetto alla settimana a cui si riferisce.
        """
        baseline = df_sentiment[col].shift(1).rolling(window).mean()
        return (df_sentiment[col] - baseline).rename(f"{col}_momentum_{window}w")

    def compute_energy_features(self, df_energy: pd.DataFrame) -> pd.DataFrame:
        feats = pd.DataFrame(index=df_energy.index)
        feats["cost_per_btc"]       = df_energy["cost_per_btc_usd"]
        feats["cost_per_btc_delta"] = df_energy["cost_per_btc_usd"].pct_change()
        if "hash_rate_ths" in df_energy.columns:
            feats["hash_rate_norm"] = (
                df_energy["hash_rate_ths"]
                / df_energy["hash_rate_ths"].rolling(4).mean()
            )
        return feats

    def build_feature_matrix(
        self,
        df_market:  pd.DataFrame,
        df_social:  pd.DataFrame,
        df_energy:  pd.DataFrame,
        **analyze_kwargs,
    ) -> pd.DataFrame:
        tech   = self.compute_technical_features(df_market)
        social = self.compute_sentiment_features(df_social, **analyze_kwargs)
        energy = self.compute_energy_features(df_energy)

        matrix = tech.join(social, how="left").join(energy, how="left")
        matrix = matrix.ffill(limit=1)
        matrix = matrix.iloc[:-1]
        return matrix