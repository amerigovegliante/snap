import os
from datetime import datetime
import kagglehub as kh
import pandas as pd
import requests
import yfinance as yf


class DataIngestor:
    def __init__(self, start_date: str, end_date: str, frequency: str = "W"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.frequency = frequency
        self.output_dir = "./data"

        os.makedirs(self.output_dir, exist_ok=True)

    def fetch_market_data(self, ticker: str = "BTC-USD") -> pd.DataFrame:
        data = yf.download(ticker, start=self.start_date, end=self.end_date)

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        aggregated_data = data.resample(self.frequency).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })

        aggregated_data.index.name = 'timestamp'

        output_path = os.path.join(self.output_dir, f"{ticker.lower()}_market_data.csv")
        aggregated_data.to_csv(output_path, index=True)

        return aggregated_data

    def fetch_mining_metrics(self) -> pd.DataFrame:
        urls = {
            "hash_rate": "https://api.blockchain.info/charts/hash-rate?timespan=all&sampled=true&format=json",
            "difficulty": "https://api.blockchain.info/charts/difficulty?timespan=all&sampled=true&format=json"
        }

        dfs = {}

        for metric_name, url in urls.items():
            response = requests.get(url)
            if response.status_code != 200:
                raise RuntimeError(f"Impossibile scaricare la metrica requested. Errore: {response.status_code}")
            data_json = response.json()

            raw_values = data_json.get("values", [])

            df_metric = pd.DataFrame(raw_values)
            df_metric.columns = ["timestamp", metric_name]

            df_metric["timestamp"] = pd.to_datetime(df_metric["timestamp"], unit="s")
            df_metric.set_index("timestamp", inplace=True)

            dfs[metric_name] = df_metric

        data = dfs["hash_rate"].join(dfs["difficulty"], how="outer")
        data = data.loc[self.start_date : self.end_date]

        aggregated_data = data.resample(self.frequency).mean()
        aggregated_data.index.name = 'timestamp'

        output_path = os.path.join(self.output_dir, "bitcoin_mining_metrics.csv")
        aggregated_data.to_csv(output_path, index=True)

        return aggregated_data

    def fetch_social_feed(self, min_followers: int = 1000) -> pd.DataFrame:
        all_dfs = []
        try:
            cache_path = kh.dataset_download("alaix14/bitcoin-tweets-20160101-to-20190329")
            files = os.listdir(cache_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                full_csv_path = os.path.join(cache_path, csv_files[0])
                df_raw = pd.read_csv(full_csv_path, sep=';', on_bad_lines='skip', engine='python')
                df_x1 = pd.DataFrame()
                df_x1["timestamp"]      = pd.to_datetime(df_raw["timestamp"], errors='coerce')
                df_x1["source"]         = "X_Twitter"
                df_x1["text"]           = df_raw["text"].astype(str)
                df_x1["user_followers"] = pd.to_numeric(df_raw["likes"], errors='coerce')
                all_dfs.append(df_x1)
        except Exception as e:
            print(f"Errore nel Dataset: {e}")

        if not all_dfs:
            return pd.DataFrame(columns=["timestamp", "source", "text", "user_followers"])

        data = pd.concat(all_dfs, ignore_index=True)
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')
        data = data.dropna(subset=["timestamp", "user_followers"])

        try:
            if data['timestamp'].dt.tz is not None:
                data['timestamp'] = data['timestamp'].dt.tz_convert(None)
        except AttributeError:
            pass

        data = data.loc[(data["timestamp"] >= self.start_date) & (data["timestamp"] <= self.end_date)]

        before = len(data)
        data = data.loc[data["user_followers"] >= min_followers]
        print(f"   Filtro like >= {min_followers}: {before:,} → {len(data):,} tweet")

        data = data.sort_values(by="timestamp").reset_index(drop=True)

        output_path = os.path.join(self.output_dir, "bitcoin_social_dataset_unified.csv")
        data.to_csv(output_path, index=False)

        return data

    def fetch_reddit_feed(self, min_score: int = 0) -> pd.DataFrame:
        all_dfs = []
        try:
            cache_path = kh.dataset_download("jerryfanelli/reddit-comments-containing-bitcoin-2009-to-2019")
            files = os.listdir(cache_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                full_csv_path = os.path.join(cache_path, csv_files[0])
                df_raw = pd.read_csv(full_csv_path, on_bad_lines='skip', engine='python')
                df_r1 = pd.DataFrame()
                df_r1["timestamp"]      = pd.to_datetime(df_raw["datetime"], errors='coerce')
                df_r1["source"]         = "Reddit_r/" + df_raw["subreddit"].astype(str)
                df_r1["text"]           = df_raw["body"].astype(str)
                df_r1["user_followers"] = pd.to_numeric(df_raw["score"], errors='coerce')
                all_dfs.append(df_r1)
        except Exception as e:
            print(f"Errore nel Dataset Reddit: {e}")

        if not all_dfs:
            return pd.DataFrame(columns=["timestamp", "source", "text", "user_followers"])

        data = pd.concat(all_dfs, ignore_index=True)
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')
        data = data.dropna(subset=["timestamp", "user_followers"])

        try:
            if data['timestamp'].dt.tz is not None:
                data['timestamp'] = data['timestamp'].dt.tz_convert(None)
        except AttributeError:
            pass

        data = data.loc[(data["timestamp"] >= self.start_date) & (data["timestamp"] <= self.end_date)]

        before = len(data)
        data = data.loc[data["user_followers"] >= min_score]
        print(f"   Filtro score >= {min_score}: {before:,} → {len(data):,} commenti")

        data = data.sort_values(by="timestamp").reset_index(drop=True)

        output_path = os.path.join(self.output_dir, "bitcoin_reddit_dataset.csv")
        data.to_csv(output_path, index=False)

        return data

    def fetch_energy_cost(
        self,
        electricity_price_kwh: float = 0.05,
        df_mining: pd.DataFrame = None,
    ) -> pd.DataFrame:

        try:
            resp = requests.get(
                "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics",
                params={"assets": "btc", "metrics": "HashRate,RevUSD", "frequency": "1d",
                        "start_time": self.start_date.strftime("%Y-%m-%d"),
                        "end_time": self.end_date.strftime("%Y-%m-%d")},
                timeout=30
            )
            resp.raise_for_status()
            df_cm = pd.DataFrame(resp.json().get("data", []))
            df_cm["timestamp"] = pd.to_datetime(df_cm["time"])
            df_cm = df_cm.set_index("timestamp")[["HashRate", "RevUSD"]].astype(float)
            hash_rate_ths = df_cm["HashRate"]
            miners_rev_usd = df_cm["RevUSD"]

        except Exception as e:
            if df_mining is not None:
                hash_rate_ths = df_mining["hash_rate"] / 1e3   # GH/s → TH/s
            else:
                csv = os.path.join(self.output_dir, "bitcoin_mining_metrics.csv")
                df_mining = pd.read_csv(csv, index_col="timestamp", parse_dates=True)
                hash_rate_ths = df_mining["hash_rate"] / 1e3
            miners_rev_usd = None

        EFFICIENCY_J_PER_TH = 20.0
        DAILY_BTC_REWARDS   = 3.125 * 144 + 15

        power_kw     = hash_rate_ths * EFFICIENCY_J_PER_TH / 1000.0
        daily_kwh    = power_kw * 24.0
        cost_per_btc = (daily_kwh * electricity_price_kwh) / DAILY_BTC_REWARDS

        result = pd.DataFrame({
            "hash_rate_ths":        hash_rate_ths,
            "power_gw":             power_kw / 1e6,
            "daily_consumption_gwh": daily_kwh / 1e6,
            "annualised_twh":       (daily_kwh * 365.25) / 1e9,
            "cost_per_btc_usd":     cost_per_btc,
        }, index=hash_rate_ths.index)

        if miners_rev_usd is not None:
            result["miners_revenue_usd"] = miners_rev_usd

        result = result.loc[self.start_date : self.end_date].resample(self.frequency).mean()
        result.index.name = "timestamp"
        result.to_csv(os.path.join(self.output_dir, "bitcoin_energy_cost.csv"))
        return result
    
    def fetch_pullpush_feed(
        self,
        subreddits: list[str] | None = None,
        min_score: int = 5,
        size: int = 100,
        sleep_between_calls: float = 2.0,
    ) -> pd.DataFrame:
        import time
        from datetime import timezone

        if subreddits is None:
            subreddits = ["Bitcoin", "CryptoCurrency", "BitcoinMarkets"]

        BASE_URL = "https://api.pullpush.io/reddit/search/comment/"
        all_rows = []

        def get_with_retry(params, max_retries=5):
            wait = 10
            for attempt in range(max_retries):
                try:
                    resp = requests.get(BASE_URL, params=params, timeout=30)
                    if resp.status_code == 429:
                        print(f"\n   [429] Rate limit. Aspetto {wait}s... (tentativo {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        wait = min(wait * 2, 120)
                        continue
                    resp.raise_for_status()
                    return resp.json().get("data", [])
                except requests.exceptions.HTTPError as e:
                    if resp.status_code == 429:
                        print(f"\n   [429] Rate limit. Aspetto {wait}s... (tentativo {attempt+1}/{max_retries})")
                        time.sleep(wait)
                        wait = min(wait * 2, 120)
                    else:
                        print(f"\n   [warn] HTTP error: {e}")
                        return []
                except Exception as e:
                    print(f"\n   [warn] Errore: {e}")
                    time.sleep(wait)
                    wait = min(wait * 2, 120)
            return []

        for sub in subreddits:
            print(f"   Scarico r/{sub}...")
            before = int(self.end_date.timestamp())
            after  = int(self.start_date.timestamp())

            while True:
                params = {
                    "subreddit": sub,
                    "q":         "bitcoin",
                    "after":     after,
                    "before":    before,
                    "size":      size,
                    "sort":      "desc",
                    "sort_type": "created_utc",
                }

                data = get_with_retry(params)

                if not data:
                    break

                for item in data:
                    ts = datetime.fromtimestamp(
                        item.get("created_utc", 0),
                        tz=timezone.utc
                    ).replace(tzinfo=None)
                    all_rows.append({
                        "timestamp":      ts,
                        "source":         f"Reddit_r/{sub}",
                        "text":           item.get("body", ""),
                        "user_followers": item.get("score", 0),
                    })

                before = min(item["created_utc"] for item in data) - 1
                print(f"   r/{sub}: {len(all_rows)} commenti finora...", end="\r")

                if before <= after:
                    break

                time.sleep(sleep_between_calls)

        print()

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "source", "text", "user_followers"])

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "user_followers"])
        df = df.loc[df["user_followers"] >= min_score]
        df = df.drop_duplicates(subset=["timestamp", "text"]).sort_values("timestamp").reset_index(drop=True)

        print(f"   Totale commenti: {len(df):,}")
        print(f"   Periodo: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        output_path = os.path.join(self.output_dir, "bitcoin_pullpush_reddit.csv")
        df.to_csv(output_path, index=False)

        return df
    
    def fetch_arctic_shift_feed(
        self,
        subreddits: list[str] | None = None,
        min_score: int = 5,
        sleep_between_calls: float = 1.0,
        window_days: int = 7,
        max_per_window: int = 200,
    ) -> pd.DataFrame:
        import time
        from datetime import timezone, timedelta

        if subreddits is None:
            subreddits = ["Bitcoin", "CryptoCurrency", "BitcoinMarkets"]

        BASE_URL        = "https://arctic-shift.photon-reddit.com/api/comments/search"
        checkpoint_path = os.path.join(self.output_dir, "bitcoin_arctic_reddit_checkpoint.csv")

        # ── Resume dal checkpoint ──────────────────────────────────────────────
        if os.path.exists(checkpoint_path):
            df_existing = pd.read_csv(checkpoint_path)
            df_existing["timestamp"] = pd.to_datetime(df_existing["timestamp"])
            all_rows = df_existing.to_dict("records")
            last_ts  = df_existing["timestamp"].max()
            print(f"   Checkpoint trovato: {len(all_rows):,} commenti fino a {last_ts.date()}, riprendo da lì.")
        else:
            all_rows = []
            last_ts  = None

        for sub in subreddits:
            print(f"   Scarico r/{sub}...")
            n_sub        = 0
            window_start = self.start_date

            while window_start < self.end_date:
                window_end = min(window_start + timedelta(days=window_days), self.end_date)

                # Salta finestre già coperte dal checkpoint
                if last_ts is not None and window_end <= last_ts:
                    window_start = window_end
                    continue

                before = window_end.strftime("%Y-%m-%dT%H:%M:%S")
                after  = window_start.strftime("%Y-%m-%dT%H:%M:%S")
                n_window = 0

                while True:
                    url = (
                        f"{BASE_URL}"
                        f"?subreddit={sub}"
                        f"&after={after}"
                        f"&before={before}"
                        f"&limit=100"
                        f"&sort=desc"
                        f"&fields=body,score,created_utc"
                    )
                    try:
                        resp      = requests.get(url, timeout=30)
                        remaining = int(resp.headers.get("X-RateLimit-Remaining", 10))
                        if remaining < 2:
                            reset = int(resp.headers.get("X-RateLimit-Reset", 10))
                            print(f"\n   [rate limit] Aspetto {reset}s...")
                            time.sleep(reset)
                        resp.raise_for_status()
                        data = resp.json().get("data", [])
                    except Exception as e:
                        print(f"\n   [warn] Errore r/{sub} [{after} → {before}]: {e}")
                        data = []
                        break

                    if not data:
                        break

                    for item in data[:max_per_window]:
                        ts = datetime.fromtimestamp(
                            item.get("created_utc", 0),
                            tz=timezone.utc
                        ).replace(tzinfo=None)
                        all_rows.append({
                            "timestamp":      ts,
                            "source":         f"Reddit_r/{sub}",
                            "text":           item.get("body", ""),
                            "user_followers": item.get("score", 0),
                        })
                        n_window += 1

                    n_sub += len(data)
                    print(f"   r/{sub}: {n_sub} commenti finora...", end="\r")

                    # Salvataggio checkpoint dopo ogni finestra
                    pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)

                    if n_window >= max_per_window:
                        break

                    oldest = min(item["created_utc"] for item in data)
                    before = datetime.fromtimestamp(oldest - 1, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

                    if oldest <= int(window_start.timestamp()):
                        break

                    time.sleep(sleep_between_calls)

                window_start = window_end

            print(f"   r/{sub}: {n_sub} commenti totali")

        if not all_rows:
            return pd.DataFrame(columns=["timestamp", "source", "text", "user_followers"])

        df = pd.DataFrame(all_rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "user_followers"])
        df = df.loc[df["user_followers"] >= min_score]
        df = df.drop_duplicates(subset=["timestamp", "text"]).sort_values("timestamp").reset_index(drop=True)

        print(f"   Totale commenti: {len(df):,}")
        print(f"   Periodo: {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

        output_path = os.path.join(self.output_dir, "bitcoin_arctic_reddit.csv")
        df.to_csv(output_path, index=False)

        # Rimuovi checkpoint ora che il file finale è salvato
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        return df
    
    def fetch_fear_greed(self) -> pd.DataFrame:
        """
        Scarica il Fear & Greed Index da alternative.me (gratuito, no key).
        Restituisce un DataFrame settimanale con index DatetimeIndex W.
        """
        resp = requests.get(
            "https://api.alternative.me/fng/?limit=1000&format=json",
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()["data"]

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
        df = df.set_index("timestamp")[["value", "value_classification"]]
        df["value"] = pd.to_numeric(df["value"])
        df = df.sort_index()
        df = df.loc[self.start_date : self.end_date]

        # Aggrega a frequenza settimanale
        weekly = df["value"].resample(self.frequency).agg(
            fear_greed_mean  = "mean",
            fear_greed_min   = "min",
            fear_greed_max   = "max",
            fear_greed_close = "last",
        )
        weekly["fear_greed_delta"] = weekly["fear_greed_close"].diff()

        output_path = os.path.join(self.output_dir, "bitcoin_fear_greed.csv")
        weekly.to_csv(output_path)
        print(f"   Fear & Greed: {len(weekly)} settimane salvate in {output_path}")

        return weekly