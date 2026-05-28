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
            cache_path = kh.dataset_download("pokeash/bitcoin-tweets-dataset-20252026")
            files = os.listdir(cache_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                full_csv_path = os.path.join(cache_path, csv_files[0])
                df_raw = pd.read_csv(full_csv_path, on_bad_lines='skip', engine='python')
                df_x1 = pd.DataFrame()
                df_x1["timestamp"]      = pd.to_datetime(df_raw["date"], errors='coerce')
                df_x1["source"]         = "X_Twitter"
                df_x1["text"]           = df_raw["text"].astype(str)
                df_x1["user_followers"] = pd.to_numeric(df_raw["user_followers"], errors='coerce')
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

        # Filtra per follower
        before = len(data)
        data = data.loc[data["user_followers"] >= min_followers]
        print(f"   Filtro follower >= {min_followers}: {before:,} → {len(data):,} tweet")

        data = data.sort_values(by="timestamp").reset_index(drop=True)
        output_path = os.path.join(self.output_dir, "bitcoin_social_dataset_unified.csv")
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