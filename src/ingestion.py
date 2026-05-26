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
    
    def fetch_social_feed(self) -> pd.DataFrame:
        all_dfs = []
        try:
            cache_path = kh.dataset_download("pokeash/bitcoin-tweets-dataset-20252026")
            files = os.listdir(cache_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                full_csv_path = os.path.join(cache_path, csv_files[0])
                df_raw = pd.read_csv(full_csv_path, on_bad_lines='skip', engine='python')
                
                df_x1 = pd.DataFrame()
                df_x1["timestamp"] = pd.to_datetime(df_raw["date"], errors='coerce')
                df_x1["source"] = "X_Twitter"
                df_x1["text"] = df_raw["text"].astype(str)
                all_dfs.append(df_x1)
                print(f"   -> Caricate {len(df_x1)} righe da Twitter (2025/2026)")
        except Exception as e:
            print(f"Errore nel Dataset 1 (Twitter 2025/2026): {e}")

        if not all_dfs:
            print("Errore critico: Nessun dataset è stato caricato correttamente.")
            return pd.DataFrame(columns=["timestamp", "source", "text"])

        print("Concatenazione e pulizia dell'archivio globale...")
        data = pd.concat(all_dfs, ignore_index=True)
        
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')
        data = data.dropna(subset=["timestamp"])
        
        try:
            if data['timestamp'].dt.tz is not None:
                data['timestamp'] = data['timestamp'].dt.tz_convert(None)
        except AttributeError:
            pass
            
        print(f"Filtraggio dati per il macro-intervallo richiesto: {self.start_date.date()} / {self.end_date.date()}...")
        data = data.loc[(data["timestamp"] >= self.start_date) & (data["timestamp"] <= self.end_date)]
        
        print("Ordinamento cronologico dei post...")
        data = data.sort_values(by="timestamp").reset_index(drop=True)
        
        output_path = os.path.join(self.output_dir, "bitcoin_social_dataset_unified.csv")
        print(f"Scrittura del file finale unificato ({len(data)} righe)...")
        data.to_csv(output_path, index=False)
        print(f"Processo completato! Dataset unificato salvato in: {output_path}")
        
        return data