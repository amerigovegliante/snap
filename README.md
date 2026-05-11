# Sentiment-News Analysis Pipeline for the value of Bitcoins.

## Week 1 (11/05/2026 - 15/05/2026)

### 1. Revisione Critica della Tesi Triennale
- [ ] **Analisi dei Risultati Precedenti:** Riesaminare il sistema basato su VADER e Logistic Regression.
- [ ] **Documentazione dei Limiti:** Formalizzare i problemi riscontrati:
    - [ ] Scarsità dei dati (12.190 post).
    - [ ] Accuratezza insufficiente (45-47%).
    - [ ] Rumore eccessivo dei social media (Mastodon).
- [ ] **Baseline di Confronto:** Estrarre le metriche della triennale da utilizzare come termine di paragone per il nuovo modello XGBoost.

### 2. Rassegna della Letteratura (Sentiment & Crypto)
- [ ] **Stato dell'Arte Recente:** Ricerca di paper (es. Mudbari, 2025) sulla predizione settimanale di Bitcoin.
- [ ] **Benchmark di Accuratezza:** Validare la soglia del 52% come obiettivo di rilievo per il task specifico.
- [ ] **Analisi Fonti Dati:** Approfondire l'efficacia delle notizie strutturate (CoinDesk, CoinTelegraph) rispetto ai social.

### 3. Studio Tecnico: FinBERT
- [ ] **Architettura del Modello:** Studiare il paper originale di FinBERT.
- [ ] **Vantaggi del Dominio:** Comprendere come il pre-training su testi finanziari migliori l'estrazione del sentiment rispetto a modelli generalisti.
- [ ] **Pianificazione Integrazione:** Studiare la documentazione di Hugging Face per l'implementazione su Google Colab con GPU T4.

### 4. Metodologie di Backtesting
- [ ] **Indicatori di Performance:** Definire rigorosamente le formule per Sharpe Ratio e Maximum Drawdown.
- [ ] **Strategie di Confronto:** Studiare il funzionamento delle baseline standard: Buy and Hold, MACD e SMA crossover.
- [ ] **Logica Long/Short:** Definire la pipeline di simulazione (acquisto su segnale UP, liquidità su segnale DOWN).

### 5. Formalizzazione Domanda di Ricerca
- [ ] **Sintesi degli Obiettivi:** Definire in che misura l'uso di LLM finanziari e notizie strutturate migliori la qualità predittiva rispetto al passato.

## Week 2 (18/05/2026 - 22/05/2026)
## Week 3 (25/05/2026 - 29/05/2026)
## Week 4 (01/06/2026 - 05/06/2026)
## Week 5 (08/06/2026 - 12/06/2026)
## Week 6 (15/06/2026 - 19/06/2026)
## Week 7 (22/06/2026 - 26/06/2026)
## Week 8 (29/06/2026 - 03/07/2026)
## Week 9 (06/07/2026)
