# SNAP: Resoconto interventi sul modello di predizione BTC

## 1. Problema di partenza

Sintomo riportato: **AUC in training/CV ≈ 0.70** ma **precision in test ≈ 0.47**, tipico gap da overfitting.

### Cause identificate nel pipeline originale (`train.ipynb`)
- **Selection bias nel tuning**: `RandomizedSearchCV` con 50 iterazioni su uno spazio di 9 iperparametri, valutato con `TimeSeriesSplit(5)` su un dataset settimanale piccolo (~100 righe). Il `best_score_` riportato era il massimo su 50 candidati, quindi sistematicamente ottimistico.
- **Target intrinsecamente difficile**: la direzione settimanale di BTC è vicina a un random walk; un AUC realistico atteso è ~0.52–0.58, non 0.70.
- **Iperparametri fissi per anni di walk-forward**: tuning fatto una sola volta sul primo 70% dei dati, poi riusato invariato attraverso regimi di mercato diversi.
- **Feature ridondanti/collineari**: `macd`, `macd_signal`, `macd_hist` derivano dalla stessa serie e sono altamente correlate.
- **Dataset piccolo** (~107–122 settimane): qualunque metrica singola ha un errore standard enorme.

## 2. Fix applicati a `train.ipynb`

| Area | Intervento |
|---|---|
| Hyperparameter tuning | Spazio di ricerca ridotto e più regolarizzato (`max_depth`≤3, più `reg_lambda`/`min_child_weight`), `n_iter` ridotto a 20, `TimeSeriesSplit(gap=2)` per evitare leakage tra fold adiacenti. `best_score_` etichettato esplicitamente come ottimistico. |
| Feature pruning | Rimozione automatica di coppie con correlazione > 0.95 (mantenuto solo `macd_hist`) + rimozione automatica delle feature a **varianza zero** (es. `has_data`, risultata costante a 1, copertura sentiment completa su tutto il periodo). |
| Train/test split | Aggiunto un *purge gap* di 2 settimane tra train e test iniziali, a protezione da feature con finestre rolling a cavallo del confine. |
| Walk-forward | Re-tuning periodico (ogni 26 settimane) **gated** su una soglia minima di dati (`MIN_RETUNE_SIZE=100`) per evitare re-tuning instabili su finestre troppo piccole. Aggiunte `precision`/`recall`/`F1` oltre ad accuracy/AUC. |
| Diagnostica statistica | **CI bootstrap al 95%** (2000 resample) su AUC/accuracy/precision del walk-forward, per stabilire se i risultati sono statisticamente distinguibili dal random, non solo guardare il valore puntuale. |
| Regime shift | Confronto tra UP-ratio del periodo di warm-up e quello della finestra di walk-forward, per escludere che il calo di performance sia dovuto a un cambio di regime di mercato. |
| Target | **Bug di disallineamento corretto**: l'offset posizionale `close.iloc[t_idx+1]` (che assumeva implicitamente nessun buco nell'indice settimanale) è stato sostituito con una verifica esplicita sulla data: se la settimana successiva nell'indice non dista esattamente 7 giorni, la riga viene scartata invece di produrre un target semanticamente sbagliato. |
| Feature importance & ablation | Nuova sezione che allena il modello per sottogruppi di feature (tecniche, sentiment, on-chain/energia, fear&greed) con iperparametri fissi, con CI bootstrap per ciascun gruppo, per capire se un sottoinsieme isolato porta segnale anche quando il modello combinato non lo fa. |
| Esperimento aggiuntivo | Sezione con **target a soglia** (±2%, configurabile): scarta le settimane con movimento troppo piccolo per essere un segnale pulito, per verificare se il rumore dell'etichetta binaria maschera un segnale più forte sui movimenti ampi. |

## 3. Verifica look-ahead bias / leakage

- **`compute_technical_features` e `compute_energy_features`** (in `src/sentiment_engine.py`): verificate: usano solo `rolling()`, `ewm()`, `pct_change()`, tutte operazioni causali. Nessun leakage.
- **`compute_sentiment_features`** (stesso file): trovato un bug reale: `interpolate(method="linear", limit_direction="both")` riempiva i buchi usando anche **dati futuri**. Corretto sostituendolo con un forward-fill causale (`ffill(limit=4).fillna(0)`), coerente con quanto già fatto manualmente in `train.ipynb`.
  - Verificato tramite `test2.ipynb` che questo metodo **non è mai stato usato** per generare l'attuale `bitcoin_sentiment.csv` (generato invece con `engine.analyze()`, che non ha interpolazione), quindi il CSV esistente **non è contaminato** e non serve rigenerarlo. Il fix resta comunque applicato per sicurezza futura.
- **`src/ingestion.py`**: verificato per intero: aggregazione `Close: 'last'` corretta (non `'mean'`), frequenza `'W'` coerente su tutte le fonti (mercato, mining, energia, fear&greed), nessuna finestra che guarda avanti nel resampling. Nessun problema trovato.

## 4. Nuove feature aggiunte

- **`sentiment_momentum`** (suggerita dal relatore): differenza tra il sentiment della settimana corrente e la media mobile delle 4 settimane precedenti (esclusa quella corrente), implementata in modo causale (`.shift(1)` prima del `.rolling()`).
- **Fear & Greed Index**: aggiunto dall'autore, non era implementato in precedenza. Integrato in `train.ipynb` con lo stesso trattamento causale delle altre fonti (ffill limitato, nessun dato futuro).

## 5. Risultati onesti ottenuti

Con il pipeline corretto, sul dataset attuale (~107–122 settimane):

- **AUC in CV (dopo il fix del tuning): 0.579**, plausibile per il task e non più gonfiato.
- **Walk-forward (metrica onesta out-of-sample)**: AUC ≈ 0.43, accuracy ≈ 0.44, precision ≈ 0.46.
- **CI bootstrap al 95%**: AUC [0.309, 0.562], accuracy [0.325, 0.558]: **entrambi includono 0.5**.
- **UP ratio**: 53.3% nel warm-up vs 49.4% nel walk-forward, differenza minima, nessun vero regime shift.

**Conclusione statisticamente corretta**: con i dati e le feature attuali, **non emerge un segnale predittivo distinguibile dal caso** per la direzione settimanale di BTC. Non è un fallimento del pipeline (che è stato verificato riga per riga per bias, leakage e bug di allineamento), è un risultato plausibile e coerente sia con la letteratura sia con i risultati della precedente tesi triennale dell'autore (45–47% accuracy). Per la discussione in tesi, questa è una lettura più difendibile e onesta di un accuracy "sospettosamente alta" senza intervalli di confidenza.

## 6. Confronto con la baseline Buy & Hold e backtesting

Il notebook include due controlli aggiuntivi, pensati per contestualizzare le metriche di classificazione con una lettura più vicina all'uso pratico (operatività):

- **Sezione 7, confronto con baseline**: accuracy/precision/AUC del modello vengono messe a confronto nella stessa tabella con Buy & Hold (compra e mantieni per tutto il periodo), MACD Crossover (strategia tecnica classica) e Random (media su 50 seed). Serve a verificare che il modello non stia semplicemente replicando un segnale banale già catturato da un indicatore tecnico elementare come il MACD.
- **Sezione 9, backtesting**: simula un capitale iniziale di 10.000 USD investito seguendo i segnali del modello (long quando previsione = UP, cash quando previsione = DOWN, con una fee dello 0.1% ad ogni cambio di posizione) e lo confronta con un Buy & Hold puro sullo stesso periodo, riportando rendimento totale, Sharpe ratio annualizzato e maximum drawdown.

**Perché va letto con la stessa cautela della sezione 5**: dato che gli intervalli di confidenza bootstrap sul walk-forward includono 0.5 (nessun segnale statisticamente distinguibile dal random), un'eventuale sovraperformance del backtest rispetto al Buy & Hold in questo specifico storico va trattata come un risultato non affidabile su cui basare conclusioni, non come conferma di un vantaggio reale, per due motivi aggiuntivi specifici al backtesting:

1. **Le fee di transazione erodono ulteriormente un segnale già debole o assente**: ogni cambio di posizione costa lo 0.1%, quindi anche un modello con un'accuracy marginalmente sopra il caso può finire in perdita netta rispetto a restare semplicemente investiti.
2. **Il Buy & Hold ha un vantaggio strutturale in un periodo di trend rialzista**: se il periodo storico usato per il backtest include una fase di crescita marcata di BTC, il Buy & Hold beneficia semplicemente di essere sempre esposto al mercato; un modello che sta spesso in cash durante quelle fasi (anche solo per un margine di errore nella soglia decisionale) sottoperformerà quasi per costruzione, indipendentemente dalla qualità del segnale.

Per la tesi, la raccomandazione è di riportare accuracy/Sharpe/drawdown del backtest come dato descrittivo, ma di ancorare la conclusione principale sulla bontà del modello alle metriche walk-forward con CI bootstrap (sezione 5), che sono l'unica misura corretta di generalizzazione fuori campione.

## 7. Piste aperte, non ancora implementate/testate

- **Più storico** (es. dal 2022 a oggi): ridurrebbe l'errore standard delle stime (~1.6× più predizioni di walk-forward), ma non garantisce che emerga segnale se il task è genuinamente vicino al caso.
- **Feature aggiuntive non ancora implementate**: correlazione con asset macro (DXY, S&P 500, Oro, rendimento Treasury 10Y), funding rate dei perpetual futures (Binance/Bybit), flussi netti verso/da exchange, variazione di supply delle stablecoin: richiedono integrare nuove fonti dati/API esterne.
- **Modelli più semplici come confronto** (es. logistic regression regolarizzata): con N così piccolo potrebbero generalizzare meglio di XGBoost, utile come baseline aggiuntiva in tesi.

## 8. File consegnati

- `train.ipynb`: pipeline di training corretta e strumentata (tuning conservativo, walk-forward con gating, diagnostica bootstrap, ablation, esperimento target a soglia).
- `sentiment_engine.py`: fix leakage + nuovo metodo `compute_sentiment_momentum`.
