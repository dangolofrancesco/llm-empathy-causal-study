# Full Dataset Scoring - Incremental Approach

## ✅ Problema Risolto

La **Batch API di Groq richiede un piano Pro** e non è disponibile con il free tier.

**Soluzione**: Approccio **incrementale** con **auto-checkpointing** usando la Standard API.

---

## 🚀 Come Funziona

### Caratteristiche Principali

✅ **Funziona con Free Tier** - Standard API Groq (6000 RPM)  
✅ **Completamente Resumable** - Stop & Resume senza perdere progressi  
✅ **Auto-Save** - Salvataggio automatico ogni 1000 conversazioni  
✅ **Sicuro** - Non perdi mai il lavoro fatto  
✅ **Real-time Progress** - Progress bar con ETA  
✅ **Rate Limit Handling** - Backoff automatico  

---

## 📋 Workflow

### 1. Verifica Progresso Attuale

```python
# Nel notebook: 03_full_dataset_incremental_scoring.ipynb
# Cell 3: Check Current Progress
```

Mostra:
- Conversazioni già processate
- Conversazioni rimanenti
- Percentuale completamento

### 2. Avvia/Riprendi Scoring

```bash
python3 scripts/score_conversations_incremental.py \
    --input data/filtered/wildchat_full_preprocessed.csv \
    --output data/scores/wildchat_full_scored_incremental.csv \
    --chunk-size 1000
```

**Nel notebook**: Esegui semplicemente la Cell 4

### 3. Interrompi Quando Vuoi

- **Ctrl+C** o interruzione kernel
- Nessun problema! Il progresso è salvato

### 4. Riprendi Quando Vuoi

- Esegui di nuovo la stessa cella
- Lo script rileva automaticamente dove era rimasto
- Continua esattamente da quel punto

---

## ⏱️ Timeline Stimata

### Per 144,439 conversazioni (~288k API calls):

| Sessione | Conversazioni | Tempo Stimato |
|----------|---------------|---------------|
| Piccola | 1,000 | 30-45 min |
| Media | 5,000 | 2.5-4 ore |
| Grande | 10,000 | 5-8 ore |
| **Completa** | **144,439** | **~12-16 ore** |

### Strategia Consigliata

**Opzione 1: Multiple Sessioni**
- Giorno 1: 10k conversazioni (5-8 ore)
- Giorno 2: 10k conversazioni (5-8 ore)
- Giorno 3: 10k conversazioni (5-8 ore)
- ...continua fino a 144k

**Opzione 2: Weekend**
- Sabato: 50k conversazioni
- Domenica: 50k conversazioni
- Lunedì: 44k conversazioni rimanenti

**Opzione 3: Sessione Lunga**
- Una sessione continua di 12-16 ore (overnight)

---

## 📂 File Creati

### Durante il Processo:
```
data/scores/
├── wildchat_full_scored_incremental.csv    # Output principale (si aggiorna)
```

### File Rimossi:
- ❌ `scripts/score_conversations_batch.py` (non funziona con free tier)
- ❌ `scripts/score_conversations_batch_chunked.py` (non funziona con free tier)
- ❌ `notebooks/03_full_dataset_batch_scoring.ipynb` (non funziona con free tier)
- ❌ `BATCH_SCORING_CHUNKED_GUIDE.md` (obsoleto)

### File Nuovi:
- ✅ `scripts/score_conversations_incremental.py` (nuovo approccio)
- ✅ `notebooks/03_full_dataset_incremental_scoring.ipynb` (nuovo notebook)

---

## 🎯 Funzionalità Avanzate

### Resume da Punto Specifico

Se vuoi ricominciare da un punto specifico (es. riga 5000):

```bash
python3 scripts/score_conversations_incremental.py \
    --input data/filtered/wildchat_full_preprocessed.csv \
    --output data/scores/wildchat_full_scored_incremental.csv \
    --start-from 5000
```

### Cambiare Chunk Size

Per salvare più frequentemente:

```bash
--chunk-size 500  # Salva ogni 500 conversazioni
```

Per salvare meno frequentemente (più veloce):

```bash
--chunk-size 2000  # Salva ogni 2000 conversazioni
```

---

## 🔍 Monitoraggio Progress

### Nel Notebook

Dopo ogni sessione, esegui la **Cell 5** per vedere:
- Conversazioni totali
- Conversazioni scored
- Percentuale completamento
- Conversazioni rimanenti

### Output Real-Time

Durante l'esecuzione vedrai:
```
Processing: 1,234/144,439 (0.9%) | Rate: 2.3 conv/s | ETA: 17.3 hours
```

---

## ⚡ Performance

### Rate Limiting
- **Groq Free Tier**: 6000 richieste/minuto
- **Script**: ~2-3 conversazioni/secondo (4-6 API calls/sec)
- **Margine di sicurezza**: Ampio, evita rate limit errors

### Error Handling
- Retry automatico (3 tentativi)
- Exponential backoff
- Log dettagliati degli errori

---

## 📊 Dopo il Completamento

Una volta completato (144,439/144,439), nel notebook puoi:

1. **Analizzare distribuzioni** (Cell 7-9)
2. **Visualizzare grafici** (Cell 10+)
3. **Esportare statistiche** (come negli altri notebook)

---

## 🆘 Troubleshooting

### "Rate limit hit"
- ✅ Lo script gestisce automaticamente con backoff
- ⏳ Attendi qualche secondo, riprenderà

### "Process interrupted"
- ✅ Nessun problema! Progress salvato
- 🔄 Esegui di nuovo la cella per riprendere

### "Output file corrupted"
- 🔧 Usa `--start-from` per ricominciare dall'ultimo checkpoint valido

---

## 🎉 Pronto per Iniziare!

1. Apri `notebooks/03_full_dataset_incremental_scoring.ipynb`
2. Esegui celle 1-3 per verificare setup
3. Esegui cella 4 per iniziare scoring
4. Interrompi quando vuoi, riprendi quando vuoi
5. Ripeti fino a 100% completamento

**Buon scoring!** 🚀
