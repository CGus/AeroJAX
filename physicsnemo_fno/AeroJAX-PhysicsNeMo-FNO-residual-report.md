# AeroJAX PhysicsNeMo — FNO residuale con curriculum multi-step

Data: 2026-09-13. Esito: **obiettivo rollout stabile raggiunto**.

## Decisione

La variante raccomandata è **FNO residual autoregressivo**. La formulazione direct-time non è stata addestrata perché era esplicitamente il fallback solo in caso di fallimento e il residual ha superato i criteri pratici: rollout 100 non esplosivo su tutti i casi originali, rollout 500 finito su due sequenze lunghe separate, divergenza controllata e 61,4 FPS su RTX 3060.

Questo risultato giustifica proseguire la validazione del residual FNO, non ancora una promozione in production o GUI.

## Isolamento e integrità

- Solver CFD production, GUI e default numerici: invariati.
- Dataset originale `teacher-pilot`: aperto solo in lettura e invariato.
- Pilot assoluto: checkpoint e risultati sotto `runs/pilot_fno/` non sovrascritti.
- Nuovo codice/config esclusivamente sotto `/home/gus/AeroJAX/physicsnemo_fno/`.
- Nuovo run esclusivamente sotto `/home/gus/AeroJAX/physicsnemo_fno/runs/residual_fno/`.
- Dataset lungo separato esclusivamente sotto `/home/gus/AeroJAX/physicsnemo_fno/datasets/long_eval_500/`.

Il worktree production era già sporco prima del lavoro; il delta nuovo resta confinato a `physicsnemo_fno/`.

## Modello e formulazione

Architettura invariata rispetto al pilot:

- PhysicsNeMo FNO 2D;
- 4 Fourier layer;
- width 32;
- modi 16×12;
- padding costante non-periodico 8;
- coordinate generate da `coord_features=True`;
- 3.152.003 parametri.

La rete predice incrementi normalizzati per canale:

`state_(t+1) = state_t + denormalize_delta(FNO(input_t))`

L'identity skip è esplicito nel codice. Delta mean/std derivano solo dal train:

| Campo | delta mean | delta std |
|---|---:|---:|
| u | 1,3074e-5 | 3,6997e-4 |
| v | 7,8620e-6 | 1,9598e-4 |
| p | 1,0284e-3 | 2,1504e-2 |

Conditioning mantenuto: mask, SDF, coordinate, U∞, viscosità, Reynolds, AoA, dt CFD, intervallo tra frame, direzione inlet, dx/dy e mappe BC inlet/outlet/wall. Il gauge pressione `outlet_zero` viene riapplicato dopo ogni incremento.

## Curriculum e loss

Curriculum: horizon 1 → 2 → 4 → 8, tre epoche per stadio. Batch 16, AdamW, LR 5e-4, weight decay 1e-4, gradient clipping 1,0.

Teacher forcing decrescente:

| Horizon | Probabilità per le tre epoche |
|---:|---|
| 1 | 1,0 → 0,9 → 0,8 |
| 2 | 0,7 → 0,6 → 0,5 |
| 4 | 0,4 → 0,3 → 0,2 |
| 8 | 0,1 → 0,05 → 0,0 |

Loss media sui passi:

`L = L_field + 0.01 L_divergence-teacher + 0.002 L_energy`

`L_field` è MSE normalizzata sulle celle fluide. La penalità divergenza confronta `du/dx+dv/dy` predetto con quello del teacher usando dx/dy del dataset: non impone divergenza ideale zero a un teacher collocated imperfetto. La loss energia confronta la media fluida di `0.5(u²+v²)`.

L'ultima epoca horizon 8 completamente libera ha train loss 2,0956e-4 e validation free loss 2,3389e-4. Il deployment checkpoint è `curriculum_final.pt`; `best.pt` è conservato per diagnostica ma appartiene a uno stadio più corto, le loss dei diversi orizzonti non sono confrontabili direttamente.

Training: 566,86 s, 1.770 optimizer step. Curva: `runs/residual_fno/residual_curriculum.png`.

## One-step sul pilot originale

Metriche in unità fisiche. L2 è RMSE fluida; relativo è norma L2 errore / norma teacher.

| Split | Modello | u rel | v rel | p rel | totale rel | totale L2 | Linf globale |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | assoluto | 1,489% | 9,101% | 7,339% | 2,658% | 0,01637 | 1,1820 |
| validation | residual | 0,0227% | 0,245% | 1,000% | **0,302%** | 0,001859 | 0,2653 |
| test | assoluto | 2,183% | 8,364% | 6,809% | 3,628% | 0,02379 | 2,6337 |
| test | residual | 0,0243% | 0,237% | 0,948% | **0,421%** | 0,002759 | 0,3428 |

## Rollout libero sul pilot originale

Medie su 6 sequenze per split, tutte finite. Le colonne u/v/p sono errori L2 relativi al frame dell'orizzonte.

| Split | H | u rel | v rel | p rel | energia err rel | divergence diff L2 | drift medio u/v/p |
|---|---:|---:|---:|---:|---:|---:|---|
| validation | 10 | 0,00526 | 0,0602 | 0,1447 | 0,00115 | 0,01220 | -4,78e-4 / 1,10e-4 / 1,36e-2 |
| validation | 50 | 0,01498 | 0,1626 | 0,2687 | 0,00229 | 0,01907 | -8,46e-4 / 3,20e-4 / 1,97e-2 |
| validation | 100 | **0,02331** | 0,2422 | 0,2359 | **0,00307** | **0,02071** | -8,61e-4 / 4,65e-4 / 9,02e-3 |
| test | 10 | 0,00679 | 0,0660 | 0,0985 | 0,00306 | 0,01444 | -1,29e-3 / -6,47e-5 / 1,16e-3 |
| test | 50 | 0,01503 | 0,1589 | 0,2519 | 0,00642 | 0,02302 | -2,85e-3 / 8,38e-5 / 5,79e-3 |
| test | 100 | **0,02136** | 0,2257 | 0,2256 | **0,00818** | **0,02513** | -3,71e-3 / 1,17e-4 / -1,83e-3 |

Il pilot assoluto, agli stessi 100 step, arrivava a u relativo 13.703/21.210, v 330.537/372.341 e p 320.012/368.730 su validation/test: differenza qualitativa fra degrado graduale e instabilità esplosiva.

## Dataset mirato e rollout 500

I file originali hanno 200 frame e non consentono una misura a 500. È stato quindi generato un dataset separato minimo, non una campagna estesa:

- 1 ellipse validation + 1 rectangle test;
- condizione U∞=1,25, Re=600;
- 520 frame ciascuna, 1.040 frame totali;
- 128×64, dt 0,001, stride 2, warmup 20;
- 74.183.219 byte, 81,11 s, 12,82 frame/s end-to-end;
- teacher production congelato e stesso schema HDF5;
- nessun uso per training: esclusivamente evaluation.

SHA256:

- ellipse: `8ceea126c47cd9e5229d2f3520b46ea41ad149dc55dd154bb280587c42fd7568`
- rectangle: `78f42a723d4f3f5f0a28d4391b5e54e663049ec6dba76a668cd6ec03f06ada07`

### Confronto a 500 step

| Split/caso | Modello | finito | u rel | v rel | p rel | energia err rel | div diff L2 | drift u/v/p |
|---|---|---|---:|---:|---:|---:|---:|---|
| ellipse validation | residual | sì | **0,0565** | 0,449 | 0,455 | **0,00576** | **0,03461** | -0,00156 / 0,00288 / -0,01159 |
| ellipse validation | assoluto | sì numericamente, esploso | 3,12e14 | 3,05e14 | 4,06e15 | 9,72e28 | 1,19e14 | non utile |
| rectangle test | residual | sì | **0,0581** | 0,476 | 0,392 | **0,01361** | **0,04366** | -0,00674 / 0,00204 / -0,02012 |
| rectangle test | assoluto | overflow | Inf | Inf | Inf | NaN | Inf | non utile |

Il residual non è accurato al 500° step per v/p in senso stretto, ma il degrado è graduale, i campi restano finiti, l'energia non esplode e u rimane entro ~5,8%. Questo soddisfa l'obiettivo di stabilità; non certifica ancora fedeltà long-horizon completa.

Confronto grafico: `runs/residual_fno/long_500_comparison.png`. Timeline complete: `long_500_raw.json`.

## Prestazioni

| Misura | Residual | Pilot assoluto |
|---|---:|---:|
| batch-1 latency | 16,291 ms/frame | 14,158 ms/frame |
| throughput | **61,39 FPS** | 70,63 FPS |
| peak VRAM allocata | 1.929,6 MiB | 362,0 MiB |
| peak VRAM riservata | 2.146,0 MiB | non registrata |
| checkpoint deploy | 12.620.981 byte | 12.618.485 byte |

Benchmark con 30 warmup, 200 iterazioni e sincronizzazione CUDA. L'overhead residuale è ~2,13 ms/frame, ma resta oltre il requisito di 30 FPS. La VRAM maggiore deriva dal backpropagation attraverso horizon 8; non è memoria di sola inferenza.

Checkpoint residual deploy SHA256: `1875c73e30851a714f29696ccce4547bdc8567807809b86e9153008a8eb494d4`.

## CL/CD e shedding

CL/CD non sono disponibili in modo affidabile: gli HDF5 non contengono forze integrate e introdurre ora un nuovo integratore di trazione su contorno raster non sarebbe teacher-consistent. La frequenza di shedding non è stimabile in modo robusto: anche le sequenze lunghe coprono solo circa 1,04 s fisici e non offrono abbastanza periodi né risoluzione spettrale per una stima difendibile. Nessun numero è stato simulato o estrapolato.

## File aggiunti

- `train_residual.py`: training curriculum e diagnostiche;
- `config_residual.json`: configurazione risolta;
- `run_residual.sh`: comando riproducibile;
- `long_eval_500.json`: configurazione del piccolo dataset lungo;
- `evaluate_long_500.py`: confronto congelato a 500 step;
- `runs/residual_fno/`: checkpoint, metriche, raw rollout e grafici;
- `datasets/long_eval_500/`: due HDF5 nuovi e separati.

Comandi:

```bash
cd /home/gus/AeroJAX
./physicsnemo_fno/run_residual.sh

PYTHONPATH=/tmp/aerojax-dataset-deps XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/gus/aerojax_env_clean/bin/python teacher_pipeline/teacher_dataset.py generate \
  --config physicsnemo_fno/long_eval_500.json \
  --output physicsnemo_fno/datasets/long_eval_500

.venv-physicsnemo/bin/python physicsnemo_fno/evaluate_long_500.py
```

## Raccomandazione e prossimi gate

**Scelta: FNO residual autoregressivo.** Non serve passare ora a direct-time e non c'è evidenza per cambiare architettura.

Prossimi passi, prima di qualsiasi integrazione production:

1. Ripetere il 500-step su più geometrie/condizioni lunghe, mantenendo la campagna ancora contenuta.
2. Estendere il curriculum a 16/32 step e selezionare il checkpoint su una metrica rollout composita, non sulla loss one-step.
3. Aumentare moderatamente il peso della divergence penalty solo dopo una ablation, perché già ora la divergence difference resta limitata.
4. Generare sequenze fisicamente abbastanza lunghe per shedding e aggiungere al teacher export osservabili CL/CD teacher-consistent, senza ricostruirli a posteriori con un operatore diverso.
5. Verificare robustezza a stati iniziali differenti e condizioni all'interno del dominio di training.
6. Solo se questi gate mostrano plateau o instabilità, confrontare direct-time; cambiare architettura resta l'ultima opzione.

Nessuna modifica AeroJAX production è stata effettuata.
