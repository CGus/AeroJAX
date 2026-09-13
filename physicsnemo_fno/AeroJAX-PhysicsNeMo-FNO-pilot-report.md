# AeroJAX → PhysicsNeMo: primo FNO 2D sul pilot congelato

Data: 2026-09-13. Esito: **training completato, checkpoint non idoneo a rollout libero lungo**.

## Confini e stato del repository

Il lavoro è stato eseguito direttamente in `/home/gus/AeroJAX`. Il `git status` iniziale mostrava un worktree già molto sporco (modifiche preesistenti in `main.py`, `solver/`, `pressure_solvers/`, `viewer/`, più molti file non tracciati). Queste modifiche non sono state alterate né incluse. L'unica nuova area di questa fase è `/home/gus/AeroJAX/physicsnemo_fno/`; il venv separato è `/home/gus/AeroJAX/.venv-physicsnemo/`. Nessun file production, GUI o teacher è stato modificato.

Il dataset originale è stato aperto solo in lettura:
`/mnt/c/users/gusta/documents/codex/2026-09-09/files-pasted-by-the-user-s/outputs/teacher-pilot`.
`runs/pilot_fno/dataset_manifest.json` conserva SHA256, sequence ID, geometry ID, family, shape e frame count di ogni HDF5 usato.

## Ambiente verificato

| Componente | Versione/misura |
|---|---|
| OS/kernel | WSL2 Ubuntu, `6.18.33.2-microsoft-standard-WSL2` |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 3060, 12,288 MiB |
| Driver WSL / KMD Windows | 615.65.06 / 616.56 |
| CUDA UMD / build PyTorch | 13.4 / 13.0 |
| PyTorch | 2.14.0+cu130 |
| PhysicsNeMo | 1.3.0 (wheel PyPI disponibile durante l'esecuzione) |
| h5py / NumPy | 3.16.0 / 2.2.6 |

`nvidia-physicsnemo[cu13]` ha emesso un warning perché il wheel 1.3.0 non dichiara quell'extra; la risoluzione ha comunque installato PyTorch CUDA 13.0 e il test reale ha confermato `torch.cuda.is_available() == True` e la RTX 3060. Lo snapshot completo è in `environment.freeze.txt`; `environment.nvidia-smi.txt` registra driver e GPU.

Installazione riproducibile (venv distinto dal CFD):

```bash
cd /home/gus/AeroJAX
python3 -m venv .venv-physicsnemo
.venv-physicsnemo/bin/python -m pip install --upgrade pip setuptools wheel
.venv-physicsnemo/bin/python -m pip install 'nvidia-physicsnemo[cu13]' h5py matplotlib
./physicsnemo_fno/run_all.sh
```

Il file `requirements.lock.txt` documenta i pacchetti principali; `environment.freeze.txt` è il lock risolto completo della macchina. Il venv occupa 5,8 GiB.

## Dati, split e leakage

| Split | Sequenze | Geometrie/famiglie | Frame | Coppie one-step |
|---|---:|---|---:|---:|
| train | 12 | 6, NACA + cilindri | 2.400 | 2.388 |
| validation | 6 | 3, ellissi | 1.200 | 1.194 |
| test | 6 | 3, rettangoli | 1.200 | 1.194 |

Il runner usa le directory split esistenti e verifica che nessun `geometry_id` compaia in più split. Non crea coppie tra sequenze. Le statistiche u/v/p provengono esclusivamente da `normalization-train.json`; anche media/std dei parametri sono calcolate esclusivamente sui file train. Per questo piccolo pilot i dati sono precaricati in RAM per eliminare il collo di bottiglia NTFS, senza scrivere negli HDF5.

## Formulazione e architettura

Input per cella: stato normalizzato `[u,v,p]_t`, mask, SDF normalizzata per sequenza, 10 parametri broadcast (`U_inf`, `nu`, `Re`, AoA, dt CFD, intervallo esportato, direzione inlet x/y, dx, dy), e quattro mappe BC (inlet, outlet, parete inferiore/superiore). Le coordinate normalizzate sono aggiunte internamente da `PhysicsNeMo FNO(coord_features=True)`. Output: `[u,v,p]` al frame successivo.

La configurazione raccomandata è rimasta invariata: 4 Fourier layer, width/latent channels 32, modi 16×12, padding costante non-periodico 8. Parametri allenabili: **3.152.003**. La loss è MSE normalizzata pesata sulla mask fluida. L'output pressione viene riportato al gauge `outlet_zero` sottraendo la media sull'outlet prima della loss e delle metriche.

Variazioni operative rispetto al piano iniziale: batch 16 (la VRAM lo consentiva), 12 epoche con cosine LR e patience 4 invece di un budget più lungo; nessuna riduzione del modello. Seed: 20260913. AdamW, LR 1e-3, weight decay 1e-4.

## Smoke, overfit e training

- Smoke: 2 update su 16 coppie; loss 2,5183 → 1,8301, finita.
- Overfit: 8 coppie, 80 update; loss 2,1005 → 0,02214 (riduzione ~94,9×). Il controllo di forte riduzione è superato, ma il target ambizioso 5e-4 **non è stato raggiunto**.
- Training reale: 12 epoche, 1.800 optimizer step, 170,23 s. Nessun early stop perché la validation è migliorata fino all'ultima epoca.
- Loss epoca 1: train 0,18412, validation 0,020834. Epoca 12/best: train 0,0005804, validation 0,0036604.

Curve: `runs/pilot_fno/training_curve.png`; dati completi: `history.json`.

## One-step (unità fisiche)

L2 è RMSE sulle celle fluide; Linf è massimo assoluto; relativo è norma L2 errore / norma L2 teacher.

| Split | Campo | L2 RMSE | Linf | L2 relativo |
|---|---|---:|---:|---:|
| validation | u | 0,01512 | 0,92782 | 1,489% |
| validation | v | 0,004510 | 0,17249 | 9,101% |
| validation | p | 0,02355 | 1,18202 | 7,339% |
| validation | totale | 0,01637 | 1,18202 | 2,658% |
| test | u | 0,02217 | 0,91352 | 2,183% |
| test | v | 0,005775 | 0,18819 | 8,364% |
| test | p | 0,03424 | 2,63370 | 6,809% |
| test | totale | 0,02379 | 2,63370 | 3,628% |

## Rollout autoregressivo libero

Ogni sequenza validation/test è avviata dal frame 0 teacher e poi evolve senza teacher forcing. I valori sono medie sui 6 casi al frame dell'orizzonte; Linf è prima calcolato per sequenza e poi mediato. Tutti i tensori sono rimasti finiti, ma “finito” non significa fisicamente stabile.

### Validation

| H | u rel/L2/Linf | v rel/L2/Linf | p rel/L2/Linf | KE err rel | div diff L2 | crescita errore |
|---:|---|---|---|---:|---:|---:|
| 10 | 0,101 / 0,0981 / 1,038 | 0,921 / 0,0417 / 0,707 | 0,264 / 0,162 / 2,337 | 0,0529 | 0,0879 | 6,62× |
| 50 | 5,891 / 5,449 / 30,36 | 115,37 / 5,436 / 26,94 | 64,21 / 14,85 / 73,69 | 141,08 | 7,695 | 912× |
| 100 | 13.702,5 / 12.195 / 69.529 | 330.537 / 14.884 / 69.469 | 320.012 / 55.646 / 205.817 | 1,551e9 | 17.009 | 3,136e6× |

### Test

| H | u rel/L2/Linf | v rel/L2/Linf | p rel/L2/Linf | KE err rel | div diff L2 | crescita errore |
|---:|---|---|---|---:|---:|---:|
| 10 | 0,123 / 0,123 / 0,987 | 0,643 / 0,0428 / 0,743 | 0,300 / 0,338 / 3,714 | 0,0330 | 0,1120 | 5,86× |
| 50 | 5,825 / 5,893 / 31,69 | 67,04 / 4,805 / 28,15 | 51,41 / 17,87 / 77,39 | 73,42 | 8,239 | 693× |
| 100 | 21.210 / 26.070 / 117.535 | 372.341 / 30.590 / 138.108 | 368.730 / 106.688 / 375.047 | 6,058e9 | 35.888 | 4,226e6× |

Curve: `runs/pilot_fno/rollout_error.png`. Le timeline complete per sequenza sono in `rollouts_validation_raw.json` e `rollouts_test_raw.json`.

La divergenza è `du/dx + dv/dy` con differenze NumPy sulle coordinate/spacing teacher e confronto diretto col teacher, non un vincolo di incomprimibilità ideale. L'energia cinetica è la media fluida di `0.5*(u²+v²)`.

CL/CD e Strouhal non sono riportati: il dataset non contiene forze integrate, il runner non dispone dell'operatore di trazione sulla superficie coerente col teacher e le traiettorie transitorie da 0,42 s sono troppo brevi per una frequenza di shedding affidabile. Ricavarli da un contorno raster con una formula nuova avrebbe prodotto numeri non comparabili.

## Prestazioni misurate

| Quantità | Risultato |
|---|---:|
| Picco VRAM PyTorch allocata | 379.545.088 byte = 362,0 MiB |
| Training | 170,23 s per 12 epoche |
| Throughput training | 168,34 coppie/s |
| Inferenza batch 1 | 14,158 ms/frame |
| Surrogate | 70,629 frame/s |
| Teacher comparabile | 22,0 frame esportati/s |
| Speedup misurato derivato | 3,210× |
| Best checkpoint | 12.618.485 byte = 12,03 MiB |
| Latest con optimizer | circa 37 MiB |

La latenza usa 30 warmup, 200 iterazioni e sincronizzazione CUDA prima/dopo il timer. Il picco VRAM è `torch.cuda.max_memory_allocated`, quindi esclude memoria del display/driver e cache riservata non allocata.

Il confronto teacher usa il benchmark production misurato precedente di ~44 CFD step/s e lo stride 2 del pilot, dunque 22 frame allo stesso intervallo fisico esportato/s. È più comparabile della velocità end-to-end HDF5 (10,15 frame/s), ma resta un confronto tra benchmark eseguiti in sessioni diverse: lo speedup 3,21× non va interpretato come benchmark simultaneo né come speedup GUI end-to-end.

## Artefatti

- Codice/config: `/home/gus/AeroJAX/physicsnemo_fno/{train_eval.py,config.json,run_all.sh,requirements.lock.txt}`
- Ambiente: `/home/gus/AeroJAX/physicsnemo_fno/{environment.freeze.txt,environment.nvidia-smi.txt}`
- Run: `/home/gus/AeroJAX/physicsnemo_fno/runs/pilot_fno/`
- Checkpoint: `best.pt` (epoca 12), `latest.pt` (epoca 12 + optimizer)
- Metriche: `smoke.json`, `overfit.json`, `history.json`, `one_step.json`, `rollouts_summary.json`, raw rollout JSON, `performance.json`, `environment.json`
- Provenienza: `dataset_manifest.json`, `normalization.json`, `config.resolved.json`

## Limiti, diagnosi e raccomandazione

**Raccomandazione netta: correggere la formulazione e il curriculum FNO; non promuovere questo checkpoint e non cambiare architettura come prima reazione.** Il one-step mostra che l'FNO apprende una mappa locale utile anche fuori famiglia, ma l'instabilità autoregressiva esplode tra 10 e 50 step. Il problema prioritario è l'addestramento esclusivamente one-step su un pilot breve, con dinamica per-frame molto piccola e senza loss multi-step/stabilità; non c'è ancora evidenza che AFNO, GINO o un'altra famiglia risolva questo difetto.

Prossimi passi concreti:

1. Predire un incremento/residuo fisico `state_(t+1)-state_t` con scala per canale, skip identità esplicito e gauge applicato dopo la ricostruzione.
2. Aggiungere curriculum teacher-forced/free 2/4/8/16-step e loss su energia, drift e divergenza rispetto al teacher; selezionare il best sulla metrica rollout, non sulla sola one-step validation.
3. Campionare finestre da più posizioni per sequenza e aumentare durata/varietà del dataset prima di giudicare shedding o long-horizon.
4. Eseguire ablation coerenti: pressure gauge/pressure-delta, BC maps, SDF scaling, rollout loss; mantenere seed e split invariati.
5. Solo dopo una baseline FNO stabile, confrontare una U-Net economica a parità di dati/budget. Considerare operatori geometrici più complessi solo con rappresentazioni/dataset che lo giustifichino.
6. Ripetere il benchmark teacher e surrogate nello stesso processo/sessione GPU se lo speedup diventa criterio di accettazione.

Il solver CFD production legacy, la discretizzazione, i default e la GUI restano invariati.
