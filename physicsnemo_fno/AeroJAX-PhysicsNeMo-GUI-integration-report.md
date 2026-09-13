# AeroJAX PhysicsNeMo FNO — integrazione GUI

Data verifica: 2026-09-13. Il solver CFD production, i suoi algoritmi numerici e gli HDF5 originali non sono stati modificati in questa fase. Il worktree era già sporco prima dell'integrazione; le modifiche preesistenti sono state preservate.

## Architettura

`AeroJAX CFD` resta il backend predefinito e usa il `SimulationWorker` esistente. `PhysicsNeMo FNO` usa un `PhysicsNeMoWorker` alternativo con la stessa interfaccia Qt e la stessa pipeline di visualizzazione.

La GUI gira in `/home/gus/aerojax_env_clean`. Il worker GUI avvia un processo persistente con `/home/gus/AeroJAX/.venv-physicsnemo/bin/python`; PyTorch e PhysicsNeMo non vengono importati nell'ambiente AeroJAX. Il processo carica una sola volta il checkpoint, conserva lo stato autoregressivo sulla GPU e comunica con la GUI tramite stdin/stdout con richieste JSON e risposte binarie length-prefixed. Ogni frame contiene `u`, `v`, `p`, velocità, vorticità, divergenza, maschera, tempo e indice step.

JAX viene avviato con `XLA_PYTHON_CLIENT_PREALLOCATE=false` per permettere a JAX e PyTorch di condividere la RTX 3060. La modifica riguarda solo la politica dell'allocatore e non cambia solver o numerica.

Checkpoint usato:

`physicsnemo_fno/runs/residual_fno/best.pt` — 12.620.981 byte, FNO residuale autoregressivo, circa 3,15 M parametri.

## Uso

```bash
cd /home/gus/AeroJAX
/home/gus/aerojax_env_clean/bin/python main.py
```

Selezionare `PhysicsNeMo FNO`, un caso pilot e una delle due condizioni, quindi Start. Pause conserva lo stato; Start riprende. Reset arresta e scarica il processo neurale; il successivo Start riparte dal frame iniziale del caso selezionato. Il ritorno a `AeroJAX CFD` è sempre disponibile.

La root del dataset pilot può essere sostituita senza modificare codice con `AEROJAX_FNO_DATASET=/percorso/teacher-pilot`.

## Dominio supportato da surrogate v1

- griglia: 128×64;
- dominio: 20×7,5;
- intervallo autoregressivo: 0,002 s (due step CFD da 0,001 s);
- condizioni: U∞=0,75 con Re=200, oppure U∞=1,25 con Re=600;
- direzione ingresso: (1,0);
- NACA: 0012/0°, 2412/3°, 4415/6° (training);
- cilindro: r=0,4, 0,6, 0,8 (training);
- ellisse: tre dimensioni pilot (solo validation);
- rettangolo: tre dimensioni pilot (solo test).

`nu` è quello risolto nel caso HDF5 dalla coppia U∞/Re e dalla lunghezza caratteristica. U∞, Re, nu, AoA/direzione e dt arbitrari non sono esposti come supportati. I controlli CFD generali restano nella GUI ma, con FNO attivo, il valore autorevole è il caso pilot mostrato nel selettore. Maschere custom, geometrie spostate/ridimensionate fuori dai casi elencati, moto dinamico e ingressi inclinati non sono validati e non sono supportati.

## Graceful fallback

Prima dell'avvio vengono verificati interprete PhysicsNeMo, server, checkpoint e dataset. Il processo verifica CUDA, caricamento del checkpoint e catalogo casi. Qualsiasi errore viene riportato alla GUI, il worker viene arrestato e il selettore torna ad AeroJAX CFD; la GUI non termina.

## Prestazioni misurate

Misura RTX 3060, 100 frame dopo 5 warm-up, caso `cylinder-1`, U∞=0,75, Re=200:

| voce | valore |
|---|---:|
| inferenza CUDA | 16,194 ms/frame |
| sincronizzazione/copia GPU→CPU | 0,272 ms/frame |
| round-trip bridge totale | 19,609 ms/frame |
| overhead bridge e serializzazione | 3,143 ms/frame |
| throughput effettivo bridge | 51,00 FPS |
| VRAM inferenza allocata PyTorch | 21,47 MiB |
| VRAM inferenza riservata PyTorch | 46,0 MiB |
| caricamento iniziale processo/checkpoint | circa 13,1 s |

Il target di almeno 30 FPS è superato nel percorso GUI-worker. Il renderer è disaccoppiato e limitato dal target visuale esistente (30 FPS). Il benchmark scientifico precedente del solo modello, senza IPC, resta 16,291 ms e 61,39 FPS. Il CFD teacher misurato negli artefatti produce 44 step/s; con stride 2 equivale a 22 frame/s memorizzati. Il confronto è indicativo perché la GUI CFD production può usare griglia e impostazioni diverse.

## Verifica numerica del percorso integrato

Il frame prodotto dal server dopo un passo è stato confrontato con il frame CFD successivo dello stesso HDF5. Tutti i campi sono finiti.

| caso U∞=0,75 / Re=200 | rel L2 u | rel L2 v | rel L2 p | rel L2 totale | errore energia | differenza L2 divergenza |
|---|---:|---:|---:|---:|---:|---:|
| NACA 2412 / 3° | 0,097% | 1,882% | 3,103% | 2,014% | 0,000044% | 0,001428 |
| cilindro r=0,6 | 0,182% | 1,561% | 1,693% | 1,480% | 0,0154% | 0,002009 |
| ellisse 0,95×0,42 | 0,087% | 0,966% | 3,943% | 2,830% | 0,0177% | 0,001537 |
| rettangolo 0,65×0,60 | 0,131% | 1,163% | 2,453% | 2,136% | 0,0263% | 0,001865 |

Questa è una verifica del primo passo e del trasporto dati, non una nuova validazione scientifica. Le metriche aggregate e i rollout 100/500 restano quelli degli artefatti validati esistenti.

## Limiti e prossimi passi minimi

Il caricamento iniziale di circa 13 s deve essere nascosto con un indicatore “loading” o con pre-warm opzionale per una demo reel-ready. Il selettore surrogate usa volutamente casi pilot discreti invece dei controlli geometrici CFD. Ellissi e rettangoli sono dimostrabili ma vanno etichettati come generalizzazione validation/test. Il renderer mostra correttamente i campi; una sagoma generica ricavata dalla maschera pilot migliorerebbe il contorno per ellissi e rettangoli senza cambiare il modello. Una misura video on-screen del render dipende dal compositor/display reale; il profiling esistente continua a mostrare timing render e FPS visuali durante la demo.
