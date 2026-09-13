"""Produce a portable pilot report from completed HDF5 data and saved checkpoint."""
import hashlib,json,platform,subprocess,sys,tarfile,zipfile
from pathlib import Path
from teacher_dataset import atomic_json,source_manifest,summarize

def main():
    root=Path(sys.argv[1]); repo=Path('/home/gus/AeroJAX');out=root/'outputs'
    quality=summarize(out/'teacher-pilot')
    checkpoint=out/'AeroJAX-teacher-baseline-20260909.tar.gz'
    production_unchanged=True
    unchanged_files=0
    with tarfile.open(checkpoint) as archive:
        for member in archive.getmembers():
            if member.isfile() and not member.name.startswith('AeroJAX/.git/') and '/__pycache__/' not in member.name:
                path=repo/Path(member.name).relative_to('AeroJAX')
                if not path.exists() or archive.extractfile(member).read()!=path.read_bytes():production_unchanged=False
                else: unchanged_files+=1
    inventory=dict(checkpoint=checkpoint.name,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        base_git_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        current_git_status=subprocess.check_output(['git','-C',str(repo),'status','--short'],text=True),
        production_source_unchanged=production_unchanged,unchanged_checkpoint_files=unchanged_files,
        teacher=source_manifest(repo),python=platform.python_version())
    atomic_json(out/'teacher-checkpoint.json',inventory)
    files=list((root/'work/teacher_pipeline').glob('*'))
    with zipfile.ZipFile(out/'AeroJAX-teacher-pipeline.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            if path.is_file(): z.write(path,'teacher_pipeline/'+path.name)
    rows=[]
    for split in ('train','validation','test'):
        seq=[s for s in quality['sequences'] if s['split']==split]
        rows.append(f"| {split} | {len({s['geometry_id'] for s in seq})} | {len(seq)} | {sum(s['frames'] for s in seq)} |")
    stats='\n'.join(f"| {name} | {s['min']:.6g} | {s['max']:.6g} | {s['mean']:.6g} | {s['std']:.6g} |" for name,s in quality['fields'].items())
    secs=quality['generation_seconds']; b=quality['bytes']; fps=quality['frame_s_including_io']
    report=f'''# AeroJAX teacher → PhysicsNeMo: consegna

Checkpoint completo `{checkpoint.name}`, SHA256 `{inventory['checkpoint_sha256']}`.
Git base `{inventory['base_git_commit']}`; il checkpoint comprende le modifiche non committate e i backup.
Nessun reset/commit indiscriminato su modifiche precedenti. Sorgenti numerici invariati: {production_unchanged}.

## Risultato del pilota

12 geometrie, 4 famiglie, 24 sequenze, {quality['frames']} frame; 128×64, float32 u/v/p,
200 frame per sequenza dopo 20 step iniziali, stride 2, dt=0,001: ogni sequenza arriva a t=0,42 s.
U=0,75/1,25; Re=200/600 con viscosità risolta in funzione della geometria.
Tempo registrato nei file {secs:.2f} s; throughput end-to-end {fps:.2f} frame/s.
Include inizializzazione/compilazione, sincronizzazione GPU, copie, compressione e scrittura;
esclude analisi finale e setup Python. La prima sequenza è stata ripresa da tre frame.
Byte HDF5 {b:,} ({b/2**20:.2f} MiB). Tutti finiti: {quality['finite']}.
Integrità geometrie/split: {quality['geometry_split_integrity']}.

| Split | Geometrie | Sequenze | Frame |
|---|---:|---:|---:|
{chr(10).join(rows)}

Train NACA/cilindri, validation ellissi, test rettangoli: split fuori famiglia volutamente severo.
Tutte le varianti U/Re/AoA di un gruppo devono restare nello stesso split. Hash delle maschere
controllati; asset importati nuovi richiedono anche un gruppo di provenienza per quasi duplicati.

| Campo | Min | Max | Media | Std |
|---|---:|---:|---:|---:|
{stats}

Statistiche globali per QC soltanto. `normalization-train.json` deriva solo dal training.
Sono dati transitori brevi a risoluzione ridotta: validano la pipeline, non una rete pronta né la statistica della scia.

## Formato e riproduzione

HDF5, un file per sequenza, chunk `[1,3,nx,ny]`, LZF/shuffle/checksum, float32 lossless.
Mask/SDF e coordinate sono statici; tempo, dt, indici e campi sono appendibili. Schema e procedure
in `teacher_pipeline/README.md`. Zarr/sharding resta una possibilità per storage distribuito;
la scelta HDF5 è motivata dal workflow locale, non da una comparazione di velocità non eseguita.

```bash
cd ~/AeroJAX
PYTHONPATH=/tmp/aerojax-dataset-deps XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  ~/aerojax_env_clean/bin/python teacher_pipeline/teacher_dataset.py generate \\
  --config teacher_pipeline/pilot.json --output /percorso/teacher-pilot
```

Installare h5py isolato usando `requirements-dataset.txt`; nessuna installazione di PhysicsNeMo.
Stesso comando per ripresa; il teacher/caso modificato viene rifiutato.
Il dataset consegnato si trova in `outputs/teacher-pilot/`.
L'archivio `AeroJAX-teacher-pipeline.zip` contiene codice, configurazione, loader NumPy e progetto surrogate.

## Architettura e formulazione raccomandate

Prima scelta: FNO 2D geometry-conditioned autoregressivo:
`[u,v,p]_t + mask/SDF + parametri risolti + BC + Δt -> [u,v,p]_(t+Δt)`.
Quattro layer, width 32, modi iniziali [16,12], padding nonperiodico e coordinate esplicite.
FNO sfrutta la griglia strutturata e mantiene lo stesso formato dei campi. Parametri iniziali da misurare,
non un modello già validato. Confronto di controllo con U-Net.

AFNO (Adaptive Fourier Neural Operator) è più interessante con campi grandi e token/patch;
non abbiamo ancora evidenza che migliori il piccolo problema 2D. GINO/DoMINO affrontano geometrie
più complesse con rappresentazioni geometriche dedicate, ma richiedono maggiore complessità e dati.
`geometry+params+time -> field` non rappresenta diversi stati iniziali/interazioni; rollout latente
aggiunge errore di ricostruzione. A è quindi la formulazione iniziale; latent è una successiva ablation.

NVIDIA: [modelli](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api_models.html),
[FNO](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/_modules/physicsnemo/models/fno/fno.html),
[esempi](https://github.com/NVIDIA/physicsnemo/blob/main/examples/README.md).
Confronti: [AFNO](https://arxiv.org/abs/2111.13587), [GINO](https://arxiv.org/abs/2309.00583).

## Scala reale e risorse (stime, non misure di training)

Prima campagna utile proposta: 200 geometrie × 5 condizioni × 2000 frame = 4 milioni di frame.
A 512×192, solo u/v/p float32 occupano 1.18 MB/frame, cioè 4.72 TB decimali non compressi;
aggiungere mask/SDF, metadata, checkpoint e ridondanza. Non extrapolare il rapporto di compressione
del breve pilota alla turbolenza. A 128×64, gli stessi frame occupano 393 GB non compressi.

Usando il precedente benchmark production ~44 step/s e stride 5: 20 milioni di step richiedono
circa 126 ore di solo CFD (5,3 giorni), più warmup/compilazione/I/O/QC. Pianificare 6–10 giorni
su GPU singola per questo scenario, da ricalibrare con sequenze lunghe e storage WSL ext4.
Il throughput pilota include NTFS e inizializzazioni frequenti, non va scalato linearmente con la griglia.

Hardware rilevato: NVIDIA GeForce RTX 3060, 12 GB VRAM, driver 616.56; WSL dispone di circa 23 GiB RAM.
Writer memorizza pochi frame, non la campagna in RAM. Conservare 4–8 GB di margine host per JAX/I/O,
misurare il picco prima di parallelizzare. FNO width32, 4 layer, batch4 a 512×192: sola attivazione
per layer ~50 MB float32; FFT/autograd/optimizer moltiplicano il costo. Stima iniziale training 4–8 GB
VRAM, 12 GB consigliati; 16–32 GB RAM con dataloader lazy. Non è un picco misurato e va verificato con
un dry run; non caricare dataset in memoria. Tempo training/inferenza non stimabile affidabilmente
senza installazione e microbenchmark. Nessun training serio avviato.

## Piano training e GUI

1. Validare il loader e un piccolo overfit su train, esclusivamente come test tecnico.
2. Campagna maggiore con tempi fisici sufficienti, controlli teacher e split di provenienza.
3. Normalizzazione dal train, training one-step, poi rollout 2/4/8 step.
4. Validazione libera 100/1000 step: errore per campo, energia, divergenza rispetto al teacher,
   CL/CD e shedding quando statisticamente misurabile; mai approvare solo su MSE one-step.
5. Misurare ms/inferenza, RTF e memoria su questa GPU, poi implementare backend Neural dietro
   reset/advance_to/snapshot/status/close con campi nelle unità/orientazione attuali.
6. GUI con selettore CFD/Neural, versione modello, dominio supportato, reset coerente della traiettoria
   e fallback esplicito CFD per casi fuori distribuzione. Nessuna modifica GUI ora.

Limiti noti conservati: teacher legacy non è ground truth certificata; residuo/projection, ordine
numerico e coordinate sono quelli documentati dall'audit. Raster arbitrario supportato non significa
generalizzazione appresa. Inlet direzione diversa da orizzontale è rifiutata; AoA è rotazione geometrica.
SDF raster è solo conditioning, non una modifica alla penalizzazione. FFT separata solo per BC compatibili;
nessun routing automatico nuovo viene attivato nella GUI.
'''
    (out/'AeroJAX-PhysicsNeMo-handoff.md').write_text(report)
    print(json.dumps(dict(production_unchanged=production_unchanged,frames=quality['frames'],bytes=b,seconds=secs)))

if __name__=='__main__':main()
