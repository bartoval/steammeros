# Validazione CPU della fase 2

## Ambiente

- Linux x86-64, kernel `7.2.13-200.fc44.x86_64`.
- AMD Ryzen 7 7840HS: 8 core fisici, 16 thread logici; 32.777.928.704 byte di RAM.
- GCC 16.2.1, CMake 4.3.0, Go 1.26.10, Python 3.14.8.
- PyTorch `2.15.0+cu132`, esecuzione CPU FP32 con GPU nascosta.
- ONNX Runtime 1.31.1 sia nel pacchetto Python sia nell'archivio nativo;
  sorgenti al commit `6af1a932e2286d5c58166085d24c68783033edee`.
- SCNet originale al commit `5d95bf96b19c3eede63248d171efeca8e3abb948`;
  checkpoint e configurazione corrispondono ai checksum del manifest.
- Otto thread per i confronti numerici e le separazioni native.

Questi sono dati della macchina di sviluppo, non requisiti minimi del prodotto.
Le misure di tempo sono esecuzioni singole, non benchmark con warm-up e ripetizioni.

## Diagnosi della precisione

L'ingresso della diagnostica è costituito dai primi 10 secondi dell'estratto
`05-dense_mix`. Ogni componente riceve esattamente l'ingresso prodotto dal
percorso originale PyTorch; in questo modo l'errore di uno stadio non viene
attribuito automaticamente allo stadio successivo.

| Componente | SNR prima | SNR con riduzioni per asse |
| --- | ---: | ---: |
| Prima GroupNorm del DualPathRNN | 58,57 dB | 138,83 dB |
| Seconda GroupNorm del DualPathRNN | 59,96 dB | 143,87 dB |
| Primo DualPathRNN completo | 71,65 dB | 137,53 dB |

Le LSTM isolate raggiungono rispettivamente 127,40 e 128,87 dB. La prima
bisezione aveva localizzato una divergenza nel blocco, ma non dimostrava un
difetto della LSTM.

La precedente riscrittura di GroupNorm a due passaggi manteneva la riduzione
su un gruppo appiattito di milioni di valori: sul caso reale non migliorava
la precisione. La correzione mantiene separati gli assi di canali e spazio,
calcola la media per asse e applica lo stesso procedimento agli scarti
quadratici. Mantiene gruppi, varianza di popolazione, epsilon e parametri
affine originali. Non riduce la precisione dei pesi né modifica l'overlap.

Il riferimento dell'export usa ora le riduzioni originali. Le riscritture
sono applicate solo alle istanze del modello da esportare, in un contesto
che le ripristina alla fine; non modificano globalmente le classi PyTorch.

Sui primi 20 secondi dello stesso estratto, a livello di spettro:

- PyTorch riscritto contro PyTorch originale: 128,61 dB.
- ONNX Runtime contro PyTorch originale: 126,74 dB.
- Errore assoluto massimo: `3.0517578125e-05`.

Queste misure sullo spettro non sostituiscono il confronto degli stem audio
finali sull'intero corpus.

## Criteri mantenuti

- Sample rate 44.100 Hz; canali, lunghezza e ordine degli stem invariati.
- Nessun valore non finito.
- SNR dell'errore almeno 50 dB per stem non quasi silenzioso.
- SNR almeno 40 dB nella peggiore finestra di un secondo non quasi silenziosa.
- Riferimento quasi silenzioso: RMS inferiore a `1e-4`; errore assoluto massimo
  non superiore a `1e-4`.
- Regressione SDR mediana non superiore a 0,1 dB per stem; tutti i casi peggiori
  devono essere riportati. Le sorgenti quasi silenziose sono escluse dall'SDR,
  ma restano coperte dalla parità assoluta.

## Comandi

L'installazione del runtime e le verifiche sono state autorizzate esplicitamente.
La telemetria Python è disabilitata prima di importare ONNX Runtime;
la build nativa usa `--no_telemetry`.

```sh
.venv/bin/python -m pip --disable-pip-version-check install --no-deps onnxruntime==1.31.1
cmake -S . -B build -DFETCHCONTENT_FULLY_DISCONNECTED=ON
cmake --build build --parallel 4

.venv/bin/python -I tools/diagnose_precision.py --stages norm0 norm1 block --out data/results/precision-dp0-current
.venv/bin/python -I tools/export_onnx.py --threads 8
OMP_NUM_THREADS=2 .venv/bin/python -I -m unittest discover -s tools -p 'test_*.py' -v

export PKG_CONFIG_PATH="$PWD/build/pkgconfig"
(cd go && GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local go test -race ./... -count=1 -v)
(cd go && GOPROXY=off GOSUMDB=off GOTOOLCHAIN=local go build -o ../build/separate ./cmd/separate)

build/separate -threads 8 -model models/scnet-large/scnet-large.onnx \
  -out data/results/native-axis-excerpts data/excerpts/*/mixture.wav
.venv/bin/python -I tools/parity.py data/results/native-axis-excerpts data/results/excerpts-deterministic
.venv/bin/python -I tools/sdr.py data/results/native-axis-excerpts data/excerpts --baseline data/results/excerpts-deterministic
readelf -d build/separate
```

I report `precision-dp0` e `precision-dp0-axis` documentano rispettivamente
la situazione precedente e successiva alla correzione; conservarli senza
sovrascriverli. Il comando sopra produce un nuovo report con il codice corrente.
I report completi e gli audio
restano in `data/`; il riepilogo revisionabile è in questo documento e in
`docs/phase2-results.json`, quando la validazione del corpus è terminata.

## Stato del corpus

Tutti i 60 stem dei 12 estratti e dei tre brani interi superano la parità;
passa anche il confronto SDR. I dati completi, inclusi i casi peggiori per
stem, sono in `docs/phase2-results.json`; il manifest del corpus è in
`docs/corpus.json`.

| Corpus | Stem | Fallimenti | SNR minimo per stem | Peggiore finestra di 1 s |
| --- | ---: | ---: | ---: | ---: |
| 12 estratti | 48 | 0 | 94,12 dB | 87,30 dB |
| 3 brani interi | 12 | 0 | 98,93 dB | 86,40 dB |

L'errore assoluto massimo su tutto il corpus è `9.5367431640625e-07`.
Il peggior calo SDR è circa `0.00001521` dB (`other`, estratto
`10-near_silent`); tutti i cali mediani per stem sono inferiori a
`0.000001` dB. Le soglie non sono state allentate.

| Brano completo | Durata audio | Tempo di separazione | RTF |
| --- | ---: | ---: | ---: |
| PR - Oh No | 76,2 s | 72,9 s | 0,957 |
| Timboz - Pony | 253,0 s | 244,1 s | 0,965 |
| Georgia Wonder - Siren | 430,4 s | 416,5 s | 0,968 |

Il picco RSS dell'intera CLI è 12.005.940 KiB, circa 11,45 GiB, rilevato
con `/usr/bin/time -v`. Comprende Go, nucleo, runtime e buffer della CLI;
non separa pesi, memoria di lavoro e risultati. Non sono ancora applicati
budget di memoria o concorrenza; il miglioramento delle risorse appartiene
alle fasi successive. I tempi provengono da una singola esecuzione con
altro carico sul sistema, senza warm-up e cinque ripetizioni: non sono
una configurazione prestazionale raccomandata.

Per ripetere il controllo dei brani completi:

```sh
build/separate -threads 8 -model models/scnet-large/scnet-large.onnx \
  -out data/results/native-axis-full \
  'data/musdb18hq/test/PR - Oh No/mixture.wav' \
  'data/musdb18hq/test/Timboz - Pony/mixture.wav' \
  'data/musdb18hq/test/Georgia Wonder - Siren/mixture.wav'
.venv/bin/python -I tools/parity.py data/results/native-axis-full data/results/full-deterministic
.venv/bin/python -I tools/sdr.py data/results/native-axis-full data/musdb18hq/test --baseline data/results/full-deterministic
```

La prova di consumo da un progetto Go esterno è passata. Il relativo binario
è stato eseguito anche con Bubblewrap: rete disabilitata, ambiente vuoto e
filesystem contenente soltanto binario, modello, quattro librerie di sistema
e loader. Ha restituito quattro stem mono di 44.100 campioni finiti ciascuno.
Il test usa il kernel e le librerie della macchina dichiarata; altre
distribuzioni Linux e altri sistemi operativi restano da verificare.
