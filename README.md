# stems

Separazione musicale in stem nello stesso processo dell'applicazione: nucleo C++20 con
API C, pacchetto Go via cgo, collegamento statico. Modello: SCNet-large.

Stato: gate tecnico CPU della fase 2 superato sul corpus locale. La separazione
nativa supera la parità per tutti i 60 stem di 12 estratti e tre brani completi.
Metodo, risultati e attività pendenti in [docs/STATUS.md](docs/STATUS.md) e
[docs/phase2-validation.md](docs/phase2-validation.md).

## Build

```sh
tools/build_ort.sh
cmake -S . -B build -G Ninja && cmake --build build
export PKG_CONFIG_PATH=$PWD/build/pkgconfig
(cd go && go test ./...)
```

`go get` da solo non basta: serve prima compilare l'archivio nativo.

Per build nativa ed esperimenti Python usare ONNX Runtime **1.31.1**, fissato in
`tools/requirements.lock`. I sorgenti nativi devono corrispondere al commit
`6af1a932e2286d5c58166085d24c68783033edee` (tag `v1.31.1`), verificato da
`tools/build_ort.sh` anche quando il checkout esiste già. Lo script rifiuta modifiche
ai file tracciati. Prima della build CMake eseguire `tools/build_ort.sh`.

L'export verifica la versione Python prima di elaborare il modello; il nucleo
verifica la versione del runtime effettivamente collegato prima di creare l'engine.
L'ambiente Python deve essere aggiornato esplicitamente, per esempio con
`.venv/bin/python -m pip install 'onnxruntime==1.31.1'`. Dopo l'allineamento occorre
rieseguire export e confronti: i risultati ottenuti con Python 1.32.0 non attestano
la parità della build nativa 1.31.1.

L'API corrente è sincrona e accetta PCM float32 mono o stereo da 8 a 384 kHz: come
nell'originale, l'audio passa a 44,1 kHz per il modello e torna alla frequenza
d'ingresso (port di `julius.resample_frac`). Le operazioni asincrone e i budget di
risorse restano nelle fasi successive; il supporto verificato è Linux x86-64 CPU.

Con più di un thread, la pipeline prepara lo spettro del segmento successivo e
ricostruisce quello precedente mentre il modello elabora il segmento corrente.
Per chiamata mantiene un segmento in preparazione e uno in ricostruzione; le iSTFT usano
fino a quattro worker su canali indipendenti. L'ordine di ricomposizione resta
invariato. `threads` regola i thread dell'inferenza, a cui si aggiungono questi
worker; con `threads=1` tutto resta sequenziale. La chiamata C/Go resta sincrona.

## Modello

SCNet-large di starrytong (commit `5d95bf9`). Licenza MIT anche per i pesi
([conferma dell'autore](https://github.com/starrytong/SCNet/issues/35#issuecomment-4999873539)).
Quattro stem (drums, bass, other, vocals), 44,1 kHz stereo. Sorgenti e checksum in
`models/manifest.json`; i pesi non sono in Git.

## Criteri

- Parità con l'originale PyTorch FP32 (con `shifts=0`; l'originale usa uno shift casuale):
  SNR della differenza ≥ 50 dB per stem; canali, lunghezza e ordine degli stem esatti.
- Qualità: SDR sul test set di MUSDB18-HQ; calo mediano massimo 0,1 dB per stem.
  Tracce da [Zenodo 3338373](https://zenodo.org/records/3338373), estratte in `data/musdb18hq/test/`.
- Build CPU: solo librerie di sistema (libc, libm, libstdc++, libgcc_s).
