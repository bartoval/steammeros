# stems

Separazione musicale in stem nello stesso processo dell'applicazione: nucleo C++20 con
API C, pacchetto Go via cgo, collegamento statico. Modello: SCNet-large.

Stato: fase 1 (riferimento del modello originale prodotto e misurato). La separazione nativa non è ancora implementata.

## Build

```sh
cmake -S . -B build -G Ninja && cmake --build build
export PKG_CONFIG_PATH=$PWD/build/pkgconfig
(cd go && go test ./...)
```

`go get` da solo non basta: serve prima compilare l'archivio nativo.

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
