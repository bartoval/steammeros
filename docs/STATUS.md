# Stato delle fasi

## Fase 0 — parziale

Il commit `fc9a891` contiene la prova minima C++/Go con archivio statico.
Mancano l'inventario hardware/software, `docs/acceptance.md` completo e la
definizione dei budget di memoria e concorrenza. La prova di build non è stata
rieseguita durante questa revisione.

## Fase 1 — riferimento locale disponibile; valutazione da completare

Sono presenti localmente riferimenti PyTorch per 12 estratti di 25 secondi,
in modalità deterministica e originale, e tre brani interi in modalità
deterministica. Tempi, SDR e manifest del corpus sono sotto `data/`, esclusa
da Git. Gli ascolti umani e la copertura delle categorie acustiche restano
da verificare. Checkpoint e configurazione sono identificati in
`models/manifest.json`.

## Fase 2 — gate tecnico CPU superato

Sono implementati l'export ONNX, STFT/iSTFT C++, il collegamento statico di
ONNX Runtime e una prova Go sincrona. Il binario aggiornato `build/separate`
è stato ricompilato offline e ha dipendenze ELF soltanto da libc, libm,
libstdc++ e libgcc_s. Un progetto Go esterno produce stem reali anche in un
filesystem isolato contenente soltanto eseguibile, modello, queste quattro
librerie e loader, con rete disabilitata.

L'indagine iniziale riportava circa 54 dB complessivi e 13 stem su 48 sotto
i criteri di parità. La diagnostica sugli stessi ingressi ha isolato la causa
nelle riduzioni molto lunghe di GroupNorm, mentre le LSTM isolate superano
127 dB. Con riduzioni per asse il primo DualPathRNN passa da 71,65 a 137,53 dB.
La rete completa raggiunge 126,74 dB sullo spettro del segmento di prova.
Tutti i 60 stem dei 12 estratti e dei tre brani interi ora superano la parità
e i criteri SDR. Il minimo SNR sull'intero stem è 94,12 dB; quello della
peggiore finestra è 86,40 dB, contro soglie invariate di 50 e 40 dB.
Il peggior calo SDR è 0,00001521 dB.

Sono state applicate e verificate correzioni per validazione dell'output ONNX, race tra
`StemNames` e `Close`, allineamento delle versioni del runtime e verifica dei
sample rate nella parità. I test Python sono passati, compreso il caso
GroupNorm con dimensioni realistiche, e l'intera suite Go è passata con
race detector, inclusa la separazione con modello reale.

La versione scelta e installata è ONNX Runtime 1.31.1 per Python e C++.
Le precedenti misure con Python 1.32.0 non costituiscono validazione della
build nativa 1.31.1.

La diagnostica riproducibile è `tools/diagnose_precision.py`. Il riferimento
rimane originale: le riscritture di FFT, GroupNorm e normalizzazione sono
applicate soltanto al percorso di export e poi ripristinate.

Metodo e comandi sono in `docs/phase2-validation.md`, tutte le misure per
stem e i checksum in `docs/phase2-results.json`, il corpus in `docs/corpus.json`.
Il picco RSS della CLI sui brani completi è 12.005.940 KiB (11,45 GiB):
comprende processo Go, nucleo, runtime e buffer della CLI; non misura
separatamente pesi e memoria di lavoro. Le risorse non hanno ancora un budget
applicato e verificato.

## Fasi successive

Il codice sincrono anticipa parti delle fasi 3 e 4; non ne dimostra il
completamento. Ricampionamento, API asincrona, cancellazione e budget
verificati restano da realizzare. Le ottimizzazioni di memoria e prestazioni
sono previste nella fase 5, dopo il consolidamento del motore e dell'API.
