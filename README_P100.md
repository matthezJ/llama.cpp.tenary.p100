# README P100/GP100 - Build und Betrieb mit Ternary-Optimierungen

Stand: 2026-10-21, Code-Stand `prism-b10787` + lokale Optimierungen (unten).
Dieses Dokument beschreibt, wie dieser Fork fuer NVIDIA Tesla P100 (GP100,
sm_60) gebaut und betrieben wird, und listet alle Schalter, die die
P100-Optimierungen und ihre Diagnose steuern. Der upstreame `README.md` gilt
unveraendert fuer alles andere; Projekt-Kontext, Messprotokoll und
Patch-Doku liegen in `../docs/` (Einstieg: `../docs/README.md`).

## Was ist optimiert (Stand dieses Dokuments)

| Optimierung | Umfang | Schalter | Doku |
|---|---|---|---|
| fp16-PQ2_0-Matvec fuer GP100 (Patch A) | Decode, 1-4 Spalten, plain 2D: ersetzt auf sm_60 die emulierte dp4a-Arithmetik des Kernels `mul_mat_vec_pq2_0_ordered` durch full-rate HFMA2. **Bit-identischer Output** zum dp4a-Pfad. SASS: -27% Instruktionen je Gewicht raw, -18% issue-gewichtet. Laufzeitmessung steht noch aus (P100). | `GGML_CUDA_GP100_MMVQ_DISABLE`, `GGML_CUDA_GP100_MMVQ_R` | `../docs/12-patch-a-fp16-pq2-matvec.md` |
| conv-Fenster in place (Patch D-Rest) | Decode: ssm_conv liest das Fenster direkt aus der Cache-Zeile und schreibt das verschobene zurueck; GET_ROWS/CONCAT/CPY entfallen. 1 Token/1 Sequenz, bit-identisch. | `GGML_CUDA_GP100_STATE_DISABLE` | `../docs/14-patch-d-e-g.md` |
| GDN-Prefill 4 State-Spalten/Warp (Patch E) | Prefill: 4 Spalten je Warp auf GP100 (bit-identisch, p100-0018: +3.8% pp dort gemessen). | `GGML_CUDA_GP100_GDN_CPW_DISABLE` | `../docs/14-patch-d-e-g.md` |
| convert_unary vektorisiert (Patch G) | Prefill: 4 Elemente/Thread im kontiguen Convert (bit-identisch; p100-0005: +4.6% realer Request). | `GGML_CUDA_GP100_CONV_DISABLE` | `../docs/14-patch-d-e-g.md` |

Der fp16-Pfad greift automatisch, wenn das Runtime-Geraet cc == 600 meldet
(P100). Auf P40 (sm_61, fp16 lahmgelegt) und allem Neueren (natives dp4a)
bleibt es ausgeschaltet; der Binary verhaelt sich dort byte-identisch zum
unveraenderten Fork (SASS-Diff-Beleg in der Doku).

## Voraussetzungen

- CUDA 12.x (CUDA 13 hat Pascal entfernt; gemessen und referenziert ist
  12.4 - 12.8+ hat cuBLAS-Abweichungen, die hier zwar keinen Pfad beruehren,
  aber die Referenzmessungen des Projekts stammen aus 12.4).
- gcc/g++ 13 als Host-Compiler (nvcc 12.4 lehnt gcc > 13 ab).
- cmake >= 3.14, Ninja empfohlen. Eine GPU ist zum Bauen **nicht** noetig;
  SASS-Analyse (`cuobjdump -sass`) arbeitet auf dem Binary.
- Debian 13 (glibc 2.41) Sonderfall: der Configure-Probe von CUDA 12.4
  braucht einen `noexcept`-Patch an vier Deklarationen in
  `crt/math_functions.h` (Original daneben als `*.orig` sichern; Details
  `../docs/11-sass-baseline.md`). Auf aelteren Distro-Staenden nicht noetig.

## Build

Rein-60-Build (nur P100; Prefill fuer PQ2_0/Q2_0 laeuft ueber Dequant +
cuBLAS-hgemm, weil MMQ bei Reinarchitektur-60-Builds wegcompiliert):

```sh
cmake -B build-60 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=60 \
      -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
      -DCMAKE_CUDA_HOST_COMPILER=g++-13
cmake --build build-60 -j$(nproc)
```

60;61-Build (P100 und P40 aus einem Binary; schaltet MMQ auch auf der P100
ein - Prefill laeuft dann durch die sm_61-getunten
`mmq-config-pascal.cuh`-Tiles mit emuliertem dp4a. Gegenueber cuBLAS-hgemm
ungemessen; vor Prefill-Optimierungen erst A/B):

```sh
cmake -B build-60-61 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='60;61' \
      -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_CXX_COMPILER=g++-13 \
      -DCMAKE_CUDA_HOST_COMPILER=g++-13
cmake --build build-60-61 -j$(nproc)
```

Vor dem Mess-Build auf der Zielmaschine die tatsaechliche Compute Capability
pruefen: `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.

Beide Builds sind fuer A/B-Vergleiche aufzuheben (MMQ-Verhalten ist
build-abhaengig, siehe `../docs/02-gates-audit.md`). Neue `.cu`-Dateien
erfordern ein Reconfigure (CMake globbt).

## Schalter im Detail

### 1. Optimierungs-Schalter der P100-Patches

| Env | Default | Wirkung |
|---|---|---|
| `GGML_CUDA_GP100_MMVQ_DISABLE` | unset (fp16-Pfad **an**) | `=1` schaltet den fp16-PQ2_0-Matvec ab; Decode laeuft dann wieder durch den dp4a-Kernel `mul_mat_vec_pq2_0_ordered`. Damit ist jeder A/B ein einziges Binary (kein Cross-Build). A/B-Protokoll: `../docs/06-validierung.md`. |
| `GGML_CUDA_GP100_MMVQ_R` | `4` | Zeilen je Warp im fp16-Kernel (Template-Parameter): `2`, `4` oder `8`; jeder andere Wert faellt auf 4 zurueck. Fuer den Launch-Sweep auf der P100 (p100-Patch-0002-Muster). R aendert nur die Launch-Geometrie, nicht die Arithmetik. |
| `GGML_CUDA_GP100_STATE_DISABLE` | unset (conv-Fenster **an**) | `=1` stock-Pfad des conv-Fensters: GET_ROWS/CONCAT/CPY werden wieder ausgefuehrt statt im ssm_conv-Kernel in place zu lesen/schreiben (GP100, Decode, 1 Token/1 Sequenz). |
| `GGML_CUDA_GP100_GDN_CPW_DISABLE` | unset (4 Spalten **an**) | `=1` 1 State-Spalte je Warp im GDN-Kernel auf GP100 (Forced-1-Instantiierung, Stock-Verhalten; 4 Spalten sind default und bit-identisch). |
| `GGML_CUDA_GP100_CONV_DISABLE` | unset (vektorisiert **an**) | `=1` generischer 1-Element/Thread-`convert_unary` statt des 4-Elemente/Thread-Pfads. |

Beide MMVQ-Schalter werden einmalig pro Prozess eingelesen (static) und wirken nur
auf cc == 600; STATE/GDN_CPW/CONV ebenso. Konvention fuer kommende Patches:
`GGML_CUDA_GP100_<NAME>_DISABLE=1`
faellt auf den Stock-Pfad zurueck; schnelle Pfade sind default an
(`../docs/09-umsetzungsplan.md`, Abschnitt Env-Konvention).

### 2. Diagnose-Schalter des Forks (fuer P100-Arbeit relevant)

| Env | Default | Wirkung |
|---|---|---|
| `GGML_CUDA_BATCH_INVARIANT` | unset | `=1` zwingt 1-Spalten-Pfade auf Multi-column-Arithmetik (saubere KLD-Vergleiche). Wichtig: das PQ2_0-Gate (dp4a-Kernel und fp16-Pfad) greift nur **ohne** diesen Flag - mit Flag laeuft PQ2_0-Decode generisch. Fuer Performance-A/B unbrauchbar. (`common.cuh`) |
| `GGML_CUDA_FWHT_FUSION` | an | `=0` schaltet die Hadamard-Fusion in den Q8_1-Quantizer ab (~390 Launches je Decode-Schritt auf Bonsai 2). Negativkontrolle fuer die Launch-Zaehlung. Greift nur bei PTQ1_0-Matvecs ab Turing - der PQ2_0-Pfad quantisiert unabhaengig davon. (`ggml-cuda.cu`, `ggml_cuda_try_fwht_q8`) |
| `GGML_CUDA_AR_COPY_THRESHOLD` | 1048576 | Schwellwert (Bytes) ab dem der interne AllReduce den Copy-Kernel statt des Nickel-Pfads nutzt (Multi-GPU). (`allreduce.cu`) |
| `GGML_CUDA_AR_BF16_THRESHOLD` | siehe Code | bf16-Drahtformat des AllReduce. `=0` liefert bit-exakten Multi-GPU-Betrieb fuer Perplexity-Vergleiche. (`allreduce.cu`) |
| `GGML_CUDA_FA_PREFILL_F16` | unset | `=N` konvertiert quantisiertes K/V bis N Zellen nach f16 fuer Prefill-FA-Batches. (`fattn-common.cuh`) |
| `GGML_CUDA_DISABLE_GRAPHS` | unset | CUDA-Graphs aus. Auf Pascal sind Graphs via cc-Gate ohnehin aus (`cc < VOLTA`) und laut p100-Messung -1.4% - hier gibt es nichts zu drehen. |

### 3. Nicht setzen (P100-Klasse Dual-Socket, gemessen schlecht)

| Env | Grund |
|---|---|
| `GGML_CUDA_P2P` | 10.7x-Regression auf Dual-Socket-Boards (Peer-Access ueber den UPI-Hop), obwohl sich beidseitige Verfuegbarkeit meldet. |
| `GGML_CUDA_ALLREDUCE` | unset lassen: 25%-Verlust im 4-Karten-Report. Der interne AllReduce auf Pascal ist ohnehin noch zugated (p100-Patch 0003 offen, `../docs/04`). |

### 4. Build-Schalter (CMake), die auf der P100 eine Rolle spielen

| CMake | Default | Wirkung |
|---|---|---|
| `GGML_CUDA` | OFF | CUDA-Backend an. |
| `CMAKE_CUDA_ARCHITECTURES` | - | `60` (nur P100) oder `'60;61'` (P100+P40, aendert MMQ-Verhalten auf der P100, s. o.). |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF | `=ON` compiliert alle Quant-Kombis fuer FlashAttention - noetig, falls quantisiertes KV (q8_0/q4_0) mit `-fa` laufen soll (PrismML-Known-Issue). Kostet Build-Zeit. |
| `CMAKE_C/CXX_COMPILER`, `CMAKE_CUDA_HOST_COMPILER` | - | fest auf gcc-13/g++-13 (s. o.). |

## Betrieb (Start-Flags)

Modell vollstaendig auf die Karte; f16-KV-Cache statt q8_0 (der groesste
kostenlose Gewinn, +24% tg bei langem Kontext im qwen3.8-Referenzlauf);
Kontext immer explizit setzen (Default zieht beim 27B den vollen 262k-Kontext
und OOMt auf 16 GB); numactl auf den GPU-eigenen NUMA-Knoten (-39% sonst):

```sh
numactl --cpunodebind=0 --membind=0 \
  ./build-60/bin/llama-server -m <modell>.gguf \
    -ngl 99 -c 32768 -fa on -ctk f16 -ctv f16
```

Benchmark-Referenzzeile:

```sh
numactl --cpunodebind=0 --membind=0 \
  ./build-60/bin/llama-bench -m <modell>.gguf \
    -p 512 -n 128 -ngl 99 -c 32768 -ctk f16 -ctv f16 -fa 1 -r 3
```

A/B-Regeln: gemirrte Arme (A, B, B, A), matched thermal start (die Karte
throttelt nach ~60 s), Decode-Kalitaet via `llama-perplexity -ub 1` gegen die
F16-Referenz - plain Perplexity laeuft nicht durch MMVQ. Volles Protokoll:
`../docs/06-validierung.md`, Betriebs-Details: `../docs/05-runtime-konfiguration.md`.

## Nach dem Aendern von Kernel-Code (lokaler Loop ohne GPU)

1. Single-TU-Iteration: nvcc-Zeile aus `build-*/compile_commands.json`
   ziehen (~100 s statt ~20 min Bibliothek).
2. SASS-Zaehlung/Diff: `python3 ../tools/sass_budget.py build-60/bin/libggml-cuda.so sm_60 <symbol>`
   (Instruktionen je Gewicht gegen die Baseline in `../docs/11-sass-baseline.md`).
3. No-op-Beweis fuer andere Architekturen: `cuobjdump -sass -arch sm_61`
   (und sm_75) beider Baeume diffen - Muster: nur neue Symbole duerfen
   dazukommen (`../docs/baseline/` haelt die Pristine-Dumps und Checksummen).
4. Hostseitiger Bit-Identitaetstest des fp16-Pfads:
   `g++-13 -O2 -std=c++17 ../tools/verify_fp16_identity.cpp && ./a.out`.
5. Alles Laufzeit-Behaftige (llama-bench, KLD, test-backend-ops) ist
   Mess-Gate auf der P100-Maschine, `../docs/06-validierung.md`.
