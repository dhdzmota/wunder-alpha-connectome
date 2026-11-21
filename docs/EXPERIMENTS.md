# Bitácora de experimentos — Alpha Connectome

Registro consolidado de lo que probamos, qué funcionó, qué no y por qué. Complementa `HANDOFF.md`
(notas cronológicas de cada sesión). Métrica: weighted Pearson con la máscara `is_scored` ("valid_mask").
Última actualización: 2026-10-18.

## 1. Estado actual

- **Mejor público: 0.6611** — `sub11_tcn` (FT minGRU-EMA + FT SSM + FT TCN64, int16), OOF 0.6871, 53m11s servidor.
- Alternativa rápida: `sub11b_tcn_mg` (FT minGRU-EMA + FT TCN), OOF 0.6862, ~26 min servidor → ~25 min libres de presupuesto.
- Top-8 (snapshot 2026-10-08): ≈0.668 público → faltan ≈ +0.007.
- Calibración: público ≈ OOF − 0.026 (estable entre 0.0236 y 0.0277). Ruido del LB ≈ ±0.001.

## 2. Submissions

| sub | contenido | valid / OOF | público | servidor | lección |
|---|---|---|---|---|---|
| baseline | GRU 2×128 ONNX | 0.6171 | 0.5935 | — | offset valid→público ≈ −0.024 |
| sub01 | LightGBM ligero (numba) | 0.6444 | 0.6178 | 12m57s | numba disponible en el scorer |
| sub02 | GRU128x2 + LightGBM | 0.6721 | 0.6449 | 33m41s | servidor ≈ 1.5× más rápido que local |
| sub04 | GRU corr + pmask, runtime fusionado | 0.6765 | 0.6488 | 21m52s | pérdida de correlación + peso P(scored) ayudan |
| sub05 | GRU + minGRU | 0.6811 | 0.6548 | 37m32s | ensamble de 2 arquitecturas +0.006 |
| sub06 | GRU + minGRU corr3_pm05 | 0.6822 | 0.6548 | 37m46s | +0.001 valid no se ve en público |
| sub07 | GRU + minGRU **fine-tune en todo valid** (mix4, 3 ép) | 0.6838 | 0.6585 | 36m31s | **palanca más fuerte**: público ganó más que CV |
| sub08a | GRU desde cero train+valid + FT minGRU | 0.6845 | 0.6583 | — | desde cero con valid no supera al FT |
| sub09 | FT minGRU + EMA z (span 2000) + FT SSM | 0.6863 | 0.6604 | 57m01s | features relativas por secuencia ayudan |
| sub09b | sub09 con runtime optimizado | 0.6863 | 0.6604 | 46m21s | fastmath + matvec 4 filas: 0.81× real |
| sub10 | minGRU-EMA + SSM144-EMA | 0.6863 | 0.6597 | ~46 min | mejor miembro individual ≠ mejor blend |
| **sub11** | minGRU-EMA + SSM + TCN64 (int16) | **0.6871** | **0.6611** | 53m11s | TCN = miembro barato |

Empaquetadas y no enviadas: sub08, sub09q16/q8, sub11b, sub12 (G+TCN-rank, 0.6859), sub12c (G+S+X, 0.6870).

## 2b. Palancas ordenadas por ganancia de ranking (2026-10-18)

Ranking estimado interpolando el snapshot del 2026-10-08 (797 equipos; único punto observado: sub02 = #574).

| paso | palanca | tipo | público | Δ score | ranking | Δ ranking |
|---|---|---|---|---|---|---|
| sub01→sub02 | LightGBM → GRU (+LGBM) | familia de modelo | 0.6178→0.6449 | +0.0271 | >650→574 | +100 o más |
| sub02→sub04 | pérdida de correlación + peso P(is_scored) | alinear con la métrica | 0.6449→0.6488 | +0.0039 | 574→~473 | **+101** |
| sub04→sub05 | 2.º miembro (minGRU) | diversidad (1.ª vez) | 0.6488→0.6548 | +0.0060 | ~473→~290 | **+183** |
| sub05→sub07 | fine-tune en valid | alinear con la distribución de test | 0.6548→0.6585 | +0.0037 | ~290→~164 | **+126** |
| sub07→sub09 | EMA z por secuencia + SSM | representación + diversidad | 0.6585→0.6604 | +0.0019 | ~164→100 | +64 |
| sub09→sub11 | 3.er miembro (TCN) | diversidad | 0.6604→0.6611 | +0.0007 | 100→~83 | +17 |
| sub06, sub08a, sub09b, sub10 | ajustes / mismo enfoque | — | ±0.0007 | 0 | 0 | 0 |

Lectura: la diversidad rinde cada vez menos (+183 → +64 → +17). Las palancas que alinean el entrenamiento con
la evaluación (métrica, máscara, distribución de test) dieron ≥ +100 cada una y no se han agotado del todo.

## 3. Registro por línea de trabajo

Screenings: GRU, 4 épocas, 40 % de train, referencia indicada. **Ruido de screening ≈ ±0.001.**

### Pérdida y pesos — ✅ adoptado
- corr-w 1 → 3: scr_corr1 0.6698 → scr_corr3 0.6753. Peso P(is_scored)^γ: γ=0.5 (scr_corr3_pm05 0.6753); γ=1/2 ≈ igual o peor.
- Hiperparámetros: lr ×4 (scr_lr4 0.6675) ✗; lr bajo, wd 1e-3, h192, 3 capas, chunk 1k: todos ≤ base.
- bs64 (más secuencias por paso de corr, aproximación a pérdida global): 0.6754 vs 0.6748 → ruido. ✗

### Fine-tune en valid — ✅ la mejor palanca
- mix4 (peso 1 + 4·is_scored), lr 3e-4, 3 épocas, cross-fit en 2 mitades (`ft_crossfit.py`).
- mix16 en SSM: +0.0004 ✗. Warm start train+valid (`--init-from`) ✗. Desde cero train+valid (sub08a) = FT en público.
- **Sweep de receta 2026-10-18 (TCN):** 6 ép 0.6838, lr 1e-3 0.6837 vs actual 0.6836 (blend 0.6872–0.6873 vs 0.6871) → receta ya óptima.
  `_lr1e4` y `_wd05` no corrieron (disco lleno); no se relanzaron.

### Representación de entrada
- ✅ EMA z-score causal por secuencia (span 2000): mejora en modelo completo (FT minGRU 0.6823 → 0.6839).
- ✗ Features LightGBM como input (scr2_feats 0.6710 vs 0.6751).
- ✗ Reloj de eventos absoluto (scr_ev 0.6719, scr_ev_ema2k 0.6718).
- ✗ Sin niveles crudos (scr_noraw 0.6722).
- ~ Rango causal (scr_rk 0.6756), reloj relativo (scr_evr 0.6755): +0.0008 en screening, combinados 0.6738 → era ruido.
  Modelo completo TCN-rank: OOF 0.6827 < 0.6836. Cerrado.
- **Patrón:** lo absoluto/hecho a mano empeora; solo lo relativo por secuencia ayudó. Coherente con AUC adversarial 0.9956 (train vs valid).

### Arquitecturas — saturado
- GRU, minGRU, SSM (ssm16 0.6801, ssm144-EMA 0.6830), TCN64 (0.6818), atención local 32/128 (≈ base): todos con corr ≈ 0.98 entre sí.
- Targets auxiliares (scr2_aux 0.6734) ✗. Targets pasados (scr_past 0.6752) ≈.
- Stacking sobre OOF (lineal/GBDT): ✗ (GBDT memoriza secuencias).
- Corrección del nivel por secuencia: oráculo +0.034 pero no recuperable causalmente (idea B) ✗.
- Calibración causal post-hoc del blend sub11 (2026-10-18, scratchpad calib.py): p' = exp(θ·z)·(p − β·EMA(p)), z = log posición,
  log RMS causal de p (spans 500/5000); ajuste en una mitad de valid, evaluación en la otra. OOF 0.6866–0.6868 vs 0.6871;
  incluso dentro del ajuste gana < +0.0005 y θ cambia de signo entre mitades → los modelos ya calibran amplitud y nivel. ✗

### Robustez al cambio de régimen
- Augmentation afín por grupo: aug 0.10 → 0.6739, 0.25 → 0.6741 vs 0.6748 ✗.
- Re-ponderar train por P(valid) adversarial (`artifacts/adv/train_pvalid.npy`: mediana 6e-4, solo 101 secuencias con p>0.5;
  pesos p/(1−p) crudos dejan tamaño efectivo 0.7 %). `--seq-weights K`: w = 1 + K·rank(p)², normalizado a media 1.
  | screening | semilla 1 | semilla 2 |
  |---|---|---|
  | referencia | 0.6748 | 0.6748 |
  | K=3 (pesos 0.5–2.0, ESS 0.83) | 0.6758 | 0.6757 |
  | K=10 (pesos 0.23–2.5, ESS 0.68) | 0.6753 | 0.6753 |
  Efecto pequeño pero consistente en screening. **Modelo completo ✗:** `tcn64_ema_sw3` base 0.6817 (= 0.6818), FT OOF 0.6836 (= TCN),
  corr 0.993 con el TCN sin pesos; sub13 (G+S+Tsw) OOF 0.6871 = sub11. Lección: ganancias de screening al 40 % de train no
  sobreviven a train completo + FT en valid.

### Latencia (restricción dura: 60 min, 1 vCPU)
- numba con fastmath propio (tanh/sigmoid/exp sin libm), matvec de 4 filas, jitclass: 0.75× local → 0.81× real en servidor.
- Pesos int16 (escala por fila): 0.82× sin cambio de score; int8 0.80× con error rms 0.003.
- Calibración: minutos servidor ≈ 0.45 × µs/fila local. Ignorar la duración interina que muestra la plataforma sin score.

## 4. Aciertos
- Medir latencia antes de invertir en un modelo; runtime numba propio en vez de ONNX.
- Cross-fit en mitades de valid para evaluar el fine-tune sin usar el LB como juez.
- Umbral de envío (≳ +0.002 OOF): evitó gastar subs en ruido (sub06, sub10 lo confirmaron).
- Miembros baratos (TCN) para liberar presupuesto de latencia.

## 5. Errores y problemas operativos
- **Disco WSL lleno (2026-10-18):** cachés uv (52 GB), pip (15 GB) y HF (37 GB) + 31 GB de predicciones de experimentos cerrados.
  Mató el sweep a media corrida y bloqueó la salida de comandos. → Ver política de retención (§6) y revisar `df -h /` antes de colas largas.
- **Throttling térmico:** la GPU se frena a ≥74 °C (épocas de 60 s → 255 s, reloj 1600 → 630 MHz). Una corrida de 8 épocas llegó a ~18 h.
- Colas muertas por reinicio de WSL/laptop sin aviso (queue_mingru, queue_ftsweep del 13-oct): revisar logs vacíos al retomar.
- Bug de semilla RNG con overflow en la primera versión de `--aug` (queue_aug, rc=1); corregido y relanzado.
- numba `cache=True` no detecta cambios en funciones llamadas → borrar `wnn/__pycache__/*.nb[ic]` tras editar kernels.
- Screenings con diferencias de +0.0008 tomados como señal (rank/evr) → costaron un modelo completo. Exigir ≥ +0.0015 o 2 semillas.
- `pkill -f` dentro de un comando mata la propia shell; matar por PID explícito.

## 6. Política de retención de artifacts (`~/data/wnn/artifacts`)
- **Siempre se conservan:** pesos (`best.pt`, `last.pt`, ~1–3 MB), `history*.json`, `ema.json`, `eval_groups.npy`, logs, submissions.
  Con los pesos, cualquier predicción se puede regenerar.
- **Predicciones de valid (`pred_valid.npy`, 300 MB) solo para miembros vivos:**
  bases `mingru_ema_c3`, `ssm16_c3`, `tcn64_ema_c3` (para rehacer cross-fit) y OOF staged
  `ft_mingru_ema_c3_mix4_full`, `ft_ssm_mix4_full`, `ft_tcn64_ema_c3_mix4_full`, `ft_ssm144_ema_c3_mix4_full`, `ft_tcn64_x_c3_mix4_full`.
- **Se borran:** `pred_ep*.npy` de `nn_ft/` una vez staged el OOF, `pred_valid*.npy` de screenings y líneas cerradas,
  matrices de features LightGBM (`gbdt/*_F.npy`, regenerables con el script de features).
- Limpieza 2026-10-18: 153 archivos de predicciones (30.9 GB) + gbdt F (4.9 GB). Lista en `artifacts/logs/deleted_2026-10-18.txt`.

## 7. Experiencia ajena: concursos similares (revisión 2026-10-18)

Ya revisados antes: Jane Street (2024) y el reto RNN previo de Wunder (ventanas > stateful → lo probamos: win500 +0.0006 valid).
Fuentes nuevas (writeups 1er lugar; varios vía resúmenes secundarios porque Kaggle bloquea la descarga):

| Concurso | Ganador hizo | ¿Aplica aquí? |
|---|---|---|
| Optiver Trading at the Close (2023) | CatBoost .5 + GRU .3 + Transformer .2, mismas 300 feats; **reentrenamiento online** con labels del test (5.444 → 5.403) | Online con labels: **no** (no recibimos targets). Lección: NN secuenciales + GBDT con pesos buscados en valid = lo que ya hacemos |
| Ubiquant (2022) | LGBM + TabNet, 400 feats, objetivo tipo Pearson | Ya usamos pérdida de correlación ✓ |
| G-Research Crypto (2022, Pearson ponderado) | LightGBM; "features >> modelo" | GBDT ya probado (0.651 < NN 0.68); no retomar |
| Google Brain Ventilator (2021, secuencias simuladas) | LSTM/CNN-Transformer, **muchos seeds/folds + ensamble mediana/media**, target cuantizado explotado (PID, redondeo) | Bagging de seeds: no cabe en latencia → **destilación**. Redondeo: no ayuda a Pearson |
| CMI Detect Sleep States (2023, series largas) | CNN downsample → GRU residual → CNN upsample (multirresolución) | Versión causal: rama GRU "lenta" que actualiza cada k pasos (costo 1/k) — no probado |
| Amex Default (2022) | LGBM + GRU; destilación con soft labels del ensamble | **Destilación ensamble→alumno**: no probado |

Ya probado y descartado aquí (no repetir): targets auxiliares (0.6734 ✗), capacidad en screening (H192, 3 capas ✗), augmentation (ruido), calibración de amplitud (✗).

## 8. Plan de acción (2026-10-19, quedan 27 días; meta top-8 ≈ 0.6688, actual 0.66134 = −0.0075)

| Fase | Qué | Costo GPU | Puerta (gate) |
|---|---|---|---|
| 0 | LB de sub15 (ventanas) + OOF 4-way (G + Gwin + S + T) | 0 | define 1 y 2 |
| 1 | Ventanas en SSM: screening ref vs win500 (40%, 4 ép.) → si gana, SSM full win + FT mix16 → sub16 | ~7 h + ~18 h | sub15 LB ≥ sub14b |
| 2 | Destilación: maestro = promedio de N modelos (seeds/regímenes) → alumno del mismo tamaño entrenado con y + soft targets | días | OOF 4-way − sub15 ≥ +0.002 |
| 3 | Rama GRU lenta (actualiza cada k=20 filas sobre promedios): screening minGRU | ~1.5 h/run | screening ≥ ref +0.001 |
| 4 | SWA (promedio de pesos últimas épocas): screening + aplicable a cualquier full run sin reentrenar si se guardan checkpoints | bajo | ≥ ref |
| — | Diario: usar los 5 slots como probes del LB; última semana: elegir finales por LB + OOF, no solo LB | — | — |
| — | Pendientes: sub14c (lr 1e-3, listo), cross-fits de sub14, borrar artifacts (comando dado al usuario) | — | — |

**Fase 0 resultado (2026-10-19):** sub15 LB 0.66110 (−0.0002 vs sub14b) → fase 1 ✗. minGRU-win OOF 0.6845 (corr 0.987 con minGRU viejo); 4-way 0.6874 vs 0.6871 (+0.0003) → fase 2 (bagging→destilación) ✗. Meseta: todos los miembros corr ≈ 0.98; cambios de régimen de entrenamiento producen la misma función.

### Diagnóstico de errores del blend sub11 (OOF valid, 2026-10-19; script scratchpad diag.py)
- Posición en la secuencia: uniforme (0.67–0.70); sólo [1000,2000) t0 0.615 (5% de filas). Sin bolsa.
- Amplitud por régimen (quintil de RMS del target por secuencia): pendiente y~p 0.91–1.03 → calibrada; confirma §calibración ✗.
- **Secuencias con máscara densa** (quintil Q5 de filas scored, 20% de filas scored, **43% de las filas del métrico**): WP 0.615 vs 0.72–0.78 en Q1–Q4. WP en todas las filas: Q5 0.544 ≈ Q4 0.578 → en Q5 la máscara es poco selectiva (lift 0.61/0.54 vs 0.78/0.42 en Q1). Parece en parte inherente; es la única bolsa grande.
- Oráculos con labels (cotas superiores, sobreajustadas): escala óptima por secuencia 0.7079, desplazamiento de media por secuencia 0.7119, magnitud |y| con signo de p 0.7144 (base 0.6871).

### Cierres 2026-10-19 (madrugada)
- Screenings vs scr_ema2k 0.6747/0.6748: rama GRU lenta k=20 0.6750, k=100 0.6750, EMA de pesos 0.998 0.6747 → ✗ (empates). EMA de pesos acelera épocas tempranas, mismo final.
- Runtime: el matvec actual (int16, 4 filas/pasada) es el más rápido de los probados (8 filas, columna-mayor, f32; versión entera int16×int16 7–10× más lenta: numba no vectoriza). Holgura de latencia ya existente (52/60 min) no tiene uso: G+SSM144+T 0.6870 vs G+S+T 0.6871; 4-way +0.0004 → capacidad/miembros extra no suben el blend. Línea cerrada.
- Reconstrucción de targets desde cambios futuros de features (para calibración online con labels retrasados): R² lineal 0.07–0.14 a cualquier horizonte → no reconstruibles; ✗.
- Stacker LightGBM sobre residuos (filas scored de valid, 120 feats: fila + preds + contexto causal; 2 mitades): OOF 0.6745 vs 0.6871, peor en todos los quintiles (Q5 0.593 vs 0.615) → sin señal extraíble en residuos; bolsa de máscara densa ✗.
- Combos de 3 miembros (OOF): mejor Gw+S+S144 0.6873 (+0.0002) pero el runtime exige SSMs de igual forma → no empaquetable.

- 2026-10-19 LB: **sub16_ftmix32** (FT mix32, 3 ép., lr 3e-4, estructura sub14b) = **0.66132** (53m11s) vs sub14b 0.66134 → empate. mix4→mix16 dio +0.0002 (sub11→sub14b), mix16→mix32 nada: dirección de peso scored saturada.
- 2026-10-19 LB: **sub14c_ftmix16_lr1e3** (FT mix16, lr 1e-3, 4 ép.) = **0.66105** (52m57s) vs sub14b 0.66134 → −0.0003: FT más agresivo no ayuda. Línea de hiperparámetros de FT cerrada; sub14b sigue siendo el mejor.
- 2026-10-19 17:18: **apuesta maestro** lanzada — GRU 384×3 + EMA 2000 (2.79M params, 11× sub-miembro), train completo 8 ép. (~90 min/ép., GPU 74 °C) → FT mix16 cross-fit → OOF. Puerta: ≥ 0.690 (blend 0.6871 + 0.003) → destilar; si no, cerrar. Script logs/queue_teacher.sh.
- 2026-10-20 02:00: maestro **por debajo** de los miembros chicos en ES (es_pm por época 0.529/0.526/0.532/0.536/0.541/0.544/0.547 vs minGRU-EMA 128×2 0.543…0.558) → puerta fallida; detenida la cola antes del FT (se deja terminar la época 8 para registrar valid). Más capacidad no ayuda tampoco con datos completos.
- 2026-10-20 04:52: maestro FINAL valid 0.6740 (best ep 7) vs minGRU-EMA 128×2 0.6801, GRU 128×2 0.6767 → **peor**. Línea maestro/destilación cerrada.

### 2026-10-20 — Mirada nueva: el target tiene 4 regímenes de tick (hallazgo)
- Cada secuencia tiene su propia unidad de target: y = k·u_s con u_s ∈ {0.1254, 0.2413, 0.4685, 0.6569} (igual para t0 y t1;
  residuo de rejilla 0). Archivos: `artifacts/preds/units_{train,valid}.npy` (unidad por secuencia, cols t0/t1).
- Mezcla de secuencias train vs valid: 12/12/33/43 % vs 21/19/23/37 % → valid tiene ~2× regímenes finos (posible parte del shift).
- Blend sub11 OOF por régimen (fracción del peso del métrico / WP): 0.125 → 28 % / 0.800; 0.241 → 16 % / 0.773;
  0.469 → 8 % / 0.736; **0.657 → 48 % / 0.434** (std pred 0.56 vs y 0.89). La "bolsa de máscara densa" es este régimen grueso.
- El régimen se identifica causalmente: LightGBM con stats de las 99 filas de warm-up, train→valid acc 0.990 (1000 filas: 0.998).
- Calibración afín por régimen optimizando el WP global (cross-fit mitades, 3 repeticiones): Δ −0.0001 ± 0.0001 → ✗.
  Los modelos ya ubican nivel/escala por régimen; lo que falta es poder predictivo DENTRO del régimen grueso.
- Ruido: WP de mitades aleatorias de valid va de 0.6725 a 0.7010 → diferencias de LB de ±0.0003 no son señal.
- **Especialista del régimen grueso (2026-10-21).** `train_gru.py --regimes 3 --class-head --ce-w K` (cabeza de clases de tick,
  predicción E[|y|y]/E[|y|]); `finetune_valid.py` con las mismas opciones. WP en valid de régimen grueso (697 secuencias):
  | modelo | semilla 1 | semilla 2 |
  |---|---|---|
  | referencia general scr_ema2k | 0.4178 | — |
  | especialista GRU regresión (screening) | 0.4268 | 0.4264 |
  | especialista GRU clases ce 1 (screening) | 0.4289 | 0.4293 |
  | especialista GRU clases ce 0.3 | 0.4281 | — |
  Escala completa `mingru_ema_cls3` (minGRU-EMA, 8 ép., best ep 5): base 0.4294 (= screening; minGRU general 0.4256),
  FT mix4 OOF 0.4332 (FT minGRU general 0.4292; blend sub11 0.4337). Mezclado en el blend sólo en secuencias gruesas (peso 0.4–0.5):
  OOF global 0.6877 vs 0.6871 (+0.0006; mitades +0.0003 / +0.0009) → bajo la puerta de +0.001; runtime no construido aún.
  Gate: `artifacts/regime/proj_spec.py`. Cola: `logs/queue_spec.sh`.
- **Especialistas con cabeza de clases en los 4 regímenes (screening GRU, 4 ép.)** vs scr_ema2k por régimen:
  r0 0.7953 vs 0.7864 (+0.009), r1 0.7699 vs 0.7587 (+0.011), r2 0.7279 vs 0.7182 (+0.010), r3 0.4289 vs 0.4178 (+0.011).
- **Runtime con ruteo por régimen (2026-10-21):** `wnn/regime.py` (regresión logística multinomial sobre stats de las 99 filas de
  warm-up; `artifacts/regime/detector.npz`; acc valid 0.9989 con entradas float32, 1 error grueso-vs-resto),
  `submission_template/routed/solution.py` (un FusedModel por régimen; todos corren el warm-up, desde la fila 99 sólo el elegido),
  cabeza de clases en `mingru_infer.cls_point` / `fused.py` (`mg_cls_v`), `scripts/package_routed.py`.
  **sub17_route3** = sub14b + régimen grueso → especialista FT mix16 .50 + S/T de sub14b reescalados. Checks: runtime nuevo = sub14b
  (diff 0), especialista numba vs torch 1.6e-5, ruteo = modelo directo (diff 0). Latencia A/B vs sub14b 1.002 / 1.004 (~52.5 min).
  OOF esperado ≈ +0.0005 (con mix4).
  ⚠ sub17_route3 falló en el servidor: "PRERUN TIMEOUT: solution timed out after 60 seconds on prerun dataset" — el arranque
  (compilación numba) cuenta en el prerun de 60 s. Cada FusedModel compilaba su propia jitclass (arranque local 91 s vs 57 s sub14b).
  Fix: `fused._holder` cachea la jitclass por tipos (una sola compilación para todos los modelos) y la plantilla routed precompila sólo
  float32. **sub17b_route3**: mismas predicciones, arranque ≈0.75× sub14b. Regla: medir arranque en frío (descomprimir zip) antes de enviar.
- **Especialistas completos r0–r2 (2026-10-21, `logs/queue_spec012.sh`, 02:06–03:18):** minGRU-EMA cls, 8 ép. → FT mix4 f0/f1 + mix16 full.
  Base valid_mask r0 0.7964 (ep5), r1 0.7695 (ep4), r2 0.7277 (ep5). Gate `artifacts/regime/proj_all.py` (ruteo con detector, G reemplazado
  por el especialista a peso wx, S/T reescalados), vs sub11 0.6871 (h0 0.6907 / h1 0.6837):
  sólo r0 +0.0005, sólo r1 +0.0003, sólo r2 +0.0001, sólo r3 +0.0005; **los 4 a wx 0.5 → 0.6886 (+0.0015; h0 +0.0011, h1 +0.0019)** → pasa la puerta.
  Paquete **sub18_route4** (sub14b + 4 especialistas mix16 a 0.5), 14.6 MB zip; log `logs/package_sub18.log`.
- LB 2026-10-21: **sub17b_route3 = 0.66155** (52m47s, 1.4 GB) vs sub14b 0.66134 → +0.0002, nuevo mejor; runtime con ruteo validado.
  sub18_route4 verificado: checks OK (numba–torch ≤1.6e-5, ruteo = directo), latencia 114.3 µs/fila (sub14b 118.3), arranque en frío 30 s (sub14b 42 s).
- **wx por régimen (2026-10-21, `artifacts/regime/proj_wx.py`):** ajustado en h0 [.3,.6,.5,.4] → h1 0.6852 vs 0.6856 con .5; ajustado en h1
  [.6,.6,.5,.7] → h0 0.6914 vs 0.6918; in-sample 0.6886 = .5 → ✗ (sobreajuste). Se queda wx = 0.5 en todos los regímenes.
- LB 2026-10-22: **sub18_route4 = 0.66226** (+0.0007 vs sub17b, +0.0009 vs sub14b; OOF +0.0015). Servidor **55m17s** pese a latencia local
  114 vs 118 µs (sub14b): el A/B local no captura el costo de 5 modelos en el servidor (1/2/5 modelos → 52m17/52m47/55m17).
  Margen ≈ 4.7 min → no agregar miembros sobre esta estructura; reemplazos de pesos (mismas formas) sí.
- **FT por régimen de SSM/TCN (2026-10-21, `logs/queue_rft.sh`, gate `artifacts/regime/proj_rft.py`):** FT mix4 de ssm16_c3 / tcn64_ema_c3
  sólo con las secuencias de valid del régimen. Por régimen S_r/T_r vs S/T general: r0 +0.0007/+0.0007, r1 +0.0003/−0.0002,
  r2 −0.0006/−0.0004, r3 +0.0012/+0.0006. Blend ruteado wx .5: S+T por régimen 0.6887 vs sub18 0.6886 (h0 +0.0002, h1 −0.0001) → ✗.
  El FT general sobre todo valid ya captura la adaptación; la ganancia de los especialistas viene del entrenamiento por régimen sobre train
  + cabeza de clases, no del FT. Cola detenida antes de los FT full mix16.
- **TCN especialistas con cabeza de clases (screening 4 ép., 2026-10-21, `logs/queue_tscr.sh`):** r3 reg 0.4242 → cls 0.4265 (+0.0023,
  igual que en minGRU); r0 cls 0.7948. Runtime con cabeza de clases en el TCN listo (`tcn_infer` / `fused` tcn_cls_v; paridad 1.7e-5).
  Diversidad (scratchpad tcn_div.py, en el régimen): corr(mg_spec, tcn_spec) 0.965 / 0.986 vs corr(mg_spec, T general) 0.956 / 0.981.
  Reemplazar T general por tcn_spec en el blend tipo sub18: r3 0.4341 → 0.4337, r0 0.7999 → 0.7990; 4-way 0.4340 / 0.7998 → ✗ a escala
  screening. Los especialistas entrenados sobre el mismo subconjunto se correlacionan más entre sí que con los generales. Pendiente: si
  vale la pena una corrida completa + FT sólo en r3 (el tcn_spec completo sería más fuerte; el minGRU spec pasó de 0.4289 a 0.4332).
- **TCN especialista r3 completo (2026-10-21, `logs/queue_tspec.sh`, gate `artifacts/regime/proj_tspec.py`):** tcn64_ema_cls3 8 ép.
  (best ep 7, base 0.4267) → FT mix4 OOF 0.4276 vs T general 0.4268 (+0.0008; el minGRU spec dio +0.004 sobre su general).
  Blend tipo sub18 con T → TCN spec en r3: 0.6885 vs 0.6886 (h0/h1 −0.0001) → ✗. Línea de TCN especialistas cerrada.
  FT mix16 full guardado en nn/ft_tcn64_ema_cls3_mix16_full (no empaquetado).
  ⚠ Bug corregido en los gates proj_tspec/proj_rft: el detector manda 1 secuencia de otro régimen a r3 (698 vs 697), y el fallback
  "si hay algún NaN usar S/T general" descartaba todo r3; ahora el reemplazo es por secuencia. Re-corrido proj_rft: mismos números (cierre válido).

## 9. Retrospectiva y fase A "detective" (2026-10-21/22)

Retrospectiva: las ganancias vinieron de alinear con la evaluación (métrica, máscara, distribución de valid) y de
descubrir estructura del target (regímenes de tick); "más modelo" (arquitectura, capacidad, ventanas, aug, stacking,
calibración, destilación) no movió nada. El private LB usa un dataset distinto → margen de latencia y
reproducibilidad importan (sub18 deja ~4.7 min). Commit de snapshot `47155e0` (incluye `pipeline/queues` y
`pipeline/regime_gates`, copiados de artifacts). Scripts de esta fase: scratchpad de la sesión (a1*, a2*, a3*).

- **A.1 Shift dentro de régimen:** AUC adversarial train-vs-valid por régimen r0 0.995, r1 0.999, r2 0.997, r3 0.995
  (global 0.9955) → el shift **no** es la mezcla de regímenes. Univariado máx. AUC 0.73 (|Δ| de volúmenes i0: valid
  ~1.7× más activo) y valid cae dentro de p1–p99 de train en casi todos los features → separación por combinación
  (huella de periodo/día), no por estar fuera de soporte. No reabre líneas cerradas por argumento de shift. ✗
- **A.2 Estructura conjunta t0/t1:** y = k·u con k entero; pares simétricos a t0↔t1 ((0,−1)≈(−1,0), (−3,1)≈(1,−3)),
  impares negativos favorecidos, k0+k1 casi siempre ≤ 0 → consistente con t0 ≈ P&L de comprar ahora / vender después y
  t1 ≈ el inverso: suma S ≈ −(spread actual + futuro), diferencia D = dirección. Cross-corr t0/t1 simétrica en lag.
  S = 5–18 % de la varianza; modelo la predice con corr 0.05–0.34. Oráculo con S verdadera: 0.6844 → 0.7316, pero
  Ŝ causal (LGBM fila actual, cross-fit) corr 0.18–0.37 y el stack p+Ŝ = 0.6833 vs 0.6839 → la parte útil de S es
  futura (spread futuro), no capturable. ✗ (la estructura es real; sirve para entender D).
- **A.3 Error del régimen grueso (OOF FT minGRU-EMA, espacio D = t0−t1):** oráculo de dirección perfecta r3 0.429 → 0.765;
  magnitud perfecta → 0.467; saber cuándo no hay movimiento → 0.431 → el déficit es de **dirección**, no de timing.
  AUC de dirección (D≠0) r3 0.806 vs 0.88–0.89 en r0–r2. LGBM sólo con la fila actual (cross-fit por secuencias en valid)
  da AUC 0.836 en r3 (top: volúmenes i0_v21/v4/v0) — pero el AUC de D es un proxy engañoso: en t0 las filas (0,−1) pesan 0,
  lo que cuenta es ordenar k0 ∈ {−3,−1,+1}. Stack LGBM(fila + p) mezclado a 0.3, 2 particiones: global 0.6849 vs 0.6839
  (+0.001, mitades +0.001), r0 +0.003, r1 +0.003, r2 +0.001, **r3 −0.0017**; a ≥ 0.5 empeora. Sobre el blend sub18 la
  ganancia sería menor y no cabe en latencia (margen 4.7 min). ✗
- **Conclusión fase A:** la estructura existe (P&L long/short, S = spread futuro, déficit de dirección en el régimen grueso),
  pero ninguna palanca causal barata sale de ella. La información instantánea de la fila no aporta más que el NN en r3.

## 10. Adaptación al periodo de test entre secuencias (2026-10-24) ✗
Idea: el runtime ve todas las secuencias de test en un proceso → acumular stats del periodo y alinear las entradas a train.
Cota superior en valid (stats de TODO valid, modelo base `mingru_ema_c3` sin FT; scratchpad `sim_xseq.py`):
sin cambio 0.6801 (r0–r3 .7942/.7677/.7256/.4256), media/desv. por columna 0.6795, mapeo de cuantiles 0.6768.
Marginales casi iguales (|Δμ|/σ 0.08, ratio σ 1.055) → el shift es de relación entrada→target (sólo lo captura el FT con labels),
no de covariables. Sin labels de test no hay ganancia → cerrado (la duda de reglas sobre estado entre secuencias queda irrelevante).

## 11. Segundo especialista r3 en lugar del SSM general (2026-10-24) ✗
Motivo: el SSM cuesta ~26 min de servidor (sub19 sin S: 0.66130 en 29m21s vs sub18 0.66226) y aporta −0.0008 OOF si se quita
(0.6886 → 0.6878 en los 4 regímenes; sólo en r3 → 0.6882). Cola `pipeline/queues/queue_spec2.sh`, gate `pipeline/regime_gates/proj_spec2.py`.
| especialista r3 (seed 2) | base valid r3 | corr con spec actual | noS + spec2 (mejor w) | S+T + spec2 |
|---|---|---|---|---|
| `--joint-head` (softmax sobre pares k0,k1) | 0.4257 | 0.969 | 0.6884 (r3 .4356) | 0.6885 |
| cabeza normal (control) | 0.4271 | 0.979 | 0.6885 (r3 .4359) | 0.6886 |
Referencias: sub18 0.6886 (r3 .4363), r3 sin S 0.6882 (r3 .4348); spec cls3 seed 1 base 0.4294.
El 2.º especialista recupera ~3/4 de lo que da el SSM, nunca lo supera; la cabeza conjunta no mejora (base −0.0014 vs control).
→ cerrado como palanca de score; `--joint-head` cerrado. Sólo serviría para una variante más rápida que sub18 a −0.0002 OOF,
y sub19 (29 min) ya cubre ese papel.

## 12. Ronda de análisis del régimen grueso r3 (2026-10-24; scratchpad a4_*.py, blend OOF sub18)
- WP por secuencia en r3: mediana 0.47 (p5 0.12, p95 0.70); cuartil peor = 31 % del peso. Peor con máscara densa y más
  actividad a2/a3/a6, mejor con más trades de i0 — mismo patrón en r0 (no es específico de r3).
- Estructura (k0,k1) en filas scored de r3: **69 % son (0,−1)/(−1,0)** ("sin movimiento"), 33 % del peso, aporte a la covarianza ≈ 0
  (−1 %). El modelo separa el lado (p0 −0.3 vs −0.9) pero no movimiento vs no movimiento (p1 −0.91 en (0,−1) vs −0.99 en (1,−3)).
- Oráculos (ajuste lineal con interacción, r3): base 0.4363; + flag movimiento 0.5005; + lado 0.8658; ambos 0.9701.
  AUC movimiento: blend 0.61, distribución del especialista P(k=0) 0.69; AUC lado 0.81.
- Lectura de la distribución de clases (stack lineal cross-fit con marginales): 0.4326–0.4266 vs 0.4329 → ✗ (el punto ya la contiene).
- Simulación: un score de movimiento independiente con AUC 0.71/0.80/0.87 daría r3 +0.004/+0.009/+0.017 (≈ global +0.002/+0.005/+0.008).
- LGBM dedicado a movimiento (fila + historia causal: filas desde último cambio de precio/trade, conteos en 100/500/2000 filas),
  cross-fit en mitades de valid r3: AUC 0.655; con P(k=0) del especialista 0.665 < 0.692 del especialista solo → ✗.
- Conclusión: la palanca grande de r3 es "¿habrá movimiento?" (vale más que mejorar la dirección a nivel realista), pero el NN ya
  extrae más de lo que da la historia de eventos explícita. Sin fuente nueva de información, cerrado.

## 13. ¿El peso de la máscara verdadera en el FT sobreajusta a la máscara de valid? (2026-10-24; scratchpad a5_mask.py)
WP en distintos conjuntos de filas (cross-fit por secuencias; "alt" = filas no scored con P(scored) estimada en el top 10 %):
| FT | scored | no scored | todas req. | alt |
|---|---|---|---|---|
| GRU sin FT | 0.6767 | 0.4720 | 0.5040 | 0.5748 |
| GRU FT pm05 (datos valid, sin máscara real) | 0.6783 | 0.4749 | 0.5066 | 0.5750 |
| GRU FT mix4 | 0.6790 | 0.4752 | 0.5070 | 0.5741 |
| GRU FT sólo máscara | 0.6797 | 0.4663 | 0.4999 | 0.5698 |
| SSM FT mix4 (h0) | 0.6882 | 0.4836 | 0.5151 | 0.5774 |
| SSM FT mix16 (h0) | 0.6886 | 0.4817 | 0.5137 | 0.5757 |
Lectura: la ganancia del FT viene de los DATOS de valid (pm05 mejora en todos los conjuntos). Más peso a la máscara real compra
+0.0004–0.0007 en las filas scored a costa de −0.001/−0.002 en filas parecidas no scored → se especializa en el proceso de
selección de la máscara. En público mix16 ≈ mix32 ≈ mix4 + 0.0002 (ruido) → neutro allí; riesgo asimétrico si la máscara del
private se genera distinto. Candidata de cobertura: estructura sub18 con FT mix4.

## 14. Técnicas sólo en el FT sobre valid (2026-10-24; cola `pipeline/queues/queue_ftaug.sh`, gate `pipeline/regime_gates/proj_ftaug.py`)
Especialista r3 `mingru_ema_cls3`, FT mix4 cross-fit (WP r3, mitades h0/h1). Nuevas opciones de `finetune_valid.py`:
`--seed --aug --noise --fdrop --crop --replay --replay-w --freeze`; `scripts/ft_soup.py` (promedio de pesos de FT).
| variante | r3 | h0 | h1 | Δ |
|---|---|---|---|---|
| referencia (seed 42) | 0.4334 | 0.4477 | 0.4191 | — |
| seed 11 / seed 12 | 0.4348 / 0.4331 | .4497/.4476 | .4200/.4185 | +0.0014 / −0.0004 |
| E1 soup 3 seeds | 0.4341 | 0.4486 | 0.4196 | +0.0007 (media de seeds 0.4338) |
| E2 recortes 10k, 6 ép. | 0.4313 | 0.4477 | 0.4148 | −0.0022 |
| E3 afín s .1 / .25 / .1 6 ép. | 0.4335 / 0.4333 / 0.4320 | | | +0.0001 / −0.0001 / −0.0014 |
| E4 ruido .05 + dropout .1 | 0.4190 | 0.4349 | 0.4030 | −0.0145 (cero no es neutro en features cuantizadas) |
| E5 replay 341 train r3 ×0.25 | 0.4328 | 0.4480 | 0.4175 | −0.0007 |
| E6 núcleo congelado | 0.4330 | 0.4478 | 0.4182 | −0.0005 |
| E7 espejo bid↔ask | — | | | no factible: sin emparejamiento 1-1 bid↔ask (|corr| .5–.79, muchos→uno), precios no simétricos, lado de trades desconocido |
Ruido de semilla del FT ≈ ±0.001 (rango 0.0018). Ninguna técnica supera el ruido; el soup queda ligeramente sobre la media de sus
seeds (reduce varianza, latencia 0). Lectura: el FT en valid está saturado — la información de valid ya se extrae con la receta actual.

## 15. Soups de 5 semillas de FT (2026-10-25; `pipeline/queues/queue_soup.sh`, gate `pipeline/regime_gates/proj_soup.py`)
| modelo (FT mix4 cross-fit) | 5 semillas | media (sd) | promedio de preds ×5 | soup de pesos |
|---|---|---|---|---|
| especialista r3 (WP r3) | .4332 .4349 .4322 .4318 .4338 | 0.4332 (0.0011) | 0.4338 | 0.4337 |
| TCN general (WP valid) | .6836 .6833 .6837 .6836 .6834 | 0.6835 (0.0001) | 0.6836 | 0.6836 |
Blend tipo sub18/sub20 con soups (r3, TCN o ambos): 0.6886 = referencia en ambas mitades. El soup iguala al ensamble ×5 sin costo
de latencia, pero en el blend la varianza de semilla ya está amortiguada → 0. No se construye sub21. ✗

## 16. Ronda de análisis r0–r2 (2026-10-25; scratchpad a6_regimes.py, a6_slag.py; blend OOF sub18)
Descomposición S = k0+k1, D = k0−k1. Oráculos (ajuste lineal con interacción, por régimen) sobre WP base:
| régimen | base | + lado | + movimiento (|D|>1) | + tamaño |D| | + S | AUC lado | corr(S, p0+p1) |
|---|---|---|---|---|---|---|---|
| r0 | 0.8021 | 0.9061 | 0.8112 | 0.8474 | 0.8516 | 0.941 | 0.35 |
| r1 | 0.7756 | 0.9098 | 0.7886 | 0.8261 | 0.8278 | 0.928 | 0.31 |
| r2 | 0.7403 | 0.9094 | 0.7767 | 0.7794 | 0.7762 | 0.926 | 0.27 |
| r3 | 0.4363 | 0.8658 | 0.5034 | 0.5098 | 0.5211 | 0.810 | 0.06 |
- Lado: ya AUC 0.93–0.94 en r0–r2 (sin margen realista). Movimiento: vale poco fuera de r3. S: +0.04–0.08 en todos y el modelo apenas lo sigue.
- S: varianza entre secuencias 2–5 %; acf dentro de secuencia L100 ≈0.37, L200 ≈0.2, L500 ≈0.05.
- Oráculo con S real retrasado L filas: r0–r2 L100 +0.010–0.012, L200 +0.003–0.004; r3 L100 +0.035, L200 +0.016.
  Si S fuera spread del libro, S_(t−100) sería observable en t. Prueba: LGBM con fila t, fila t−100 y su diferencia → S_(t−100) en r3
  (cross-fit): corr 0.36, R² 0.125, stack p + Ŝ 0.4340 = p sólo (oráculo 0.4669) → S NO es spread observable en los features;
  contiene información que el libro no muestra (ejecuciones/llenados). ✗ Confirma el cierre de la fase A, ahora para los 4 regímenes.

## 17. Representación de entradas en el especialista r3 (2026-10-25/26; wnn/repr_feats.py, `--extra-feats na,pb`, queue_repr.sh, gate proj_repr.py)
Perfil de columnas: precios ~40 valores distintos por secuencia y un valor en el 57 % de filas (casi categóricos);
a0–a7 con centinela −5.2 en el 7 % de valores (33 % en a7), que también contamina la EMA.
- na: centinela de a0–a7 → 0 antes de crudo/diff/EMA + 8 indicadores. pb: 8 tramos lineales (PLE) por cada una de las 44 columnas
  de precio, cortes fijos = cuantiles de valores distintos en train (artifacts/repr/pb_edges.npy). 244k → 290k parámetros.
| semilla | ref sin FT | na+pb sin FT | ref FT r3 | na+pb FT r3 (h0 / h1) | Δ |
|---|---|---|---|---|---|
| 42 | 0.4294 | 0.4302 | 0.4334 | 0.4329 (0.4477 / 0.4181) | −0.0005 |
| 2 | 0.4271 | 0.4294 | 0.4315 | 0.4321 (0.4460 / 0.4182) | +0.0006 |
- Media +0.0000 tras el FT (gate +0.0015 ✗). Sin FT +0.0015 de media, pero el FT en valid lo absorbe. Ablación na / pb cancelada. ✗
- Época de entrenamiento r3 ≈5 min sin contención (antes medí 14 con otra carga); corrida completa + FT ≈1 h.

## 18. G1: codificador por fila (MLP residual) antes del minGRU, especialista r3 (2026-10-26; `--row-enc 256`, queue_rowenc.sh, gates proj_repr.py / proj_rowenc.py)
Hipótesis: todos los miembros mezclan las columnas de una fila con una sola capa lineal; más interacción no lineal entre columnas
(paso previo a un GNN con adyacencia aprendida = atención entre las 112 columnas, G2) podría captar estructura del libro. 244k → 452k parámetros.
| semilla | ref sin FT | G1 sin FT | ref FT r3 | G1 FT r3 (h0 / h1) | Δ |
|---|---|---|---|---|---|
| 42 | 0.4294 | 0.4303 | 0.4334 | 0.4332 (0.4491 / 0.4173) | −0.0003 |
| 2 | 0.4271 | 0.4289 | 0.4315 | 0.4341 (0.4483 / 0.4200) | +0.0026 |
- Media +0.0012, una semilla en contra → gate ✗ (borde). En el blend tipo sub18 (OOF 0.6886): G1 s2 0.6887, G1 s42 0.6886,
  promedio de 2 especialistas ref 0.6886 / G1 0.6887 / ref s42 + G1 s2 0.6888 (+0.0002, y costaría un 2.º especialista) → ✗.
- Mismo patrón que §17: mejora pequeña antes del FT que el FT en valid absorbe. G2 (GNN/atención entre columnas) no se implementa:
  el eje "interacciones entre columnas dentro de la fila" queda cerrado.

## 19. Paso atrás 2026-10-26: kumo-engine, encadenamiento de secuencias, prior de nivel por periodo (scratchpad stitch_*.py, knn_*.py, ft_leak.py)
- **NVIDIA kumo-engine** (github.com/NVIDIA/kumo-engine): motor para datos relacionales multi-tabla (grafo por llaves foráneas,
  GNN por lotes en GPU, churn/fraude/recomendación). Sin tablas que relacionar ni inferencia de baja latencia en CPU → no aplica. ✗
- **Hallazgo:** las secuencias son tramos contiguos de un flujo. Última fila de A vs primera de B (fracción de columnas idénticas):
  pares con ≥ 0.7 tienen t0 final de A = t0 inicial de B (corr 1.000; sin filas repetidas → adyacencia, no traslape).
  Fracción con sucesor ≥ 0.7 (null = fila 10 de B): valid→valid 0.30 (null 0.13), train→train 0.43 (0.22), **valid↔train 0.000** →
  train y valid son periodos disjuntos sin frontera común; valid parece una muestra de un periodo (cadenas cortas, 685 grupos,
  máx. 24 secuencias) → hipótesis: test intercalado con valid (coherente con FT en valid ganando más en público que en CV).
- **Reglas** (`docs/data_overview.md`): "Sequences are independent: do not join them or carry state across them" → coser en inferencia prohibido.
- **Persistencia del nivel entre vecinos** (1188 enlaces mutuos, OOF sub18): nivel del target corr 0.48/0.51, nivel predicho 0.80/0.84,
  **nivel del residuo 0.05/0.02**. Desplazar por el residuo del predecesor verdadero empeora (0.6886 → 0.6852 con λ .25) →
  el modelo ya captura toda la parte persistente; ni coser (ilegal) aportaría nivel. ✗
- **Idea 1, prior de nivel por kNN/ridge sobre huellas causales** (media/desv./|Δ|/frac. cambios de las primeras N filas, 448 dims,
  PCA 32; folds agrupados por cadena vs aleatorios): corr(r̂, r) 0.00–0.14; mejor 0.6885–0.6886 vs 0.6886 (oráculo desde N=1000: 0.7152). ✗
- **Idea 2, ¿las mitades de FT filtran por vecinos de cadena?** 51 % de los enlaces cruzan h0/h1, pero la ganancia del FT no es mayor
  con vecino en la mitad de entrenamiento (minGRU +0.0033 vs +0.0035 vecino en la misma mitad vs +0.0068 sin enlace; SSM y TCN igual) →
  los gates de cross-fit siguen siendo válidos; no hace falta rehacerlos. ✗ (sin acción)
- **Idea A, mapa no lineal conjunto y_j = g_rj(p0, p1) por régimen** sobre el blend OOF sub18 (scratchpad mapA.py; rejilla de cuantiles
  B×B, media por celda con pesos del métrico, encogida hacia un ajuste lineal; 5 folds agrupados por cadena; mezcla (1−a)·p + a·g):
  2D mejor 0.6888 (a .25–.5; h0 +0.0001, h1 +0.0002; r3 0.4363 → 0.4365), 1D 0.6886, a = 1 empeora (0.6880). El blend ya es casi
  óptimo como función de (p0, p1) → +0.0002 no justifica tocar el runtime. ✗

## 20. Idea 3: EMA de memoria larga (spans 2000,10000) en el especialista r3 (2026-10-26; queue_ema.sh, gate proj_ema.py)
| semilla | ref sin FT | lm sin FT | ref FT r3 | lm FT r3 (h0 / h1) | Δ |
|---|---|---|---|---|---|
| 42 | 0.4294 | 0.4288 | 0.4334 | 0.4308 (0.4462 / 0.4155) | −0.0026 |
| 2 | 0.4271 | — | 0.4315 | 0.4310 (0.4453 / 0.4167) | −0.0005 |
Peor en ambas semillas → memoria más larga no aporta (coherente con §19: el residuo de nivel no es recuperable). ✗

## 21. Idea D: cabeza binaria "¿habrá movimiento?" sólo de entrenamiento en el especialista r3 (2026-10-27; `--move-w`, queue_move.sh, gate proj_move.py)
BCE sobre |k0−k1| > 1 (no es par de un tick (0,−1)/(−1,0)), ponderada como la CE de la cabeza de clases; runtime sin cambios.
| variante (seed 42) | sin FT r3 | FT r3 (h0 / h1) | Δ vs 0.4334 | blend tipo sub18 (OOF 0.6886) |
|---|---|---|---|---|
| move-w 1 | 0.4306 | 0.4344 (0.4494 / 0.4194) | +0.0010 | 0.6887 (r3 .4369) |
| move-w 3 | 0.4318 | 0.4340 (0.4506 / 0.4174) | +0.0006 | 0.6887 |
| promedio mv1 + mv3 (2 especialistas) | | | | 0.6888 |
Mismo patrón que §17/§18: gana antes del FT (+0.0012/+0.0024), el FT lo absorbe; en el blend ≤ +0.0002. Etapa 2 saltada. ✗

## 22. Idea B: stack LightGBM por fila sólo en r0–r2 sobre el blend sub18 (2026-10-27; scratchpad stackB*.py)
Entradas: fila (112) + diferencia con la fila previa (112) + blend p (2) + miembros spec/S/T (6); filas scored de valid; pesos |y|;
mezcla (1−a)·p + a·stack; r3 sin tocar. Cross-fit en las mitades del FT y en 2 folds agrupados por cadena.
| split | árboles × hojas | a | global (base 0.6886) | mitades vs base | r0 / r1 / r2 (base .8021/.7756/.7403) |
|---|---|---|---|---|---|
| mitades FT | 200×31 | .2–.35 | 0.6890 | +0.0003/+0.0005 | .8029 / .7767 / .7400 |
| mitades FT | 400×63 | .2 | 0.6891 | +0.0005/+0.0005 | .8031 / .7768 / .7405 |
| cadenas | 200×31 | .35 | 0.6892 | +0.0008/+0.0005 | .8031 / .7773 / .7404 |
| cadenas | 400×63 | .35 | 0.6893 | +0.0009/+0.0005 | .8034 / .7773 / .7399 |
Ganancia real y consistente (todas las mitades suben) pero chica: +0.0004–0.0007; viene de r0/r1, r2 ≈ 0; a ≥ .5 empeora.
Costo: r0–r2 = 63 % de las secuencias; 200×31 ×2 targets = 400 árboles por fila (estimado 8–15 µs en esas filas) vs margen sub18 ≈ 10 µs.
- **Variante para runtime** (sólo fila + diferencia + blend, sin miembros individuales; sólo r0/r1): igual o mejor (cadenas 200×31 a .35:
  0.6893). **Intercambio elegido para sub21:** en r0/r1 quitar el SSM (spec .6 / TCN .4, como sub19) y poner ahí el stack →
  OOF 0.6889 (mitades FT +0.0003/+0.0002) / 0.6891 (cadenas +0.0005/+0.0005) vs sub18 0.6886, **con menos latencia**.
  Árboles: ~0.09 µs por árbol en numba (400 por fila en r0/r1); el SSM cuesta ~37 µs/fila.
- **sub21_route4_stack** (2026-10-27): copia de sub18 (su runtime `wnn_rt` intacto) + `wnn_rt/stack.py` + `submission_template/routed_stack`;
  stacks finales `scripts/train_stack.py` (todo valid; r0 988k filas, r1 534k; 200 árboles/target, a .35) → artifacts/stack/;
  empaquetado `scripts/package_stack.py --name sub21_route4_stack`. Chequeos: árboles numba = LightGBM (0.0), ruteado vs offline 1.2e-7,
  latencia ponderada por mezcla de regímenes 106–107 vs 120 µs/fila sub18 (0.88–0.89× → ~49–51 min servidor), arranque en frío
  33.9 s (sub18 33.5 s). Trampa evitada: quitar el SSM con n_ssm = 0 cambia los tipos numba → 2.º jitclass, arranque 57 s;
  se dejan las llaves ssm_* con dimensión inicial 0 (mismos tipos, el bucle no corre). Stacks entrenados con preds OOF mix4; sub21 usa los
  miembros mix16 de sub18 (diferencia de preds mínima). **Público 0.66255 (48m51s, 1.4 GB)** — nuevo mejor, +0.00029 vs sub18, 6.5 min más rápido.
- **Gates post-sub21** (scratchpad stackB4.py, base sub21-like 0.6889 mitades / 0.6891 cadenas):
  - Stack en **r3** ✗: a .15/.25/.35 → −0.0003/−0.0008/−0.0015 (mitades), −0.0001/−0.0005/−0.0010 (cadenas); r3 es ruidoso (corr .436), los árboles sobreajustan.
    Stack en **r2**: ±0.0000. No hay más que sacar del stack fuera de r0/r1.
  - **SSM de vuelta en r0 bajo el stack**: +0.0002 / +0.0002 (ambas divisiones, ambas mitades); en r1 +0.0001. sub22_route4_stack_r0ssm empaquetado
    (stack r0 reentrenado sobre el blend con SSM; sub21 respaldado en artifacts/stack_sub21/). Paridad OK, pero **latencia r0 150 vs 86 µs**
    (el SSM cuesta ~65 µs en r0, no ~37 como estimé) → ponderado 120.7 µs = 1.01–1.03× sub18 → ~55–57 min en servidor: fuera de la regla ≤52 min.

## 23. Etiquetas suaves de un maestro con vista al futuro (2026-10-27; idea de las notas públicas de jy447)
Mecanismo: un maestro que en entrenamiento lee las filas futuras da E[y | pasado + futuro cercano], una versión
menos ruidosa del target; los alumnos (desplegables, causales) se entrenan contra (1−a)·y + a·maestro, pesos (|y| + 0.2)·rw.
- Maestro `teacher_la50_gru`: GRU 2×128 + `--extra-feats fut` (wnn/fut_feats.py: media de las filas t+1..t+L − x_t, L 2/5/12/25/50,
  columnas i0 0..51), sólo train, 4 ép: es_pm 0.7725 → 0.7979; **valid 0.8025 / todas 0.746** (causal ~0.682 / 0.51). Sin fuga en valid.
- Gate FT por mitades (`--soft`, mix4, pipeline/regime_gates/proj_soft.py):
  | miembro | FT normal | suave a .8 | suave a .5 |
  |---|---|---|---|
  | TCN (todas) | 0.6836 (0.6871/0.6804) | **0.6845** (0.6879/0.6812) | 0.6843 (0.6878/0.6810) |
  | especialista r3 (filas r3) | 0.4338 (0.4477/0.4191) | **0.4355** (0.4483/0.4218) | — |
  | blend tipo sub18 con ambos cambiados | 0.6886 | 0.6889 | — |
  Control de la confusión de chunk (el TCN suave usó --chunk 1000): FT normal chunk 1000 = 0.6874 / 0.6803 → ganancia real de
  las etiquetas suaves en el TCN +0.0005 / +0.0009 (chunk solo: +0.0003 / −0.0001, ruido). Se mantiene.
- sub22 (queue_sub22.sh): estructura de sub20 con los 6 FT completos con etiquetas suaves a .8 + swap de stack de sub21.
- Siguiente (escenario B): `scripts/teacher_label.py` (etiquetas del maestro en todo train) + `train_gru.py --soft-train` para reentrenar bases.
- **sub22_route4_soft_stack** (2026-10-27 20:40): 6 FT completos en todo valid con etiquetas suaves (mix4, a .8; especialistas r0–r3,
  TCN, SSM; `queue_sub22.sh`) en estructura sub20 (`sub22_route4_soft`, 121.7 µs ≈ 55 min) + swap de stack de sub21 (`--base sub22_route4_soft`).
  Paridad numba–torch ≤3e-5, ruteado 0.0, stacks = los de sub21 (¡ojo!: artifacts/stack/stack_r0 había quedado con la versión
  r0+SSM; movida a artifacts/stack_r0ssm/ y restaurada la de sub21 antes de empaquetar). Latencia 0.87–0.89× base ≈ 49 min;
  arranque en frío 34.8 s; 12.6 MB. **Público 0.66277** (+0.00022 vs sub21). OOF esperado ≈ +0.0006–0.0010 sobre sub21 (gate: TCN +0.0007, r3 +0.0017 en filas r3).
- Escenario B lanzado (2026-10-27 noche): `scripts/teacher_label.py` → artifacts/teacher/teacher_la50_gru_train.npy (float16);
  `queue_soft_tcn.sh`: tcn64_ema_c3_st8 desde cero con `--soft-train` (a .8) + FT suave por mitades (chunk 1000) vs el sa8 actual.
  Después `queue_teacher_conv.sh`: maestro `teacher_la64c_gru` = `--arch la` (train_gru.LANet: GRU causal + convs dilatadas con
  compuertas leídas hacia atrás sobre las 64 filas futuras, proyección inicializada en 0) + `--extra-feats fut`; probado: alcance 63 filas,
  por chunks = sin chunks (3e-6).
- tcn64_ema_c3_st8 (TCN desde cero con etiquetas suaves): es_pm por época 0.5495 / 0.5547 / 0.5595 / 0.5617 vs TCN original
  0.5351 / 0.5480 / 0.5540 / 0.5553 (mejor final 0.5603 en ép 6) → +0.006 a igual época, ya supera al original en la ép 3.

## 24. Plan de acción (2026-10-27 noche; quedan 18 días; mejor público sub22 0.66277; top-10 0.6711 = +0.0083)
Nota: las 4 rutas tienen especialista → el minGRU del modelo default sólo corre en el calentamiento; no se reentrena.
Miembros que importan: especialistas r0–r3, SSM, TCN.
| Fase | Fechas | Qué | Gate | Sub |
|---|---|---|---|---|
| 0 | 27-oct noche | TCN desde cero con etiquetas suaves (queue_soft_tcn) + maestro conv (queue_teacher_conv) | TCN tras FT ≥ +0.002 vs 0.6845; maestro conv > 0.8025 | — |
| 1 | 28-oct | elegir maestro, etiquetar train; FT cruzado del maestro en valid (5 folds); gates FT: no-puntuadas 0.1, lr 1e-4 | ≥ +0.0005 ambas mitades | — |
| 2 | 28–30-oct | escenario B: especialistas r0–r3 + SSM desde cero con --soft-train; FT por mitades; stacks r0/r1 reentrenados | blend OOF ≥ +0.0015 | sub23 (~0.665–0.667) |
| 3 | 31-oct–4-nov | ensamble de maestros (look-ahead + causales); miembro "fusion" multi-EMA (1/8, 1/64, 1/512) destilado | ≥ +0.001 blend | sub24–25 |
| 4 | desde 29-oct (CPU) | decodificador de movimiento: ¿t0 + t1 ≈ −spread? reconstrucción de ticks/spread | estructura existe → entradas nuevas, gate r3 | sub26 |
| 5 | 10–15-nov | finales (agresiva + robusta), código limpio con semillas, snapshot git, borrador del reporte | — | finales |
Reglas: gates con números y mecanismo descartado; buscar notas de la comunidad cada 3–4 días; latencia ≤ 52–53 min;
laptop sin suspensión; disco antes de etiquetar; nada pesado en RAM mientras se entrena.
- Gate "etiquetas retrasadas del maestro" (2026-10-27 22:40, scratchpad delayed.py): en t, la salida del maestro la50 para la fila t−L
  (L ≥ 50) es causal. Stack lineal por régimen (mitades de valid) con p(t), q(t−L), residuo q−p retrasado y sus EMAs.
  Control sólo p: 0.6827 (el reajuste por régimen pierde −0.0058 en el métrico global); con features del maestro 0.6824–0.6825 → **±0 / −0.0003** ✗.
  Oráculo con y real retrasado: L50 0.8435 (+0.155), L100 0.7789 (+0.090).
  **Mecanismo descartado:** el residuo persistente (y − p, acf alta a 50–100 filas) NO está en el libro, ni siquiera mirando el futuro;
  la ganancia del maestro (0.80) es la componente rápida. El reentrenamiento con etiquetas suaves ataca esa componente; la lenta no es recuperable.
- Gate D "secuencias de máscara densa" (scratchpad dense.py; densidad causal = EMA_2000 de P(scored)): WP del blend por quintil
  0.743 / 0.742 / 0.743 / 0.699 / **0.449**; el quintil denso es 93 % r3 (r3 denso 0.38). Escalas por bin (y régimen × bin), cross-fit:
  −0.0001 / −0.0003 ✗. La bolsa densa = r3; asignar varianza no ayuda → el esfuerzo va a la calidad en r3, no a una ruta nueva.
- Etiquetas del maestro en train in-sample vs out-of-fold (scratchpad teacher_gap.py, 250 seqs c/u): corr ponderada por P(scored)
  0.6486 vs 0.6488 → el maestro no memorizó; no hacen falta maestros cross-fit.
- Código nuevo: `--soft-conf c` (peso × exp(−c (y − maestro)²), etiquetas que el maestro llama ruido), `--fut-lags` (horizontes del
  maestro; wnn.fut_feats.set_lags; teacher_label.py los lee de history). `queue_label_screen.sh` (tras el maestro conv): etiquetas la64c,
  maestro corto la12 (2,5,12) + etiquetas, mezcla la50+la12, y screening TCN (frac 0.4, 4 ép): plain | la50 | la50+conf1 | la64c | la12 | mix.
  Decide la receta de etiquetas para los reentrenamientos de la fase 2.
- Plan §24 actualizado (2026-10-27 22:45): ideas nuevas en fase 3 — (1) capacidad reabierta por destilación (r3 2×256 / 3 capas; latencia pagada
  con el stack r0/r1), (5) maestro adaptado a valid (FT cruzado del maestro), (6) auto-destilación gen 2 (ensamble nuevo = maestro).
  Orden por promesa: reentreno con etiquetas suaves > capacidad r3 ≈ receta de etiquetas > ensamble de maestros > hidden distill > maestro-valid
  > gen 2 > move decoder > TTT > multi-EMA. Nota operativa: GPU throttled 75 °C / 712 MHz / 16.6 W a las 22:38 (≈3–4× más lento).
- **Gate escenario B — TCN desde cero con etiquetas suaves (tcn64_ema_c3_st8, 2026-10-27 23:40) ✓**: valid sin FT 0.6856 vs 0.6818 (+0.0038).
  Tras FT suave por mitades: **0.6873 (h0 0.6907, h1 0.6841) vs 0.6845 (0.6879 / 0.6812) → +0.0028 / +0.0029** (el FT absorbe ~25 %).
  Blend sub18-like: 0.6895 (0.6926 / 0.6865) vs 0.6888 con el TCN sa8 → **+0.0007 ambas mitades** (vs sub18-like 0.6886: +0.0009).
  → fase 2 confirmada: reentrenar desde cero especialistas (r3 primero) y SSM con etiquetas suaves.
- Prioridad del usuario (2026-10-28 00:00): **subs frecuentes para iterar por LB**. Cola única secuencial `queue_night2.sh`
  (sin esperas pgrep): sub23a (TCN st8) → `queue_patient.sh` (TCN st16 = destilación paciente, 16 ép; gate + paquete sub23b) →
  maestro conv → screening de recetas de etiquetas. Cada reentreno que pase su gate se empaqueta como sub propio.
- **sub23a_route4_softtcn_stack** (2026-10-28 00:15): sub22 con el TCN cambiado por tcn64_ema_c3_st8 (FT suave completo, chunk 1000).
  Paridad numba–torch ≤3e-5 (TCN 7.8e-7), ruteado 0.0, stack 1.2e-7, stacks = sub21. Latencia nueva 104.1–105.6 µs/fila (= sub22,
  ≈49 min; la rep 0 de la base salió ruidosa). Arranque en frío 36.0 s (con la GPU entrenando). 12.6 MB. OOF esperado +0.0007 vs sub22. **Público 0.66338 (+0.00061), 48m02s.**
- **Destilación paciente (tcn64_ema_c3_st16, 16 ép vs st8 8 ép; 2026-10-28 07:00)**: es_pm se aplana en 0.5663 desde ép 12 (= st8); valid sin FT
  0.6862 vs 0.6856 (+0.0006), all 0.5174 vs 0.5150. Tras FT suave por mitades: 0.6881 (0.6915 / 0.6848) vs 0.6873 (0.6907 / 0.6841) → +0.0008 / +0.0007.
  Blend: 0.6897 (0.6929 / 0.6867) vs 0.6895 → **+0.0002 / +0.0003** (consistente pero chico). 2× GPU (≈5 h con throttling) por +0.0002 →
  fase 2 queda en 8 épocas; 16 sólo si sobra GPU al final. sub23b se empaqueta igual (prueba LB barata).
- **sub23b_route4_softtcn16_stack** (2026-10-28 07:30): = sub23a con el TCN de 16 ép. Paridad TCN numba–torch ≤1.1e-6, stack 1.2e-7, stacks = sub21,
  código de runtime idéntico a sub23a (md5). Latencia y arranque en frío locales ruidosos por el maestro conv entrenando + CPU throttled:
  arranque desde unzip fresco sub23a 69 s / sub23b 78 s / sub22 87 s medidos seguidos (sub22 corrió bien en el servidor) → ruido local, no regresión.
  Ojo: re-ejecutar cold.py en la misma carpeta es caliente (numba cache=True escribe __pycache__); medir siempre desde unzip fresco.
  OOF esperado +0.0002 vs sub23a (debajo del ruido del LB). **Público 0.66364 (+0.00026), 50m12s.**
- **Maestro conv teacher_la64c_gru** (`--arch la` + fut, 4 ép, 2026-10-28 08:50): es_pm por época 0.8819 / 0.8924 / 0.8999 / 0.9016 (la50: 0.7725 → 0.7979);
  **valid mask 0.8805 vs 0.8025 (+0.078)**, all 0.853 vs 0.746. Convs aprendidas sobre 64 filas futuras ≫ medias fijas.
  Cola reordenada (prioridad subs): `queue_la64c.sh` = etiquetas la64c → TCN desde cero con la64c (receta sub23a, tcn64_ema_c3_c8) → FT suave
  (valid preds la64c) por mitades → gate → FT completo → sub23c. Screening (la12, mix, conf) aplazado.

**2026-10-28 TCN con etiquetas del maestro conv (tcn64_ema_c3_c8, receta sub23a, 8 ép.):** es_pm 0.5640 (st8 0.5663),
valid 0.6842 (st8 0.6856); FT suave por mitades 0.6861 (st8 0.6873, st16 0.6881); blend 0.6892 (st16 0.6897) ✗.
Mejor maestro ≠ mejor alumno (brecha de capacidad: la64c ve 64 filas futuras, etiquetas menos aprendibles por un causal;
loss de entrenamiento 2.4 vs 1.8). sub23c empaquetada pero NO se sube. Lección: la calidad del maestro en valid no predice
el valor como etiqueta; filtrar por el alumno. Siguiente (fase 2): especialista r3 desde cero con la50 → sub23d
(queue_p2_r3.sh). Pendiente para etiquetas: mezcla la50/la64c o maestro conv con menos futuro (16–32 filas).

**2026-10-28 Fase 2 r3 (mingru_ema_cls3_st, receta cls3, desde cero con la50 a .8, 8 ép.):** es_pm 0.3885 (orig 0.3770, best
ep 7 → aún subiendo), valid r3 0.4382 (orig 0.4294); FT suave por mitades r3 rows 0.4410 vs 0.4355 (solo FT suave);
blend con st16 **0.6906 vs 0.6897 (+0.0009, h0 +0.0010 h1 +0.0005)** ✓ → sub23d_route4_r3scratch_stack (~47 min, cold 38 s).
Siguiente: r2 desde cero (queue_p2_r2.sh → sub23e). Ojo: base r2 del gate = FT plano por mitades (no hay mitades FT suave cls2).
- LB sub23c 0.66333 (53m01s) vs sub23b 0.66364: −0.0003 (OOF −0.0005) → confirma OOF; descartar etiquetas la64c tal cual. Tiempo servidor 53 min con 105 µs/row local: varianza ±3 min.

**Fase 2 r2 (mingru_ema_cls2_st, desde cero la50):** es_pm 0.5654 (orig 0.5535), valid r2 0.7328 (orig 0.7277); FT mitades r2 rows
0.7416 vs 0.7372 (FT plano), pero blend 0.6906 = sub23d (+0.0000/+0.0001) ≈ neutro → sub23e empaquetada, prioridad baja.
Siguiente: r3 desde cero 16 ép. (queue_p2_r3_16.sh → sub23f), r3 8 ép. tuvo best ep 7.
- LB sub23d 0.66418 (47m47s) = nuevo mejor, +0.00054 vs sub23b (OOF +0.0009). Reentrenar especialistas desde cero con etiquetas suaves funciona en LB.

**Fase 2 r3 16 ép. (mingru_ema_cls3_st16):** sobreajusta desde ép. 9 (es_pm 0.3870 ép. 7 → 0.3642 ép. 13); FT mitades r3 rows 0.4401
vs 0.4410 (8 ép.), blend 0.6905 vs 0.6906 ✗. Cola cortada antes del FT completo (sub23f no se empaqueta). Para r3: 8 ép.; más
capacidad/épocas necesita regularización. Siguiente: SSM desde cero (queue_p2_ssm.sh → sub23g).

**Fase 2 SSM (ssm16_c3_st, desde cero la50, 8 ép.):** es_pm 0.5686 (orig 0.5597), valid 0.6878 (orig 0.6801); FT suave mitades
0.6893 vs FT plano 0.6834; blend 0.6916 vs 0.6906 (sub23d) **+0.0010 (h0 +0.0010 h1 +0.0009)** ✓ → sub23g_route4_ssmscratch_stack
(~46 min estimado, 104 µs/row).

**Cola hacia sub24 (2026-10-29 02:45):** queue_p2_r01.sh (r1, r0 desde cero → gate DECIDE en sub23g-like → sub23h) y luego
queue_sub24.sh: gate proj_stack2.py (stacks r0/r1 reajustados sobre el blend nuevo vs stack sub21 aplicado al blend nuevo,
mitades FT, a .25/.35/.45; pasa si ambas mitades suben y suma ≥ +0.0004) → train_stack.py --spec/--tcn/--out stack_v2 →
package_stack.py --stack-dir stack_v2 → sub24_route4_stack2. train_stack/package_stack ahora parametrizados (defaults = sub21).
- LB sub23g 0.66453 (49m36s) = nuevo mejor, +0.00035 vs sub23d (OOF +0.0010; traslado ~35 %).

**Fase 2 r0/r1 desde cero (cls0_st/cls1_st, la50):** es_pm r0 0.7028 (orig 0.6978), r1 0.6436 (orig 0.6354); valid r0 0.8008 (0.7964),
r1 0.7744 (0.7695). FT mitades filas r0 0.8038 vs 0.7999, r1 0.7777 vs 0.7738, pero blend plano sub23g-like: r0 +0.0007/−0.0001 ✗,
r1 +0.0001/+0.0003 ✗ (umbral). **Stack r0/r1 reajustado** sobre blend nuevo (specs viejos): 0.6916 = stack viejo ✗ → sub23h/sub24 no
construidos. En curso (queue_p3_mix.sh): r0/r1 nuevos bajo el stack (proj_stack2) + r3 desde cero con etiquetas mezcla .5 la50 + .5 la64c.

**r3 etiquetas mezcla .5 la50 + .5 la64c:** FT mitades r3 rows 0.4407 vs 0.4410, blend 0.6916 = base ✗. la64c descartado también en mezcla.
**sub24_route4_r01scratch_stack2** (04:42): sub23g + r0/r1 desde cero (FT suave completo) + stacks r0/r1 reajustados (stack_v2, a .35,
specs st + TCN st16). Gate apilado 0.6923 vs 0.6916 (+0.0010/+0.0004). Latencia 103.5 µs/row (0.89× base), cold 34 s, paridad 1.2e-7.
Siguiente (queue_p3_a.sh): receta en r3 — soft-a .6 / 1.0 / conf 1.

## 25. Paso atrás 2026-10-29: LB #82 (0.66453); top-10 0.67122 (+0.0067), top-4 ~0.674
- jy447 (log 10-25) también en 0.66454 con la misma familia (FT valid + maestro look-ahead + especialistas por unidad) → esa ruta es el
  techo 'conocido'; el top tiene otra palanca. Notas suyas no probadas aquí: FT lr 1e-4 (+0.0016–0.0020 en un fold; en curso,
  queue_p3_lr.sh), pesos fp16 (54→42 min servidor), GRU clásico 2×256 como miembro principal (minGRU −0.007 vs GRU en su medición),
  6 maestros destilados (+0.0024), 2.ª semilla de miembro (+0.0004–0.0006).
- package_routed.py sólo admite especialistas minGRU (GRU requeriría runtime).

- **Auditoría de latencia (B):** pesos ya en int16 (= bytes de fp16), así que la palanca fp16 de jy447 ya estaba tomada. Perfil por miembro
  (model_r3, local, cargado): base sin miembros ~9 µs, minGRU ~33, TCN ~35, **SSM ~68 de los cuales ~31 son exp_neg_f** (4096 por fila,
  tanh racional + 2 divisiones). Nuevo fastmath.exp_neg_fast (tabla 2^-n + polinomio cephes exp2f, sin división; err abs 8e-8):
  scan 30.6 → 6.3 µs/fila (micro); SSMRunner A/B 140–150 → 87–103 µs bajo carga. Paridad numba–torch SSM ≤1.8e-5. Empaquetado
  sub24f_route4_fastexp_stack (= pesos de sub24, runtime nuevo). A/B del paquete completo pendiente con CPU tranquila (carga 8).

- **P3 recetas r3 (2026-10-29):** soft-a .6 → r3 0.4386 / blend 0.6913 ✗; a 1.0 → 0.4406 / 0.6915 ✗ (base a .8: 0.4410 / 0.6916).
  cf (pesos por confianza) cortado en ep 2 para liberar GPU al GRU. FT lr 1e-4 (2 ep y 5 ep) sobre spec r3 + TCN st16: neutral
  (0.6915–0.6916 vs 0.6916) — la palanca de jy447 no se reproduce aquí (su FT ya usaba lr distinto). Eje receta/FT agotado.
- **sub24 LB 0.66487** (48m47s) = mejor; +0.00034 vs sub23g (OOF +0.0007, carry-over ~50 %).
- **sub24f (fast exp) LB 0.66487 en 42m22s** — mismo score, −6.4 min de servidor. Margen a 52 min: ~9.5 min ≈ +20 µs/fila local (0.45 min por µs).

## 26. Calibración online por secuencia con pseudo-labels retrasados del maestro (2026-10-29)
- pipeline/regime_gates/proj_tta.py: a la fila t el maestro la50 de la fila t−50 ya es causal (valid corr 0.80). EMA (500/2000/8000)
  de (maestro − blend) como corrección de nivel, o pendiente maestro~blend como escala, sobre el blend tipo sub23g (0.6916):
  todas peores, monótono en b (b .25 span 8000: 0.6912/0.6913; b 1 span 500: 0.6709/0.6598). El maestro no lleva información
  del desplazamiento por secuencia que mide el oráculo (+0.02): ✗ cerrado.
- GRU 2×256 (soft la50, EMA): es_pm 0.5440/0.5476/0.5487/0.5511/0.5547 vs SSM 0.5501/0.5563/0.5597/0.5641/0.5656 → ✗; 2×128 cancelado.

## 27. Limpieza y punto de partida = sub24f (2026-10-29)
- Repo reducido a lo que reproduce sub24f (`docs/REPRODUCE.md`) + gates vivos; borradas ~99 colas/gates/scripts viejos
  (LightGBM, ensemble, oof, adv, soups, screenings), plantillas lgbm/ensemble y HANDOFF.md. Lo de arriba sigue siendo el registro.
- Disco ~80 → ~24 GB: se quedan los miembros de sub24f y sus corridas base (para etiquetas del ensamble), los maestros y sus
  etiquetas, las mitades de valid que usan los gates, pmask, detector, stack_v2, filas del stack (artifacts/stack_rows,
  antes en el scratchpad) y 5 corridas de arquitecturas distintas (ssm144_ema_c3, gru/mingru_attn32_c3, gru/mingru_corr3_pm05)
  como posibles maestros diversos. Paquetes: sub14b (default), sub24, sub24f.
- Ideas nuevas en curso: (1) auto-destilación del ensamble (`scripts/ens_label.py`, `queue_ens.sh`, gate `proj_ens.py`);
  (2) estructura del target (`proj_target.py`): en r3 t0 + t1 = −1 tick en 76 % de las filas (≈ −spread; los dos targets son
  los dos lados del libro), el blend predice mal la suma (WP 0.12 en r3) y bien la diferencia (0.51); prueba de reexpresar la
  respuesta como suma/diferencia (`proj_sumdiff.py`).
- Suma/diferencia post-hoc (proj_sumdiff, factor l_s por régimen, cross-fit): h1 0.6884 → 0.6880, h0 0.6948 → 0.6946 ✗ — confirma §A.2:
  el blend ya es casi óptimo como función de (p0, p1); la suma depende del spread futuro.
- Escala por régimen y target (proj_scale, cross-fit sobre el blend tipo sub24): −0.00021 / −0.00008 ✗ (factores 0.93–1.03,
  el blend ya está bien escalado). Variante con offsets cortada para liberar la GPU a la auto-destilación.

## §28 Distribution distillation (idea b, 2026-10-29)
Look-ahead teacher with a tick-class head on r3 (`teacher_la50_cls3`, la50 recipe + `--regimes 3 --class-head`);
`teacher_label.py` now also writes `<tag>_dist_train.npy` (class probabilities, r3 sequences only). Student: sub23d r3
recipe + `--kd-dist teacher_la50_cls3 --kd-w 0.5` (class CE target = .5 one-hot + .5 teacher probabilities; point soft
labels stay la50). Gate `proj_kd.py` vs the current r3 specialist (seed-2 run printed as noise reference).
Queue `queue_kd.sh` (after queue_seed_r3).
- §27 idea 1 result (2026-10-29 23:00): TCN 8 ep on ens24_la50 labels: alone better (valid 0.6868 vs st8 0.6856 / st16 0.6862;
  after soft FT 0.6883 vs st8 0.6873 / st16 0.6881) but in the blend 0.6915 vs st8 0.6914 / st16 0.6916 → DECIDE_ENS 0 ✗.
  Mechanism: ensemble labels pull the member toward the ensemble it joins → its own gain is diversity lost. Self-distillation
  of a member that stays in the blend does not pay here; r3 stage skipped. (Would only help a *single* model replacing the blend.)
- Second-seed r3 specialist (sub23d recipe, seed 2): alone 0.4404 vs 0.4410 (r3 rows), 2-seed mean 0.4422, blend
  +0.00008 / +0.00004 → DECIDE_SEED 0 ✗. Seed bagging of the specialist is worth ~+0.0001 here (cf. jy447's +0.0003).
- §28 result (2026-10-30 01:40): teacher_la50_cls3 (r3, 8 ep) es_pm 0.7576 / valid r3 0.6690. KD student (kd-w .5): valid 0.4386
  (base 0.4382, seed-2 0.4393); after soft FT r3 rows 0.4410 = base 0.4410; blend −0.00002 / −0.00001 → DECIDE_KD 0 ✗.
  Mechanism: the teacher's distribution adds nothing beyond its point label at a .8 — the student's tick probabilities are
  limited by causal information, not by the label's form. (Bug fixed on the way: soft point labels must read Y[..., 2:4].)

## §29 Slow branch on the soft-label r3 specialist (2026-10-30 02:10, user request)
`--slow-k` now also for the minGRU (GRU over 100-row block means, its state added to rows of the next block; causality
and chunking verified). Run `mingru_ema_cls3_slow100` (sub23d recipe + `--slow-k 100`), queue `queue_slow.sh`, gate proj_kd.py
(replaces the r3 specialist in the blend). Prior: plain-GRU screening k=20/100 tied (0.6750 vs 0.6748).
- §29 result (2026-10-30 09:30; laptop slept ~6 h): slow100 r3 valid 0.4355 (base 0.4382); after soft FT r3 rows 0.4392 vs 0.4410;
  blend −0.00032 / −0.00015 → ✗. Second negative for the slow branch (plain GRU tie, r3 specialist worse): the 100-row block
  memory adds nothing the EMA z-features and the minGRU state do not already carry. Closed.

## §30 Bid/ask imbalance probe for r3 "move vs no-move" (2026-10-30, user request; scratchpad a7_imb.py, log logs/a7_imb.log)
Same rows/labels/halves as §12 (1.72 M scored r3 valid rows, weights |y|). Added 62 features per instrument pair: per-index
bid−ask volume diffs (quantile space; indices are not levels, so no true L1), global imbalance on exp(v) and its EMAs 20/100/500,
order-flow-imbalance proxy (bid minus ask volume changes) EMAs 5/20/100, volume activity EMAs, per-index diff EMAs; for i0 and i1.
| features (LightGBM cross-fit on valid halves) | AUC | h0 / h1 |
|---|---|---|
| specialist P(no-move) alone | 0.6918 | — |
| spec + imbalance | 0.6576 | .6565 / .6590 |
| history + row + spec (§12 repro) | 0.6651 | .6644 / .6664 |
| history + row + spec + imbalance | 0.6616 | .6602 / .6635 |
- Univariate: imbalance/OFI ≈ 0.47–0.51 (no signal); only volume activity separates (i0 0.639, i1 0.60 inverted), already in history feats.
- Any tree model on top of the specialist loses to the specialist itself: what it learns on one valid half does not transfer
  (shift between sequences). Imbalance adds nothing the NN does not already read from the raw volumes. ✗ Closed.

## §31 Community check (2026-10-30)
- jy447.github.io "Alfa Connectome notes" (last update 10-25, 0.664539, rank 66): nothing new since §25. Relevant for the next bet:
  "delta memory" within noise; online correction with look-ahead teachers gains exactly 0; per-seq level oracle +0.029 but nothing
  visible in advance predicts the error runs (corr ≤ 0.03). → weak prior for TTT/DeltaNet per-sequence adaptation.
- Not yet tried here from their log: Hire-MLP input front (+0.0005 in their package, +18–30 µs/row), FT unscored weight 0.1 + 6 ep
  (+0.00054 public, ours mix4 = 0.2, 3 ep). Both small.
- Other repos (XArhaanshX, OogwaySeinse) publish no method before 11-15. No top-10 write-up public → the gap to 0.671 is unpublished.

## §32 jy447 FT recipe: unscored weight 0.1 (mix9), 6 ep, lr 2e-4 (2026-10-30; queue_ft01.sh, gate proj_ft01.py)
r3 spec + TCN st16 soft-FT halves vs current mix4 / 3 ep / 3e-4 (base_g 0.6916, h0 .6948 h1 .6884):
| epoch read | r3 rows | TCN alone | blend both (Δ h0 / h1) |
|---|---|---|---|
| 3 | 0.4403 vs .4410 | 0.6879 vs .6881 | −0.00025 / +0.00001 |
| 4 | 0.4404 | 0.6882 | −0.00002 / +0.00005 |
| 5 | 0.4409 | 0.6882 | **+0.00013 / +0.00008** |
| 6 | 0.4406 | 0.6882 | +0.00010 / +0.00007 |
Consistent but tiny (+0.0002 sum at ep 5, below the +0.0004 rule; jy447: gate gains < +0.0004 never showed on the board). ✗ as a
standalone sub. The FT axis (lr §25, unscored weight, epochs) is saturated in our pipeline.

## §33 TTT-style per-sequence memory: DeltaNet delta-rule layer on the r3 specialist (2026-10-31; queue_delta.sh, gate proj_kd.py)
`--delta 4` (train_gru.DeltaLayer): residual layer after the minGRU blocks, 4 heads × 32, L2-normalised k/q, sigmoid β,
S_t = S_{t−1} + β(v − S k)kᵀ, o = S q, zero-init output; chunkwise-parallel form (C 100), parity with the naive recurrence 1e-6,
chunked = full, causal. Sub23d recipe from scratch, 8 ep (11–27 min/ep vs 3.5 for the base: the chunk loop is GPU-bound).
| | es_pm | valid r3 (pre-FT) | FT h0 / h1 (r3) | r3 rows (cross-fit) | blend Δ h0 / h1 |
|---|---|---|---|---|---|
| base mingru_ema_cls3_st | 0.3885 | 0.4382 | .4558 / .4254 | 0.4410 | — |
| delta4 | 0.3856 | 0.4389 | .4571 / .4244 | 0.4411 | +0.00012 / −0.00010 |
| seed-2 reference | | | | 0.4404 | |
Exact tie, sign flips between halves = seed noise. Same as jy447 ("delta memory" within noise). Mechanism ruled out: a learned
unsupervised in-context memory does not recover the per-sequence level (the +0.03 oracle); the minGRU state + EMA z-feats already
hold whatever the sequence's past says about it. ✗ Closed. Code kept (`--delta`), no runtime port.

## §34 Final-candidate robustness: is the r0/r1 row stack specialised to the valid mask? (2026-10-31; scratchpad a8_robust.py)
Stack as in sub24f (LightGBM on row + diff + route blend, fitted on SCORED valid rows, a .35), cross-fit on the FT halves.
Rows of r0/r1 sequences only. "unscored" = 5 % random sample of required non-scored rows; "alt" = its top 10 % by estimated P(scored).
| rows | n | no stack | stack | Δ |
|---|---|---|---|---|
| scored (gate) | 1.52 M | 0.7967 | 0.7977 | **+0.0011** |
| unscored | 663 k | 0.5209 | 0.5179 | **−0.0031** (r0 −0.0025, r1 −0.0038) |
| alt (look like scored) | 66 k | 0.6688 | 0.6669 | **−0.0019** (r0 −0.0013, r1 −0.0030) |
The stack learns the mask-selection process, not the target: it loses on every required-row set outside the scored rows, even
on the ones that look most like them. Public LB still credited it (+0.0003) because the public test mask is made like valid's.
→ Finals: aggressive = sub24f_route4_fastexp_stack (public 0.66487); robust = sub24f_route4_fastexp (same members, no stack).
- Follow-up (scratchpad a9_robust_ssm.py): the existing no-stack package uses spec .5 + SSM .25 + TCN .25 on r0/r1. Same row sets:
  vs .6/.4: scored +0.0004 (0.7971), unscored +0.0016 (0.5226), alt +0.0014 (0.6702) → vs the stack: scored −0.0006, unscored
  +0.0047, alt +0.0033. Cold start from fresh unzip: no-stack 28.3 s, stack 29.9 s; local A/B latency equal within noise
  (2 rounds, 4 workers: 138/153 vs 160/129 µs). Robust final = sub24f_route4_fastexp (not yet on the public LB).

## §35 Route blend weights re-fit (2026-10-31; scratchpad a10_weights.py)
Weights unchanged since sub18 (spec .5 / SSM .25 / TCN .25) though every member was retrained. Per-route grid (step .05) fitted on
one half, scored on the other (no-stack blend, base h0 0.69561 h1 0.68854):
fit h0 → SSM-heavy (r3 .45/.50/.05), eval h1 −0.00021; fit h1 → spec-heavy (r0 .70/.20/.10), eval h0 −0.00008. Optima flip
between halves and straddle the current weights → flat surface, the fitted part is noise. ✗ Keep .5/.25/.25.

**§34 LB check (2026-10-31 12:41):** robust final sub24f_route4_fastexp (no stack, spec .5/SSM .25/TCN .25 on r0/r1) public
**0.66456** vs stack 0.66487 → stack worth +0.0003 on the public mask (≈30 % of its +0.0011 scored-row OOF gain). The public LB
uses the same mask process as valid, so this does not test the unscored/alt risk; keep both finals as planned.

## §36 Community check #2 (2026-11-06/07)
LB 11-06: #1 0.67596, #10 0.67209, #50 0.66754; we #97 (0.66487). Searched GitHub (repos pushed after 10-20) and the web.
- jy447: no update since 10-25.
- SHARIQ-09/wunder_rnn_challenge (public repo, docs/wunder_challenge_report.md): best public 0.65539. Pre-LN GRU, model soup,
  SWA, C kernel. Their valid→platform drop is 0.024–0.030 on every model. Raw 112 columns ≥ their 32 engineered features.
  Nothing above our level.
- Marcocxm171 (TCN search vs GRU), MarkBrezina (research log, 10-21), Jenish1706 (scaffold only), XArhaanshX (private until 11-15):
  no scores above ours, no new lever.
- rules.md forbids sharing detailed strategies before the end, so the top-10 methods will not be public before 11-15. ✗

## §37 Permutation importance of the train-only bases (2026-11-07; scratchpad a11_perm.py, logs a11_r3/a11_tcn)
Column j (raw + its EMA-z) replaced by the same column of another valid sequence of the same regime; drop in scored WP,
240 valid sequences each. r3 spec mingru_ema_cls3_st (r3 base 0.4308); TCN tcn64_ema_c3_st16 (r0/r1/r2/r3/all .795/.773/.738/.430/.674).
| group permuted | spec r3 | TCN r0 | TCN r1 | TCN r2 | TCN r3 |
|---|---|---|---|---|---|
| all i0 | −0.430 | −0.386 | −0.481 | −0.213 | −0.396 |
| i0 volumes | **−0.280** | −0.024 | −0.071 | −0.088 | −0.282 |
| i0 prices | −0.035 | **−0.104** | −0.044 | −0.013 | −0.017 |
| i0 trades p / v | −0.013 / −0.026 | −0.006 / −0.003 | | | −0.022 / −0.021 |
| a0–a7 | −0.033 | −0.028 | −0.023 | −0.018 | −0.044 |
| all i1 | **−0.002** | −0.044 | −0.016 | **−0.100** | −0.012 |
- Top single columns (both models): i0_v4, i0_v21, i0_v15, i0_v0, i0_v19, i0_v20, a6 (same as the §9 A.3 row LightGBM).
  Sum of single drops (spec −0.19) ≪ group drop (−0.43): the volume levels are used jointly (profile), not one by one.
- Fine tick (r0) leans on i0 prices; coarse tick (r3) almost only on i0 volumes; r2 uses i1 heavily (TCN −0.10); the r3 spec ignores i1.
- No column whose permutation helps beyond noise (max +0.001). Importance shows what the models use, not what is missing;
  the volume-profile direction (row encoder §18, imbalance §30) is already closed. No new lever. ✗ (informative only)

## §38 Valid-adapted look-ahead teacher (2026-11-09; queue_vt.sh, scripts/vteacher_labels.py, gate proj_vt.py) ✗
teacher_la50_gru fine-tuned (mix4, 3 ep, lr 3e-4) on one valid quarter, scored on the partner quarter of the same FT half
(`finetune_valid.py --fit-groups/--eval-groups`). Scored WP before → after FT:
q0 0.7956 → 0.7949, q1 0.8024 → 0.8014, q2 0.8033 → 0.8025, q3 0.8085 → 0.8070 (every epoch flat or down).
Students gain +0.003–0.007 from the same FT; the teacher gains nothing: its edge comes from the future rows, which carry the
period information that FT gives to causal students. Labels would be ~unchanged → student gate stopped before it ran. ✗

## §39 Delayed reconstructed labels from the look-ahead teachers ✗ (LEAK in the first reading; 2026-11-09; scratchpad a12_delayed.py, a12b_sweep.py)
Per regime, cross-fit on the FT halves, linear stack on scored rows (weights |y|): y_k ~ p_k(t) + d(t−L) + EMA_100(d)(t−L) + 1,
d = teacher − p; p = sub23g-like OOF blend (base_g 0.6916, h0 .6948 h1 .6884).
| source | L | WP | h0 / h1 | r0 / r1 / r2 / r3 |
|---|---|---|---|---|
| la50 (fut means; reach 50) | 50 / 100 | 0.6908 / 0.6907 | −.0003/−.0009 | all ≤ 0 (= §24) |
| la64c (conv) | 64 | 0.7079 | +0.0156 / +0.0173 | +.0015 / +.0047 / +.0079 / +.0499 |
| la64c | 65 / 68 / 72 / 80 / 90 / 100 | .7071 / .7049 / .7023 / .6977 / .6937 / .6915 | smooth decay to 0 | |
| oracle y | 50 / 64 / 100 / 200 / 500 | .8489 / .8271 / .7865 / .7330 / .6963 | | r3 +.37 at L50 |
**Not causal.** la64c's reach is NOT 64 rows: its conv reads the inputs of rows s..s+63 and every input row carries the fut
means up to 50 rows further, so q(s) depends on rows ≤ s+113. At L64 the feature sees rows up to t+49 → the +0.016 is leakage.
Causal from L ≥ 114, where the gain is already 0 (L100 +0.0001). la50 (reach exactly 50, causal at L50) gives ≤ 0.
Conclusion: a causal estimate of the recent realised target adds nothing beyond the blend; the oracle's value lies in
information that is only in the future. Lesson: compute a teacher's true receptive field (input features' own look-ahead
included) before reading a delayed-teacher feature; a smooth decay is no proof of causality.

## §40 Causal hindsight heads ✗ (2026-11-09; scratchpad a13_hind.py; model nn/mingru_ema_hind, stopped after ep 2, GPU thermal-throttled)
minGRU-EMA2000 trained on train with aux heads (w 4) for y(t−16/−32/−64/−128) from rows ≤ t; heads read at eval (train_gru AUX_EVAL).
Same stack as §39 with d_L = ŷ_L(t) − p(t−L): every lag ≤ 0 (L16 −.0001/−.0007, all lags −.0004/−.0010).
With the leaky la64c L64 feature added: +.038/+.040 (vs +.016 alone): the causal estimate of the recent target only pays when
paired with future rows, which confirms §39 is leakage. Delayed-label / label-reconstruction direction closed.

## §41 Unconventional-ideas round (2026-11-10; 5 days left; queue_lab.sh, gate proj_lab.py, scripts/smooth_labels.py)
Filter from §1–§40: every change that improves the base *before* FT (architecture, inputs, row encoder, move head, EMA spans,
regularisers) is absorbed by the valid FT (§17, §18, §20, §21); the only family that kept paying after FT is the **label**
(soft labels from privileged look-ahead teachers, §23–§24). Online / per-sequence adaptation is closed (§10, §19, §26, §33, §39,
§40: the per-sequence level is not in causal info). So the round is label-side, on the r3 specialist (48 % of metric weight,
~1 h per variant), gate = replace or **pair** with the current r3 spec (a 2nd r3 spec costs ~+33 µs on r3 sequences only, fits the
~9.5 min headroom of sub24f). Seed-2 pair reference: +0.00008 / +0.00004.
Literature / community behind each variant:
- Teacher with less future (la25, la12): capacity gap (Mirzadeh et al., TAKD, AAAI 2020: a stronger teacher can make a worse
  student), Label Horizon Paradox (arXiv 2602.03395, ICML 2026: the best training label is often an intermediate horizon),
  jy447 notes (each student loses as the teacher sees more future; they moved to 25-row teachers). Our la64c ✗ fits this.
- Teacher ensemble labels (m25 = la50 + la25): jy447 used an average of several teachers.
- Teacher-free temporal label smoothing (s16, sm): centred mean of y over ±16 rows (WP vs y 0.98, corr with la50 0.64) — a
  different denoiser of the same target, no teacher, cost 20 s.
- Privileged-noise weights (cf1, `--soft-conf 1`): LUPI / heteroscedastic privileged information (Lambert et al. 2018):
  down-weight rows the look-ahead teacher calls noise. (Cut at ep 2 on 10-29, never read.)
Considered and not run (mechanism already ruled out here): label-free TTA (TAFAS/DynaTTA/COSA, needs delayed labels → §26/§39),
SAM / flat minima (base-side, absorbed by FT), new recurrent archs (xLSTM, RWKV-7, Mamba-2; members corr 0.98, §3),
LOB features OFI/HLOB (§30), foundation models (latency), feature-distillation à la TimeKD (§28: the student is limited by causal
information, not by the label's form).
- **s16 ✗ (2026-11-10 00:50, stopped at ep 2):** es_pm 0.3402 / 0.3395 / 0.3524 vs the hard-label run 0.3475 / 0.3592 / 0.3693 and
  la50 soft 0.3583 / 0.3713 / 0.3781 → below hard labels at every epoch (−0.017 at ep 2). Mechanism ruled out: a symmetric
  temporal average of y is not a denoiser here — it moves label mass to rows *before* the move, which a causal student cannot
  explain; the look-ahead teacher helps because it keeps only the part of y that the book's near future explains. The .5 mix
  with la50 (sm) dropped unrun. Queue reordered by prior: la25 teacher → la25 → cf1 → m25 → la12 (`scripts/mix_labels.py`).
- Operational: GPU thermally throttled after ~4 min (76 °C, 630 MHz, 14 W): r3 epoch 180 s cold → 760 s throttled.

## §42 Advisory 2026-11-10: two mechanisms closed without GPU, target definition from jy447
- **Per-regime re-calibration under period shift ✗** (`pipeline/regime_gates/a14_scale.py`): idea = fit the 4 route scales to the
  test period with LB probes. Proxy: train-only models scored on valid (same kind of shift as valid → test), per-regime and
  per-target scale / affine fitted IN-SAMPLE on valid: tcn64_ema_c3_st16 0.6874 → +0.0001 (scale) / +0.0004 (affine);
  ssm16_c3_st 0.6890 → +0.0001 / +0.0004; fitted scales 0.92–1.02. Even with the labels of the target period there is nothing
  to recover → models stay calibrated across periods; LB scale probes not worth a submission.
- **Target definition (jy447 notes, read 11-10):** t0 = F − ask, t1 = bid − F, one reference price F on the tick grid (on a quote
  when the spread is 1 tick, on the midpoint when 2); F moves only when i0's quotes move and settles within the next ~8–16 quote
  changes (dozens to hundreds of rows). So t0 + t1 = −spread (our §27 finding) and y is constant between quote changes.
  Consequence for §26/§39/§40: y(t−L) shares the same future F as y(t) while fewer than ~8–16 quote changes have passed, so the
  delayed-label oracle is future information by construction, not a hidden level → closed for good. They report the spread
  "worth +0.06 but unreadable at the scored rows" (price columns = clipped linear mix of levels); de-anonymised inputs pending there.
- Only untested direction with top-10-sized upside: de-anonymising the book (un-mixing the price / volume columns into true L1
  quotes and sizes). Low probability in 5 days; CPU-only analysis.

## §43 De-anonymisation probe (2026-11-10; pipeline/regime_gates/a15_level.py; time-boxed, CPU only) ✗
- **Price columns decoded in r3.** The 22 i0 price columns move together and take only 17 joint values over 40 coarse-tick
  sequences (8–15 per sequence); 6 values cover 99.9 % of rows. Each value fixes the book change of that row: unchanged (74 % of
  rows, 15 % scored), ask up → spread 2, bid up → spread 1, bid down → spread 2, ask down → spread 1, unchanged at spread 2,
  full-tick moves (rare). d(t0 − t1) is the modal value of its state on 99.5 % of rows, so dmid ∈ {0, ±½, ±1} tick is read
  exactly and L = cumsum(dmid) is the true mid path of the sequence (features = book relative to the previous row).
  y changes only on rows where the price state changes (P = 0 otherwise). Long stretches are a periodic one-tick quote flicker
  around a fixed F; quote-change rows are almost never scored (0.5–5 %).
- **The decoded level adds nothing** (r3 scored rows, sub23g-like blend 0.4433, cross-fit FT halves): L − EMA_n(L) (n 3…3000),
  last move direction, time since last move, activity: weighted corr with y ≤ 0.054, with the blend residual ≤ 0.07;
  linear stack p + decoded 0.4403 vs p only 0.4412; LightGBM 0.3976 vs 0.4361 (trees on top of the blend lose, as in §12/§30).
  Mechanism ruled out: the NN already integrates the tick path (it is a linear code in the inputs); on scored rows the
  price position relative to its past says almost nothing about the future reference price (= jy447's "unreadable at the
  scored rows").
- **Volume columns:** all 11 bid columns change on every volume event; event directions cluster only partly on 11 lines
  (cosine to nearest of 11 centres: median 0.974, 65 % > 0.95) → dense linear mix of level sizes plus a per-column warp.
  A linear un-mix is already available to the input layer, and the non-linear row encoder (§18) was absorbed by the FT, so
  recovering true L1 sizes needs the warps (days of work) for a mechanism with a null prior. Not pursued. De-anonymisation closed.

## §44 SSM back in r0 / r1 under the row stack (2026-11-10; pipeline/regime_gates/a16_ssm01.py, queue_sub25.sh) ✓ gate
§22 follow-up (+0.0002/+0.0002 for r0, dropped for latency before fastexp). Current members, stacks refit per variant, a .35,
cross-fit FT halves (4747 scored rows of mis-detected sequences use the general members):
| r0 / r1 routes | all | h0 | h1 | Δ vs sub24f |
|---|---|---|---|---|
| no stack, no SSM | 0.69181 | 0.69530 | 0.68842 | |
| no stack, SSM (robust final) | 0.69201 | 0.69558 | 0.68855 | |
| stack, no SSM (= sub24f) | 0.69227 | 0.69576 | 0.68890 | — |
| stack, SSM in r0 | 0.69246 | 0.69604 | 0.68901 | +0.00027 / +0.00011 |
| stack, SSM in r1 | 0.69234 | 0.69585 | 0.68894 | +0.00009 / +0.00004 |
| stack, SSM in r0 + r1 | 0.69252 | 0.69612 | 0.68905 | **+0.00036 / +0.00015** ✓ |
SSM and stack are additive (the SSM adds +0.0002 with or without the stack). Cost: SSM (~40 µs with fastexp) on the r0 + r1
sequences (40 % of valid) ≈ +16 µs/row ≈ +7 min → ~50 min estimated; uses the headroom, so the r3 pair (+0.0001, +5 min) does
not fit on top. Package: sub25_route4_ssm01_stack = sub24f_route4_fastexp + stack_v3 (`train_stack.py --with-ssm --ssm`,
`package_stack.py --keep-ssm 0,1`).
- **sub25_route4_ssm01_stack packaged (02:16):** stacks numba = LightGBM (0.0), routed vs offline 1.2e-7, r2 route 0.0; zip 14.9 MB.
  Local latency and cold start were measured with the GPU queue and the labelling job running, so only ratios mean anything:
  cold start from a fresh unzip 176.7 s vs 170.0 s for sub24f_route4_fastexp_stack measured back to back (sub24f passes the
  prerun on the server); latency reps 1.07× / 0.83× the no-stack base → noise. Server estimate from the SSM's cost
  (sub18 − sub19 = 26 min on all sequences, fastexp −41 %, r0 + r1 = 40 % of sequences): 42.4 + ~6 ≈ 48–49 min.
- teacher_la25_gru (horizons 2,5,12,25; 4 ep): es_pm 0.7206, valid 0.7599 (la50: 0.7979 / 0.8025).

## §45 Plan for the last days (2026-11-10 03:00)
- GPU budget: one full propagation of a label recipe ≈ 1 day (r3 1 h, SSM 4.5 h, r0/r1 2 h, TCN 4–7 h) → at most two before 11-15.
  Screening variants are cut as soon as one passes the r3 gate; losers are stopped at ep 3–4 when es_pm trails the la50 run
  (0.3583 / 0.3713 / 0.3781 / 0.3827 / 0.3859 / 0.3872 / 0.3882 / 0.3885) by more than 0.003.
- `pipeline/queues/queue_prop.sh <X> <train labels> <valid labels> "r3pkg ssm r01 tcn"`: member from scratch → soft FT halves →
  incremental gate `proj_prop.py --incremental` (lean, scored rows only; reproduces proj_lab / a16 numbers) → full FT → own
  package (sub26a…d, stack_v3, SSM kept in r0/r1), copied to `submissions/`.
- Throughput: GPU is capped by the laptop profile (enforced limit 20 W, sw power cap + sw thermal slowdown at 74 °C, Windows
  plan Balanced): r3 epoch 860 s vs 180 s cold. Dell thermal mode / Windows power mode are the largest lever left on speed.

## §46 Brainstorm round 2 and standing loop (2026-11-10 03:15; user: keep launching new cases until the score moves)
Early read: la25 student es_pm 0.3633 / 0.3760 / 0.3819 vs la50 0.3583 / 0.3713 / 0.3781 (+0.004–0.005 at equal epochs).
Throughput: r3 epoch 812 s / 367 s / 182 s / ~380 s — it tracks MY CPU side jobs (gates, packaging, cold-start timing), not
only the profile (enforced limit flips 40 W ↔ 20 W with heat). Rule: no CPU work while a GPU run is going; gates run between runs.
Queue of mechanisms, by expected gain per GPU hour (each with a gate; winners become their own sub):
1. **Pretrain → specialise** (`queue_pt.sh`, `--init-from` now tolerant of a fresh head): specialists start from the general
   minGRU trunk (all 10.6k sequences) instead of scratch on 1.3k (r0, r1) / 4.6k (r3) sequences. r0 from scratch is still
   improving at ep 7 and r3 overfits at 16 ep → both look data-limited. r0 + r1 = 44 % of the metric weight. Never tried
   (the only warm start in the log is train+valid, §3).
2. Label horizon: la25 (running) → m25 → la12 → cf1; best one is propagated with `queue_prop.sh`.
3. If 1 works: pretraining source with soft labels (general minGRU on the winning labels), then all four specialists from it.
4. Teacher ensemble across seeds (2nd la25 / la50 teacher, averaged labels): jy447 +0.0024 on their gate with 6 teachers.
5. General regime-conditioned member: one model on all regimes with tick-normalised targets (y / unit) and a common class grid
   — the limit case of 1; only if 1 passes.
Not run (mechanism closed): mirror augmentation (jy447 negative, no 1-1 bid/ask map §14), cross-sequence lookup of valid
neighbours (edge rows only, against the rules' spirit), capacity (§8), online adaptation (§42).

## §47 la25 labels pass on r3 → sub26a (2026-11-10 04:00)
- Teacher la25 (horizons 2,5,12,25; es_pm 0.7206 vs la50 ~0.80) → r3 specialist from scratch: es_pm 0.3907 vs 0.3885 (la50);
  lead at equal epochs shrinks (+0.0050 ep0 → +0.0019 ep3 → +0.0022 ep7) but survives soft FT (FT labels = la25 too):
  r3 rows 0.4428 vs 0.4410 (corr 0.982), blend replace +0.00016/+0.00032 ✓, pair +0.00017/+0.00024 (no extra over replace).
  Weaker teacher with less future = better labels (label-horizon / capacity-gap argument, as jy447 reported).
- Screening cut (cf1, m25, la12 not run) to spend the GPU on propagation: `queue_loop1.sh` = r3 package → pretrain→specialise
  gates r0, r1 (`queue_pt.sh`) → la25 on SSM, r0/r1, TCN (`queue_prop.sh`).
- `sub26a_route4_r3la25_stack` = sub25 with the la25 r3 specialist. Parity: trees 0.0, routed 1.2e-7; 14.9 MB; same
  architecture as sub25 → same latency (ratio to the no-stack base 1.09, as sub25).

## §48 Pretrain → specialise ✗ (2026-11-10 04:26)
Trunk from the general `mingru_ema_c3` (all regimes, hard labels), fresh tick-class head, usual recipe at lr 1e-3 (`queue_pt.sh`).
- r0: es_pm 0.6950 at ep 0 (scratch 0.6879) but 0.7008 at ep 7 (scratch 0.7028); after FT member 0.8009 vs 0.8038, blend −0.00053/−0.00029.
- r1: member 0.7745 vs 0.7777, blend −0.00015/−0.00032.
Head start, lower ceiling: the general trunk is a worse basin for the specialist than its own init — the specialists are NOT
limited by their number of sequences; what the other regimes teach does not transfer. Closes the "general → specialise" family
(incl. the regime-conditioned single model, idea 5 of §46). GPU cold: r0 epoch 56 s (vs 220 s throttled).

## §49 sub25 public 0.66506 (2026-11-10)
`sub25_route4_ssm01_stack` (id 1pnaclkr): 0.66506 vs sub24f 0.66487 → +0.00019 (OOF +0.00036/+0.00015, carry-over ~37 %).
Server time not reported yet. New best; next in line: sub26a (= sub25 + la25 r3).

## §50 sub26a public 0.66502 — la25 r3 does not carry over (2026-11-10)
- sub25 0.66506, 51m02s, 1.3 GB (id 1pnaclkr). sub26a (= sub25 + la25 r3) 0.66502, 51m26s (id 278lzrsh): −0.00004 for an OOF
  gate of +0.00016/+0.00032. First gated label-side change with zero carry-over (previous ones 35–60 %).
- Latency: my estimate for sub25 was 48–49 min, real 51m02s (sub24f 42m22s): SSM in r0/r1 costs ~8.7 min on the server, not ~6.
  Budget (≤52 min) is used up: nothing that adds runtime fits on top of sub25; only swaps.
- Reading: the la25 gain lives in the FT-halves protocol (FT labels also la25, same valid period) and is small enough to be
  period noise on public. A +0.0005 r3-only gate is at the resolution limit of the public LB (~±0.0002).

## §51 Deep research round + long-reach teacher probe (2026-11-10 11:00; queue_reach.sh, pipeline/regime_gates/a17_reach.py)
- Community (rules forbid sharing before 11-15): no public solution above our level. jy447 (0.6645, same plateau): labels
  +0.0125 of +0.0167, architecture 0, "gains under +0.0004 don't show on the board", "a model run backwards scored 0.99",
  "correcting each error run's level with true targets is worth +0.029". mingyyu (#186): distillation + C engine. dozken: GRU
  192×2, sequence-level Pearson loss. Previous Wunder challenges: winners unpublished; public repos are GRU/LSTM seed ensembles.
- Literature (DAIN/BiN adaptive normalisation, AdaRNN, label-delay continual learning NeurIPS 2024, TimeKD): nothing that our
  closed list does not already cover; label-delay paper: SSL/TTA on the unlabeled recent data does not beat the naive baseline.
- The only mechanism with a measured ceiling of the size we need is the delayed target: oracle y(t−L) +0.095 (L100), +0.041
  (L200), +0.005 (L500). Teachers so far: reach 50 → 0.80 → 0; reach 113 (la64c) → 0.90 → 0. Untested: reach 200 / 400
  (does the slow level get resolved by the future before its value decays?). Probe = GRU teacher with fut means up to 200 / 400
  rows, then the §39 stack at L = reach vs the oracle at the same L.
- SSM la25 run paused at ep 2 (last.pt; ~1 h/epoch throttled, public value of la25 ≈ 0 after sub26a) to free the GPU.

## §52 Long-reach teacher ✗; i1 path probe launched (2026-11-10 12:40)
- teacher_la200_gru (fut means to 200 rows): es_pm 0.9121, valid scored 0.8986 (la50 0.80, la64c 0.90). Delayed stack at L200:
  0.6908 vs base 0.6916 (h0 −0.0002, h1 −0.0009; all regimes ≤ 0) vs oracle y(t−200) 0.7328 (+0.036/+0.046, r3 +0.127).
  Confirms §42: a reconstructible past target no longer shares the future reference price → closed for good. la400 cancelled.
- SSM la25 paused at ep 2 (nn/ssm16_c3_la25/last.pt), not resumed (la25 did not carry over in sub26a).
- Next probe `pipeline/regime_gates/a18_i1path.py <regime>`: i1 price columns have the same discrete joint-state structure as
  i0 (r3: 21–52 states per sequence, 2 states cover ~98 %); the r3 specialist ignores i1 (§37). Counts of i1 states over
  trailing windows 20–2500 (= net drift of i1) stacked on the blend at scored rows, i0 as control. Log logs/a18_r3.log.

## §53 Second deep-research round: i1 path ✗, spread level ✗ (2026-11-10 14:30; a18_i1path.py, a19_sstate.py, a20_spread.py, a21_slast.py)
Three mechanisms with a measured ceiling of top-10 size were tested today; all three ceilings are future information.
- **i1 path in r3 ✗** (a18): i1's 22 price columns have the same discrete joint-state code as i0 (398 states, top-16 cover 99.6 %,
  two "unchanged" states 67 % / 28 %). Counts of each state over trailing 20 / 100 / 500 / 2500 rows (= net drift of i1) stacked
  on the blend, r3 scored rows, cross-fit halves: 0.4302–0.4373 / 0.4158–0.4172 vs p only 0.4571 / 0.4274 (i0 control equally
  worse). The r3 specialist ignores i1 (§37) because there is nothing there, not because it cannot integrate it.
- **Spread level in r3 ✗** (a19–a21): stack p × one-hot(true S) = 0.5771 / 0.5493 vs 0.4587 / 0.4276 (+0.12 in r3 ≈ +0.057
  global). But on scored rows whose two legs are unclipped, S is given EXACTLY by the last non-modal price state (accuracy
  1.0000; S = −1 on 98 % of them): the true spread is already in the inputs. The oracle's value sits in the rows where one leg
  is at the −3-tick floor ((1,−3), (2,−3), (3,−3), (4,−3) and mirrors): there S encodes how far F is from the quotes, i.e. the
  size of the coming move. S_hat from integration (dS purity 0.995 per row, drift → 70 % level accuracy) adds nothing
  (0.4577 / 0.4275). Same conclusion as §9 A.2 / §16, now with the mechanism.
- **Delayed target ✗** (§52). Community / literature: nothing above our level is public (§51).
- valid → public offset is a constant −0.024…−0.027 since the baseline (0.6171 → 0.5935), so the top teams (0.676 public ≈ 0.70
  OOF) are ahead by ~+0.011 at model level, not through a period trick.
- r3 scored-row weight is concentrated in pre-jump rows: ((0,−1), (−1,0)) = 71 % of rows with weight 0.66 on one leg, the
  multi-tick pairs = 27 % of rows with weights up to 2 on both legs.

## §54 r3 jump specialist ✗ (2026-11-10 21:10)
Point-loss row weights (|y| + floor)^p (`--jump-p`, train + soft FT), la50 labels, r3 recipe from scratch (`queue_jump.sh`), gate proj_lab.py
(base 0.6916 | h0 0.6948 h1 0.6884; current r3 on r3 rows 0.4410).
- p 1.5: r3 rows 0.4373; replace −0.00063/−0.00029; pair −0.00022/−0.00006.
- p 2: r3 rows 0.4390 (corr 0.974); replace −0.00006/−0.00036; pair +0.00012/−0.00004.
- p 3: r3 rows 0.4335; replace −0.00141/−0.00083; pair −0.00045/−0.00015.
Monotone: the more weight on big moves, the worse. The |y|-weighted metric is already what the loss optimises (weights |y|, corr term);
sharpening the weights only raises the variance of the fit. Same verdict as the move head (§ move-w). Closed: row re-weighting of r3 by move size.

## §55 Independent audit round (2026-11-13/14): stretch smoothing ✗, capacity re-test on r3 ✗
- **Averaging the blend inside constant-target stretches ✗** (scratchpad aud1.py; y is constant between quote changes, §42).
  sub24-like OOF blend 0.69201: causal EMA a .5 → 0.68920; EMA reset at the TRUE target changes a .5 → 0.69045; running mean
  since the true change → 0.63932. Even with oracle stretch boundaries smoothing loses: the newest row carries the information
  (scored rows sit just before the move), there is no row noise to average out.
- **Wider r3 specialist ✗** (`queue_wide.sh 192`, gate `a22_wide.py`): mingru_ema_cls3_w192 = sub23d recipe with --hidden 192
  (513k vs 244k params), nothing else changed. es_pm by epoch 0.3583 / 0.3710 / 0.3775 / 0.3820 / 0.3851 / **0.3873** / 0.3857 /
  0.3849 vs 2×128 0.3583 / 0.3713 / 0.3781 / 0.3827 / 0.3859 / 0.3872 / 0.3882 / 0.3885: same curve to ep 5, then overfits.
  valid r3 0.4370 (0.4382); after soft FT r3 rows 0.4402 vs 0.4405 (corr 0.988); replace with SSM kept −0.00000 / −0.00013;
  r3 route without SSM (wide .6 + TCN .4) −0.00044 / −0.00027 (current spec without SSM: −0.00045 / −0.00013).
  A clean capacity test (same data, labels, schedule): width behaves like a second seed. Mechanism ruled out: the r3
  specialist is not capacity-limited at 2×128; more width only overfits earlier. Epoch 8–10 min (GPU 75–80 °C, 23–32 W).
- Next: teacher seed average (§46 item 4, never run): `queue_tens.sh`.

## §56 Teacher seed average ✗ (2026-11-14 00:32; `queue_tens.sh`, gate proj_lab.py)
teacher_la50_gru_s2 (seed 2, same recipe): es_pm 0.7985, valid 0.8037 (seed 1: 0.7979 / 0.8025). Labels t2avg = mean of the two
teachers (train + valid). r3 specialist from scratch on t2avg (`mingru_ema_cls3_t2`): es_pm 0.3589 / 0.3715 / 0.3777 / 0.3829 /
0.3850 / 0.3872 … vs la50 0.3583 / 0.3713 / 0.3781 / 0.3827 / 0.3859 / 0.3872 (tie at every epoch; la25 led by +0.002–0.005).
After soft FT: r3 rows 0.4417 vs 0.4410 (corr 0.992); replace −0.00005 / +0.00023, pair +0.00002 / +0.00015 → DECIDE 0 (bar +0.0008).
Mechanism ruled out: the la50 teacher's seed noise is not what limits the students (two seeds agree to ~0.001 on valid and the
student curve does not move); a third seed would add less. Teacher-ensemble family closed for seed averaging. Not propagated.
Outcome of the audit round (§55–§56): no change to the packages; finals unchanged in content.

## §57 Final standings and post-competition clean-up (2026-11-21)
- Private LB: sub26a **0.67220, #137 / 1448** (public #120 at close; #8 0.68566, #100 0.67419). Finals: sub26a, sub25, sub24f stack.
- Repo cleaned for publication (branch `portfolio`): only what rebuilds sub26a stays in `wnn/`, `scripts/`, `submission/`;
  queues, gates and superseded scripts in `archive/`. Organizer docs and baseline removed.
- Runtime: `wnn/fused.py` reduced to minGRU + SSM + TCN; int16 helpers in `wnn/quant.py`, tree helpers in `wnn/trees.py`;
  one packager `scripts/package.py` + `configs/final.json`. `tests/parity.py`: the rebuilt package == the submitted sub26a zip,
  max |diff| 0 on valid sequences 0–7 (160k rows, both regimes' routes and stacks).
- Training: `scripts/train.py` + `wnn/nets.py` = `train_gru.py` pruned to the final options; `finetune_valid.py` reads the
  architecture from the base run. Old vs new on short CPU runs of every final recipe (teacher, r3 specialist, SSM, TCN,
  FT half + full; 64–128 sequences, 1 epoch, CPU): max |diff| 0 on every best.pt / last.pt and prediction file. `train_stack.py` reads the valid data directly (no preds/ caches):
  rebuilt stack_v3 r0 / r1 identical (npz and booster text).
- `build_pmask.py` made self-contained again (its row-sampling helper lived in the deleted `train_lgbm.py`);
  `fit_regimes.py` replaces the interactive session that produced `units_*.npy` and `detector.npz`. Units identical; refit
  detector valid acc 0.9989 (= shipped), same prediction as the shipped one on 1871 / 1873 valid sequences.
- Documentation: `docs/WRITEUP.md` (illustrated plain-language write-up) and `docs/img/` (4 diagrams + 5 charts, light / dark
  SVG), drawn by `docs/figures/make_figures.py`. New number measured for it: on every 8th valid sequence, the la50 teacher's
  weighted correlation is 0.725 / 0.708 (t0 / t1) vs 0.549 / 0.540 for the train-only TCN student.
