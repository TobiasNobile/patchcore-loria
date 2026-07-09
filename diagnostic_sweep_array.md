# Diagnostic — Job array d'inspection PatchCore (SLURM `200884`)

**Date :** 2026-07-08
**Lanceur :** [`run_inspect_array.sh`](run_inspect_array.sh) → `inspect_array.sbatch` (array `0-11%4`)
**Script exécuté :** [`bin/inspect_patchcore_celeba.py`](bin/inspect_patchcore_celeba.py)
**Partition :** `gpu_prod_long` · **Walltime/tâche :** `02:00:00` · **Concurrence :** 4

---

## 1. Résumé exécutif

Sur les **12 tâches** (1 seed × 2 `train_subset` × [1 identity + 5 coreset], chacune sur les images 961 & 405) :

- ✅ **6/12 réussies** — toutes les tâches `train_subset=2000`.
- ❌ **1/12 échouée** — tâche 6 (`ts=10000`, sampler `identity`) : **tuée par l'OOM-killer** (`ExitCode 9:0`, `State=FAILED`).
- 🟠 **5/12 encore en cours ou en attente** au moment du diagnostic — les 5 tâches `ts=10000` coreset. **Non encore générées**, et exposées à deux risques (timeout 2h et OOM, cf. §6).

**Cause racine du plantage :** le job ne réservait **aucune mémoire** (`--mem` absent du sbatch). À `train_subset=10000`, la banque de features fait ~**5,5 M vecteurs × 1024 dims ≈ 22 Go de RAM**. Sur un nœud partagé, la tâche 6 n'a pas eu assez de mémoire et a été tuée pendant l'extraction des features.

---

## 2. Ce qui était attendu

24 images PNG = 12 combinaisons × 2 images, rangées ainsi côté serveur :

```
results/idx961/inspect_celeba_hat_<sampler>_<tag>_s42_ts<TS>_wideresnet50_idx961.png
results/idx405/inspect_celeba_hat_<sampler>_<tag>_s42_ts<TS>_wideresnet50_idx405.png
```

avec les 12 combinaisons :

| # tâche | train_subset | sampler | percentage |
|---|---|---|---|
| 0 | 2000 | identity | — |
| 1-5 | 2000 | approx_greedy_coreset | 0.01 / 0.1 / 0.2 / 0.5 / 0.7 |
| 6 | 10000 | identity | — |
| 7-11 | 10000 | approx_greedy_coreset | 0.01 / 0.1 / 0.2 / 0.5 / 0.7 |

---

## 3. État réel observé (`sacct -j 200884`)

| # | train_subset | sampler / pct | État | Elapsed |
|---|---|---|---|---|
| 0 | 2000 | identity | ✅ COMPLETED | 00:01:32 |
| 1 | 2000 | p=0.01 | ✅ COMPLETED | 00:02:02 |
| 2 | 2000 | p=0.1 | ✅ COMPLETED | 00:04:12 |
| 3 | 2000 | p=0.2 | ✅ COMPLETED | 00:07:14 |
| 4 | 2000 | p=0.5 | ✅ COMPLETED | 00:16:03 |
| 5 | 2000 | p=0.7 | ✅ COMPLETED | 00:22:04 |
| 6 | 10000 | identity | ❌ **FAILED (9:0)** | 00:38:05 |
| 7 | 10000 | p=0.01 | 🟠 RUNNING | 01:23+ |
| 8 | 10000 | p=0.1 | 🟠 RUNNING | 01:13+ |
| 9 | 10000 | p=0.2 | 🟠 RUNNING | 01:06+ |
| 10 | 10000 | p=0.5 | 🟠 RUNNING | 00:48+ |
| 11 | 10000 | p=0.7 | ⏳ PENDING (throttle %4) | — |

**Observation clé sur le coût :** à `ts=2000`, le temps croît fortement avec le percentage (identity 1min30 → p=0.7 22min) — c'est le **subsampling coreset** qui domine (ex. task 5 : `Subsampling 1 097 600 features [20:46]`). À `ts=10000`, même les faibles percentages tournent **> 1h20**.

---

## 4. Ce qui a / n'a PAS été généré

**Généré (12 PNG, `s42`) :** les 6 combinaisons `ts=2000` × 2 images, dans `results/idx961/` et `results/idx405/`.

```
results/idx961/inspect_celeba_hat_identity_nopct_s42_ts2000_wideresnet50_idx961.png
results/idx961/inspect_celeba_hat_approx_greedy_coreset_p0.01_s42_ts2000_wideresnet50_idx961.png
   … (p0.1, p0.2, p0.5, p0.7)
results/idx405/…  (idem, 6 fichiers)
```

**NON généré (12 PNG manquants, tous `ts=10000`) :**

| Combinaison | Raison |
|---|---|
| `ts=10000 identity` (idx961 + idx405) | ❌ **Tâche 6 tuée (OOM)** — ne sera pas produite sans correctif |
| `ts=10000 p=0.01 / 0.1 / 0.2` (×2 img) | 🟠 tâches encore en cours — produites *si* elles ne timeout/OOM pas |
| `ts=10000 p=0.5 / 0.7` (×2 img) | 🟠 en cours / en attente — **risque de timeout le plus élevé** |

> Remarque : la connexion `Broken pipe / Connection closed by remote host` vue côté client **n'est pas** un plantage des jobs — c'est la session SSH `tail -f` qui a été fermée par le login node. Les jobs SLURM sont indépendants de cette session.

---

## 5. Cause racine du plantage (tâche 6)

Extrait du log `slurm_logs/inspect_200884_6.out` :

```
Computing support features...:  84%|████████▍ | 1047/1250 [03:00<00:34,  5.86it/s]
Computing support features...:  84%|████████▍ | 1048/1250 [03:05<05:31,  1.64s/it]
Computing support features...:  84%|████████▍ | 1049/1250 [33:46<30:54:45, 553.66s/it]
/var/spool/slurm/.../slurm_script: line 30: 23375 Killed   python bin/inspect_patchcore_celeba.py …
```

**Interprétation :**
1. À l'itération 1048→1049, la vitesse s'effondre de **~6 it/s à 553 s/it** : le process passe ~30 min bloqué sur un seul batch → **thrashing mémoire** (la RAM est saturée, le système swappe).
2. Le process reçoit ensuite un **SIGKILL** (`Killed`) → c'est la signature typique d'un **OOM-kill** (out-of-memory), déclenché soit par le cgroup mémoire SLURM, soit par l'OOM-killer du noyau.
3. `sacct` confirme : `State=FAILED`, `ExitCode=9:0`, alors que l'étape `.extern` est `COMPLETED` → l'échec vient bien du process Python, pas de l'infrastructure.

**Pourquoi la mémoire explose à `ts=10000` :**
- La banque de features PatchCore contient ~**549 vecteurs/image** (mesuré : `1 097 600 features` pour `ts=2000` → 548,8/image).
- À `ts=10000` → ~**5,49 M vecteurs × 1024 dims × 4 octets ≈ 22 Go** rien que pour la matrice de features, avant même l'index FAISS.
- L'étape `Computing support features` **accumule tous ces vecteurs en RAM** → pic > 20 Go.

---

## 6. Facteurs contributifs

1. **Aucune réservation mémoire.** Le sbatch (`inspect_array.sbatch`) ne contient pas de `#SBATCH --mem=…`. La tâche prend donc la mémoire *disponible* sur un nœud **partagé** → sur un nœud déjà chargé par d'autres jobs, les ~22 Go nécessaires ne sont pas garantis. C'est pourquoi la tâche 6 est morte alors que les tâches 7-11 (même stade mémoire) ont survécu : simple différence de RAM libre selon le nœud.
2. **`identity` est le pire cas mémoire en aval.** Contrairement au coreset qui sous-échantillonne, `IdentitySampler` **conserve 100 % des features** dans la banque + l'index FAISS → empreinte maximale. À `ts=10000` c'est intenable sans réservation dédiée.
3. **`ts=10000` est intrinsèquement lourd** (temps ET mémoire). Les 5 tâches coreset restantes sont donc exposées :
   - **Timeout :** task 7 (p=0.01) tourne déjà depuis 1h23 sur un walltime de 2h ; les p=0.5/0.7 (subsampling le plus long) risquent de dépasser 2h.
   - **OOM :** elles construisent aussi la matrice ~22 Go transitoirement ; sans `--mem` le risque persiste selon le nœud.

---

## 7. Correctifs recommandés

> **Statut (2026-07-08) — correctifs appliqués** dans [`run_inspect_array.sh`](run_inspect_array.sh) :
>
> **Contrainte matérielle découverte :** les nœuds GPU (`gpu_prod_long` **et** `gpu_inter`)
> n'ont que **30 Go de RAM chacun** (`RealMemory=30000 Mo`), partagés entre jobs. D'où :
> - `--mem=64G` **rejeté** (« Requested node configuration is not available »).
> - **identity à ts=10000 est infaisable ici** : il garde 100 % des features **+** un index
>   FAISS complet ≈ 2 × 22,5 Go ≈ **45 Go** > 30 Go. → **exclu** du sweep au-delà de
>   `IDENTITY_MAX_TS=2000`.
> - coreset à ts=10000 : pic ~22,5 Go → tient si la tâche a le nœud pour elle.
>
> Correctifs : `--mem=29G` (réserve ~tout le nœud → pas d'OOM par partage),
> `--time=04:00:00`, exclusion identity@ts>2000, et **skip idempotent** des PNG déjà
> présents. Relancer le script ne refait que les combinaisons manquantes.
>
> **Mise à jour — ts=10000 abandonné (même en coreset).** Le subsampling coreset ne
> réduit la mémoire qu'APRÈS avoir accumulé toutes les features : le pic ~22,5 Go
> (+ copies transitoires) sature les 30 Go du nœud vers **84 % de « Computing support
> features »** → thrashing (74 s/it) puis timeout/OOM, au même point que l'identity.
> Conclusion : **`train_subset` plafonné à 5000** sur ces nœuds (pic ~20 Go, tient).
> `TRAIN_SUBSETS=(2000 5000)`. 7000 serait la limite haute risquée (~28 Go).

### 7.1 Réserver la mémoire (prioritaire)
Ajouter au sbatch une réservation confortable pour `ts=10000` :

```bash
#SBATCH --mem=48G          # marge sur les ~22 Go de la matrice de features
```

(dans [`run_inspect_array.sh`](run_inspect_array.sh), section `cat > "${SBATCH}"`).

### 7.2 Relancer uniquement les tâches manquantes
Ne pas relancer les 6 tâches `ts=2000` déjà réussies. Deux options :
- **Relance ciblée** de la tâche 6 (identity ts=10000) et de toute tâche 7-11 qui aurait timeout/OOM, avec `--mem=48G` (voire `--mem=64G` pour l'identity).
- **Ré-array partiel** : régénérer un manifest ne contenant que les combinaisons `ts=10000` manquantes et soumettre `sbatch --array=…`.

### 7.3 Décision sur `identity @ ts=10000`
Cette combinaison est la plus coûteuse (mémoire + temps de recherche NN sur banque complète). À trancher :
- soit lui allouer explicitement beaucoup de RAM (`--mem=64G`) et l'isoler,
- soit l'**exclure** du sweep à `ts=10000` (peu d'intérêt : banque non compressée = référence, mais très lente).

### 7.4 (Optionnel) Walltime
Si des p=0.5/0.7 à `ts=10000` timeout, `gpu_prod_long` autorise jusqu'à **2 jours** — passer `SLURM_TIME` à `04:00:00`.

---

## 8. Annexe — commandes de diagnostic utilisées

```bash
# Bilan par tâche
ssh nobile_tob@dce.metz.centralesupelec.fr \
  'sacct -j 200884 --format=JobID%15,State%12,Elapsed,ExitCode,ReqMem --units=G'

# Ce qui tourne encore
ssh nobile_tob@dce.metz.centralesupelec.fr 'squeue -u nobile_tob'

# PNG réellement produits
ssh nobile_tob@dce.metz.centralesupelec.fr \
  'find ~/patchcore-inspection/results -name "*.png"'

# Cause d'échec d'une tâche
ssh nobile_tob@dce.metz.centralesupelec.fr \
  'grep -nE "Killed|Error|Traceback|out of memory" \
     ~/patchcore-inspection/slurm_logs/inspect_200884_6.out'
```
