#!/usr/bin/env bash
# Corre las configuraciones del experimento tc5061_ai_vs_human_tfidf_logreg.
# Uso: DATA=data/AI_vs_Human_Text_Classification_Benchmark.csv ./run_experiments.sh
set -euo pipefail
DATA=${DATA:-data/AI_vs_Human_Text_Classification_Benchmark.csv}

python train.py --data-path "$DATA" --run-name r1_baseline                                    # valores por defecto
python train.py --data-path "$DATA" --run-name r2_C_0.001        --C 0.001
python train.py --data-path "$DATA" --run-name r3_C_100          --C 100
python train.py --data-path "$DATA" --run-name r4_mf_10          --max-features 10
python train.py --data-path "$DATA" --run-name r5_mf_50_ng1      --max-features 50 --ngram-max 1
python train.py --data-path "$DATA" --run-name r6_ng3_md10       --ngram-max 3 --min-df 10
python train.py --data-path "$DATA" --run-name r7_mf_10_C_0.01   --max-features 10 --C 0.01
# Reproducibilidad: mismos argumentos que r4 -> mismas metricas y misma huella de predicciones
python train.py --data-path "$DATA" --run-name r8_mf_10_repeat   --max-features 10
