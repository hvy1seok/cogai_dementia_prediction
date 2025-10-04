#!/usr/bin/env bash
# run_all.sh — True SigLIP2 EN+MN Training (ERM vs SAM) + Loss Landscape + Repr Viz + W&B
set -euo pipefail

############################
# 0) 사용자 설정
############################
DATA_DIR="../../training_dset"
OUTPUT_DIR="../modules/outputs/siglip-sam/AllInOne_EN_CN"
CKPT_DIR="${OUTPUT_DIR}/checkpoints"
FIG_DIR="${OUTPUT_DIR}/figures"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "$OUTPUT_DIR" "$CKPT_DIR" "$FIG_DIR" "$LOG_DIR"

# W&B 환경변수는 외부에서 export된 값을 그대로 사용 (프로젝트 이름만 기본 제공)
export WANDB_PROJECT=${WANDB_PROJECT:-"siglip2_enman_allinone"}

# 분석 파라미터 (필요시 조정)
LAND_ALPHA_MIN=-1.0
LAND_ALPHA_MAX=1.0
LAND_ALPHA_STEPS=21
LAND_BETA_MIN=-1.0
LAND_BETA_MAX=1.0
LAND_BETA_STEPS=21
VAL_BATCHES=6
EMBED_TOOL="umap"   # or tsne
EMBED_SEED=0
EMBED_MAXPTS=2000

python3 - <<'PY'
import sys, subprocess
req = [
    "wandb","numpy","torch","scikit-learn","umap-learn","matplotlib","plotly"
]
for p in req:
    try:
        __import__(p.split('-')[0])
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
print("[Setup] OK")
PY

############################
# 1) ERM(AdamW) 훈련
############################
echo "[Train] ERM (AdamW)"
bash train_true_siglip2_2languages_en_cn_erm.sh | tee "${LOG_DIR}/train_ERM.log"

# 베스트 체크포인트 표준화
BEST_ERM="$(ls ../modules/outputs/siglip-sam/checkpoints/best_model*.pt 2>/dev/null | head -1 || true)"
if [ -z "$BEST_ERM" ]; then
  BEST_ERM="$(ls ../modules/outputs/siglip-sam/checkpoints/checkpoint_epoch_*.pt 2>/dev/null | tail -1 || true)"
fi
if [ -z "$BEST_ERM" ]; then echo "❌ ERM 체크포인트를 찾지 못했습니다"; exit 1; fi
cp "$BEST_ERM" "${CKPT_DIR}/siglip2_ENCN_ERM.pt"
echo "[Train] ERM best -> ${CKPT_DIR}/siglip2_ENCN_ERM.pt"

############################
# 2) SAM 훈련
############################
echo "[Train] SAM"
bash train_true_siglip2_2languages_en_cn.sh | tee "${LOG_DIR}/train_SAM.log"

BEST_SAM="$(ls ../modules/outputs/siglip-sam/checkpoints/best_model*.pt 2>/dev/null | head -1 || true)"
if [ -z "$BEST_SAM" ]; then
  BEST_SAM="$(ls ../modules/outputs/siglip-sam/checkpoints/checkpoint_epoch_*.pt 2>/dev/null | tail -1 || true)"
fi
if [ -z "$BEST_SAM" ]; then echo "❌ SAM 체크포인트를 찾지 못했습니다"; exit 1; fi
cp "$BEST_SAM" "${CKPT_DIR}/siglip2_ENCN_SAM.pt"
echo "[Train] SAM best -> ${CKPT_DIR}/siglip2_ENCN_SAM.pt"

############################
# 3) Loss Landscape (ERM/SAM/Barrier)
############################
echo "[Analysis] Loss Landscape (ERM)"
python3 analysis_loss_landscape.py \
  --checkpoint "${CKPT_DIR}/siglip2_ENCN_ERM.pt" \
  --data_dir "${DATA_DIR}" \
  --alphamin ${LAND_ALPHA_MIN} --alphamax ${LAND_ALPHA_MAX} --alphasteps ${LAND_ALPHA_STEPS} \
  --betamin  ${LAND_BETA_MIN}  --betamax  ${LAND_BETA_MAX}  --betasteps  ${LAND_BETA_STEPS} \
  --val_batches ${VAL_BATCHES} \
  --out "${FIG_DIR}/loss_landscape_ERM.png" \
  --wandb_title "Loss Landscape ERM"

echo "[Analysis] Loss Landscape (SAM)"
python3 analysis_loss_landscape.py \
  --checkpoint "${CKPT_DIR}/siglip2_ENCN_SAM.pt" \
  --data_dir "${DATA_DIR}" \
  --alphamin ${LAND_ALPHA_MIN} --alphamax ${LAND_ALPHA_MAX} --alphasteps ${LAND_ALPHA_STEPS} \
  --betamin  ${LAND_BETA_MIN}  --betamax  ${LAND_BETA_MAX}  --betasteps  ${LAND_BETA_STEPS} \
  --val_batches ${VAL_BATCHES} \
  --out "${FIG_DIR}/loss_landscape_SAM.png" \
  --wandb_title "Loss Landscape SAM"

echo "[Analysis] Loss Barrier (ERM→SAM)"
python3 analysis_loss_landscape.py \
  --checkpoint_a "${CKPT_DIR}/siglip2_ENCN_ERM.pt" \
  --checkpoint_b "${CKPT_DIR}/siglip2_ENCN_SAM.pt" \
  --data_dir "${DATA_DIR}" \
  --loss_barrier \
  --t_steps 31 \
  --out "${FIG_DIR}/loss_barrier_ERM_to_SAM.png" \
  --wandb_title "Loss Barrier ERM→SAM"

############################
# 4) Representation Viz (UMAP/t-SNE)
############################
echo "[Analysis] Representation (ERM & SAM)"
python3 analysis_repr.py \
  --checkpoints "${CKPT_DIR}/siglip2_ENCN_ERM.pt" "${CKPT_DIR}/siglip2_ENCN_SAM.pt" \
  --data_dir "${DATA_DIR}" \
  --embed_source "penultimate" \
  --tool "${EMBED_TOOL}" \
  --seed ${EMBED_SEED} \
  --max_points ${EMBED_MAXPTS} \
  --out_prefix "${FIG_DIR}/repr_ENCN" \
  --wandb_title "${EMBED_TOOL^^} ERM vs SAM (EN+CN)"

echo "✅ Done"
echo " - Checkpoints: ${CKPT_DIR}/siglip2_ENCN_ERM.pt, ${CKPT_DIR}/siglip2_ENCN_SAM.pt"
echo " - Figures: ${FIG_DIR}/*.png"
echo " - Logs: ${LOG_DIR}/*.log"
echo " - W&B Project: ${WANDB_PROJECT:-unset}"


