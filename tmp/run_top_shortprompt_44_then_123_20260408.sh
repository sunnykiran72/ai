#!/bin/zsh
set -euo pipefail

python3 /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/scripts/run_bulk_tryon_benchmark.py \
  --base-url https://1r6ln3rbln3jhh-8000.proxy.runpod.net \
  --users-dir /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/tmp/top_tryon_users_20260408_cached110 \
  --garment-url https://glamifydevstorage.blob.core.windows.net/wardrobe-outputs/1a53648c-d068-4fce-a0c9-2fe6ed1b7989.png \
  --garment-type top \
  --garment-prompt-file /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/tmp/top_garment_prompt_short_20260408.txt \
  --steps 12 \
  --seed 44 \
  --guidance-scale 2.5 \
  --lora-scale 1.0 \
  --output-max-edge 1280 \
  --prepare-cache-jsonl /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/debug_outputs/user_prep_minicpm_closed_vocab_20260408_cached110/results.jsonl \
  --prepare-cache-csv /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/pod_sync/2026-04-08/latest_pull_20260408_111049/new_dataset_prepare_only_20260408/results.csv \
  --prepare-cache-only \
  --output-dir /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/debug_outputs/bulk_tryon_seed44_steps12_top_1280_20260408_shortprompt_cached110 \
  --resume

python3 /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/scripts/generate_bulk_tryon_local_report.py \
  --run-dir bulk_tryon_seed44_steps12_top_1280_20260408_shortprompt_cached110 \
  --title "Glamify Bulk Tryon Testing" \
  --output-name local_report.html

python3 /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/scripts/run_bulk_tryon_benchmark.py \
  --base-url https://1r6ln3rbln3jhh-8000.proxy.runpod.net \
  --users-dir /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/tmp/top_tryon_users_20260408_cached110 \
  --garment-url https://glamifydevstorage.blob.core.windows.net/wardrobe-outputs/1a53648c-d068-4fce-a0c9-2fe6ed1b7989.png \
  --garment-type top \
  --garment-prompt-file /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/tmp/top_garment_prompt_short_20260408.txt \
  --steps 12 \
  --seed 123 \
  --guidance-scale 2.5 \
  --lora-scale 1.0 \
  --output-max-edge 1280 \
  --prepare-cache-jsonl /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/debug_outputs/user_prep_minicpm_closed_vocab_20260408_cached110/results.jsonl \
  --prepare-cache-csv /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/pod_sync/2026-04-08/latest_pull_20260408_111049/new_dataset_prepare_only_20260408/results.csv \
  --prepare-cache-only \
  --output-dir /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/debug_outputs/bulk_tryon_seed123_steps12_top_1280_20260408_shortprompt_cached110 \
  --resume

python3 /Users/kiran/Documents/github/clients/glamify/hybrid_vto_v1/ai/scripts/generate_bulk_tryon_local_report.py \
  --run-dir bulk_tryon_seed123_steps12_top_1280_20260408_shortprompt_cached110 \
  --title "Glamify Bulk Tryon Testing" \
  --output-name local_report.html
