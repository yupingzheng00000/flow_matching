#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the repository root of flow_matching
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

EXPERIMENTS_FILE="experiments/experiments.tsv"
LOG_DIR="logs"
LEDGER_FILE="${LOG_DIR}/runs.tsv"
MASTER_PORT="${MASTER_PORT:-29501}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-./examples/image/train.py}"
TRAIN_BASENAME="$(basename "${TRAIN_SCRIPT}")"
WAIT_FOR_ACTIVE="${WAIT_FOR_ACTIVE:-1}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-60}"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${EXPERIMENTS_FILE}" ]]; then
    echo "Missing experiment manifest: ${EXPERIMENTS_FILE}" >&2
    exit 1
fi

if [[ ! -f "${LEDGER_FILE}" ]]; then
    # Include per-experiment lut_recon_weight column for provenance
    echo -e "timestamp\tphase\tgamma\tlambda\tlut_recon_weight\trun_name\toutput_dir\tgit_commit\tnote\tstatus" > "${LEDGER_FILE}"
fi

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")"

COMMON_ARGS=(
    "--dataset=cifar10"
    "--discrete_flow_matching"
    "--use_ema"
    "--metric_induced"
    "--batch_size=256"
    "--eval_batch_size=625"
    "--lr=2e-4"
    "--accum_iter=1"
    "--epochs=5900"
    "--fid_samples=5000"
    "--eval_frequency=100"
    "--eval_start_epoch=5600"
    "--resume=models/baseline.pth"
    "--mi_use_gumbel"
    "--mi_gumbel_tau=1.0"
    "--mi_gumbel_tau_start=2.0"
    "--mi_gumbel_tau_end=0.5"
    "--mi_gumbel_tau_anneal_steps=10000"
    "--mi_gumbel_tau_schedule=cosine"
    "--mi_learnable_lut"
    "--mi_lut_num_channels=3"
    "--mi_lut_param_mode=none"
    "--mi_lut_init_method=small_noise_qr"
    "--mi_lut_init_noise_scale=0.1"
    "--mi_metric=lp"
    "--mi_lp=2.0"
    "--mi_lut_emb_dim=16"
    "--mi_freeze_beta_schedule"
    "--compute_fid"
    "--save_eval_gif"
    "--sym_func"
    "--cfg_scale=0.0"
    "--mi_lut_weight_decay=0"
    "--mi_lut_renorm_init_norm"
    "--mi_use_normalized_distance"
    "--wandb"
    "--wandb_project=111"
    "--no_cosine_attention"
    "--no_head_weight_norm"
    "--force_display_start_epoch=4500"
    "--t_weight_mode=linear_t"
    "--t_weight_normalize"
    "--bf16"
)

# Fixed per-experiment value for lut_recon_sample_frac (sampling budget)
DEFAULT_LUT_RECON_SAMPLE_FRAC=0.25

print_command_to_file() {
    local file="$1"
    shift
    local group="$1"
    shift
    {
        echo "# Generated $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# git commit: ${GIT_COMMIT}"
        echo "# CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
        echo "# Arguments recorded for reproducibility"
        echo "WANDB_RUN_GROUP=${group} \\"
        local idx=0
        local total=$#
        for arg in "$@"; do
            idx=$((idx + 1))
            if [[ ${idx} -lt ${total} ]]; then
                printf '  %q \\\n' "$arg"
            else
                printf '  %q\n' "$arg"
            fi
        done
    } > "${file}"
}

wait_for_existing_job() {
    if [[ "${WAIT_FOR_ACTIVE}" == "0" ]]; then
        return
    fi

    if ! command -v pgrep >/dev/null 2>&1 && ! command -v lsof >/dev/null 2>&1; then
        echo "Warning: could not find pgrep or lsof; skipping wait-for-active safeguard."
        return
    fi

    while true; do
        local pid_pattern=""
        local pid_port=""

        if command -v pgrep >/dev/null 2>&1; then
            pid_pattern=$(pgrep -f "torchrun.*${TRAIN_BASENAME}" || true)
        fi
        if command -v lsof >/dev/null 2>&1; then
            pid_port=$(lsof -ti ":${MASTER_PORT}" 2>/dev/null || true)
        fi

        if [[ -z "${pid_pattern}" && -z "${pid_port}" ]]; then
            break
        fi

        local msg="Detected existing training job; waiting ${WAIT_POLL_SECONDS}s before retry..."
        if [[ -n "${pid_pattern}" ]]; then
            msg+=" (pgrep PIDs: ${pid_pattern//$'\n'/, })"
        fi
        if [[ -n "${pid_port}" ]]; then
            msg+=" (port ${MASTER_PORT} in use)"
        fi
        echo "${msg}"
        sleep "${WAIT_POLL_SECONDS}"
    done
}

run_experiment() {
    local gamma="$1"
    local lambda="$2"
    local lut_recon_weight="$3"
    local note="$4"

    if [[ -z "${gamma}" || "${gamma}" == "gamma" ]]; then
        return
    fi

    gamma="${gamma//$'\r'/}"
    lambda="${lambda//$'\r'/}"
    note="${note//$'\r'/}"

    # Resolve lut_recon_weight: use provided per-experiment value or default
    if [[ -z "${lut_recon_weight}" ]]; then
        lut_recon_weight="0.0"
    fi

    # Build run name and output dir; include lut_recon_weight suffix for provenance
    local run_name_base
    run_name_base=$(printf "learned_emb_16d_ga%s_la%s" "${gamma}" "${lambda}")
    local run_name
    run_name=$(printf "%s_rw%s" "${run_name_base}" "${lut_recon_weight}")
    local output_dir="./output_dir/${run_name}"
    local log_file="${LOG_DIR}/${run_name}.log"
    local args_file="${output_dir}/args.txt"
    local group_name="${run_name}"

    mkdir -p "${output_dir}"

    wait_for_existing_job

    local cmd=(
        "torchrun" "--standalone" "--nnodes=1" "--nproc_per_node=8" "--master_port=${MASTER_PORT}"
        "${TRAIN_SCRIPT}"
    )
    cmd+=("${COMMON_ARGS[@]}")
    cmd+=("--output_dir=${output_dir}")
    cmd+=("--wandb_run_name=${run_name}")
    cmd+=("--t_bias_gamma=${gamma}")
    cmd+=("--t_weight_lambda=${lambda}")
    cmd+=("--lut_recon_weight=${lut_recon_weight}")
    cmd+=("--lut_recon_sample_frac=${DEFAULT_LUT_RECON_SAMPLE_FRAC}")

    local start_time
    start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo -e "${start_time}\tSTART\t${gamma}\t${lambda}\t${lut_recon_weight}\t${run_name}\t${output_dir}\t${GIT_COMMIT}\t${note}\t-" >> "${LEDGER_FILE}"
    printf ">>> [%s] Starting gamma=%s lambda=%s (%s)\n" "${start_time}" "${gamma}" "${lambda}" "${note}" | tee -a "${log_file}"

    print_command_to_file "${args_file}" "${group_name}" "${cmd[@]}"

    set +e
    WANDB_RUN_GROUP="${group_name}" "${cmd[@]}" 2>&1 | tee -a "${log_file}"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time
    end_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    local status="OK"
    if [[ "${exit_code}" -ne 0 ]]; then
        status="FAIL(${exit_code})"
    fi
    echo -e "${end_time}\tEND\t${gamma}\t${lambda}\t${lut_recon_weight}\t${run_name}\t${output_dir}\t${GIT_COMMIT}\t${note}\t${status}" >> "${LEDGER_FILE}"
    printf "<<< [%s] Finished gamma=%s lambda=%s status=%s\n" "${end_time}" "${gamma}" "${lambda}" "${status}" | tee -a "${log_file}"

    if [[ "${exit_code}" -ne 0 ]]; then
        echo "Run ${run_name} exited with status ${exit_code}; continuing to next experiment." | tee -a "${log_file}"
    fi
}

while IFS= read -r line || [[ -n "$line" ]]; do
    # Trim line and skip blank or comment lines
    # (keep leading/trailing whitespace handling simple)
    if [[ -z "${line//[[:space:]]/}" ]]; then
        continue
    fi
    if [[ "${line}" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Split on tabs into array 'parts'
    IFS=$'\t' read -r -a parts <<< "$line"
    gamma="${parts[0]:-}"
    lambda="${parts[1]:-}"
    if [ "${#parts[@]}" -ge 4 ]; then
        lut_recon_weight="${parts[2]}"
        note="${parts[3]}"
    elif [ "${#parts[@]}" -eq 3 ]; then
        # Legacy 3-column format: gamma, lambda, note
        lut_recon_weight=""
        note="${parts[2]}"
    else
        lut_recon_weight=""
        note=""
    fi

    run_experiment "${gamma}" "${lambda}" "${lut_recon_weight}" "${note:-}"
done < "${EXPERIMENTS_FILE}"

echo "All experiments from ${EXPERIMENTS_FILE} have been processed."
