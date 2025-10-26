#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--runs N] <onnx_model> [<onnx_model> ...]"
  echo "       (legacy) $0 <onnx_model> [run_count]"
}

if [ $# -lt 1 ]; then
  usage
  exit 1
fi

runs=10
runs_set=false
declare -a models=()

while [ $# -gt 0 ]; do
  case "$1" in
    --runs)
      if [ $# -lt 2 ]; then
        echo "Error: --runs requires a numeric argument." >&2
        exit 1
      fi
      if ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: run count must be numeric; got '$2'." >&2
        exit 1
      fi
      runs="$2"
      runs_set=true
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      models+=("$1")
      shift
      ;;
  esac
done

if ! $runs_set && [ "${#models[@]}" -ge 2 ]; then
  last_index=$((${#models[@]} - 1))
  if [[ "${models[$last_index]}" =~ ^[0-9]+$ ]]; then
    runs="${models[$last_index]}"
    unset "models[$last_index]"
  fi
fi

if [ "${#models[@]}" -eq 0 ]; then
  usage
  exit 1
fi

output_dir="benchmark"
mkdir -p "${output_dir}"

declare -a csv_files=()
declare -a title_labels=()

for onnx_model in "${models[@]}"; do
  if [ ! -f "${onnx_model}" ]; then
    echo "Error: ONNX model not found: ${onnx_model}" >&2
    exit 1
  fi

  model_basename="$(basename "${onnx_model}")"
  model_stem="${model_basename%.*}"
  data_file="${output_dir}/benchmark_times_${model_stem}.csv"
  plot_file="${output_dir}/benchmark_times_${model_stem}.png"
  title_label="${model_stem}"

  if [[ "${model_stem}" =~ _([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)_ ]]; then
    input_batch="${BASH_REMATCH[1]}"
    input_channels="${BASH_REMATCH[2]}"
    input_height="${BASH_REMATCH[3]}"
    input_width="${BASH_REMATCH[4]}"
  else
    echo "Error: Could not infer input shape from model filename '${model_basename}'. Expected pattern '*_<BxCxHxW>_*'." >&2
    exit 1
  fi
  echo "Detected input shape: ${input_batch}x${input_channels}x${input_height}x${input_width}"

  csv_files+=("${data_file}")
  title_labels+=("${title_label}")

  echo "Saving timing data to ${data_file}"
  echo "run,avg_ms" > "${data_file}"

  for run_id in $(seq 1 "${runs}"); do
    echo "[${model_basename}] Running iteration ${run_id}/${runs}"
    tmp_output="$(mktemp)"

    if ! uv run sit4onnx -if "${onnx_model}" -tlc 1000 -oep tensorrt -fs "${input_batch}" "${input_channels}" "${input_height}" "${input_width}" -fs 1 "${run_id}" 4 | tee "${tmp_output}"; then
      echo "sit4onnx failed on iteration ${run_id} for ${model_basename}; aborting." >&2
      rm -f "${tmp_output}"
      exit 1
    fi

    if ! avg_line="$(grep 'avg elapsed time per pred' "${tmp_output}")"; then
      echo "Could not find average elapsed time in the output for iteration ${run_id} (${model_basename}); aborting." >&2
      rm -f "${tmp_output}"
      exit 1
    fi

    avg_value="$(printf '%s\n' "${avg_line}" | awk '{print $(NF-1)}')"
    avg_value_rounded="$(printf '%.2f' "${avg_value}")"
    echo "Parsed avg elapsed time per pred: ${avg_value_rounded} ms"
    printf '%d,%s\n' "${run_id}" "${avg_value_rounded}" >> "${data_file}"
    rm -f "${tmp_output}"
  done

  uv run python - "${data_file}" "${plot_file}" "${title_label}" <<'PY'
import csv
import sys

data_file, plot_file, title_label = sys.argv[1:]

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
except ImportError as exc:
    sys.exit(f"matplotlib is required to render the plot: {exc}")

runs, times = [], []

with open(data_file, newline="") as fp:
    reader = csv.DictReader(fp)
    for row in reader:
        runs.append(int(row["run"]))
        times.append(float(row["avg_ms"]))

if not runs:
    sys.exit("No timing data available to plot.")

fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.plot(runs, times, marker="o", linestyle="-", color="#1f77b4")
ax.set_xlabel("Number of people to be inferred")
ax.set_ylabel("Average elapsed time per prediction (ms)")
ax.set_title(f"{title_label}\navg elapsed time per prediction")
ax.set_xticks(runs)
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

for x, y in zip(runs, times):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(plot_file, dpi=200)
PY

  echo "Saved plot to ${plot_file}"
done

if [ "${#csv_files[@]}" -gt 1 ]; then
  combined_plot="${output_dir}/benchmark_times_combined.png"
  python_args=("${combined_plot}")

  for idx in "${!csv_files[@]}"; do
    python_args+=("${csv_files[$idx]}")
    python_args+=("${title_labels[$idx]}")
  done

  uv run python - "${python_args[@]}" <<'PY'
import csv
import sys

args = sys.argv[1:]

if len(args) < 3 or len(args) % 2 == 0:
    sys.exit("Combined plot requires pairs of <csv_path> <label> arguments.")

combined_plot = args[0]
pairs = args[1:]

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
except ImportError as exc:
    sys.exit(f"matplotlib is required to render the combined plot: {exc}")

series = []
for i in range(0, len(pairs), 2):
    csv_path = pairs[i]
    label = pairs[i + 1]
    runs, times = [], []

    with open(csv_path, newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            runs.append(int(row["run"]))
            times.append(float(row["avg_ms"]))

    if not runs:
        sys.exit(f"No timing data found in {csv_path}.")

    series.append((label, runs, times))

fig, ax = plt.subplots(figsize=(6.4, 4.0))

for label, runs, times in series:
    ax.plot(runs, times, marker="o", linestyle="-", linewidth=1.5, label=label)

ax.set_xlabel("Number of people to be inferred")
ax.set_ylabel("Average elapsed time per prediction (ms)")
ax.set_title("Combined avg elapsed time per prediction")
ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend()

fig.tight_layout()
fig.savefig(combined_plot, dpi=200)
PY

  echo "Saved combined plot to ${combined_plot}"
fi
