#!/usr/bin/env python3
"""Compare configured and spectral primal/dual parameters on random Gset cases."""

from __future__ import annotations

import argparse
import csv
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.problem_parser import generate_Max_cut as generate_max_cut, parse_gset
from src.solver import PDBO_CPU
from utils import fix_seed


@dataclass(frozen=True)
class Configuration:
    key: str
    label: str
    primal_lr_mode: str
    dual_init_mode: str
    spectral_step_fraction: float = 0.5


CONFIGURATIONS = (
    Configuration(
        key="control",
        label="对照组（固定 lr_x、固定 dual_init）",
        primal_lr_mode="configured",
        dual_init_mode="configured",
    ),
    Configuration(
        key="spectral_dual",
        label="实验组一（谱 dual_init、固定 lr_x）",
        primal_lr_mode="configured",
        dual_init_mode="spectral",
    ),
    Configuration(
        key="full_spectral",
        label="实验组二（谱 dual_init、谱 lr_x）",
        primal_lr_mode="spectral",
        dual_init_mode="spectral",
        spectral_step_fraction=0.99,
    ),
    Configuration(
        key="fixed_dual_spectral",
        label="实验组三（固定 dual_init、谱 lr_x）",
        primal_lr_mode="spectral",
        dual_init_mode="configured",
        spectral_step_fraction=0.99,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare four parameter configurations across Gset instances."
        )
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=None,
        help="sample this many instances for a reduced run; default is all available Gset cases",
    )
    parser.add_argument(
        "--gset-ids",
        nargs="+",
        type=int,
        help="explicit instance IDs; primarily useful for a small smoke run",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=0,
        help="seed used only for random instance selection (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="PDBO seed (default: 0)")
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--lr-x", type=float, default=0.02)
    parser.add_argument("--lr-y", type=float, default=0.01)
    parser.add_argument("--dual-init", type=float, default=5.0)
    parser.add_argument("--dual-patience-threshold", type=float, default=1e-4)
    parser.add_argument("--dual-patience-every", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/results/lr_x_dual_init"),
    )
    return parser.parse_args()


def available_gset_ids() -> List[int]:
    ids = []
    for path in (ROOT / "instance" / "Gset").glob("G*.txt"):
        match = re.fullmatch(r"G(\d+)\.txt", path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def select_gset_ids(args: argparse.Namespace) -> List[int]:
    available = available_gset_ids()
    available_set = set(available)
    if args.gset_ids:
        selected = list(dict.fromkeys(args.gset_ids))
        missing = sorted(set(selected) - available_set)
        if missing:
            raise ValueError(f"Gset instances not found: {missing}")
        return selected
    if args.instance_count is None:
        return available
    if args.instance_count > len(available):
        raise ValueError(
            f"instance-count={args.instance_count} exceeds {len(available)} available cases"
        )
    return sorted(random.Random(args.selection_seed).sample(available, args.instance_count))


def validate_args(args: argparse.Namespace) -> None:
    if args.instance_count is not None and args.instance_count < 1:
        raise ValueError("instance-count must be positive")
    if args.max_iters < 1 or args.sample_every < 1:
        raise ValueError("max-iters and sample-every must be positive")
    if args.sample_every > args.max_iters:
        raise ValueError("sample-every cannot exceed max-iters")
    if args.batch < 1:
        raise ValueError("batch must be positive")
    if args.lr_x <= 0.0 or args.lr_y <= 0.0 or args.dual_init <= 0.0:
        raise ValueError("lr-x, lr-y, and dual-init must be positive")
    if args.dual_patience_threshold < 0.0:
        raise ValueError("dual-patience-threshold must be non-negative")
    if args.dual_patience_every < 1:
        raise ValueError("dual-patience-every must be positive")


def checkpoint_iterations(max_iters: int, sample_every: int) -> List[int]:
    checkpoints = list(range(sample_every, max_iters + 1, sample_every))
    if checkpoints[-1] != max_iters:
        checkpoints.append(max_iters)
    return checkpoints


def run_configuration(
    data: Dict[str, Any],
    config: Configuration,
    args: argparse.Namespace,
    checkpoints: Sequence[int],
) -> Dict[str, Any]:
    checkpoint_set = set(checkpoints)
    samples: Dict[int, Dict[str, float]] = {}
    cumulative_step_seconds = 0.0

    def capture(step: int, step_seconds: float, objective: float, _incumbent: Any) -> None:
        nonlocal cumulative_step_seconds
        cumulative_step_seconds += step_seconds
        iteration = step + 1
        if iteration in checkpoint_set:
            samples[iteration] = {
                "best_objective": float(objective),
                "best_cut": -float(objective),
                "solve_seconds": cumulative_step_seconds,
            }

    # Match a separate invocation of src/main.py for every paired configuration.
    fix_seed(args.seed)
    initialization_started = time.perf_counter()
    solver = PDBO_CPU(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        batch_size=args.batch,
        primal_lr=args.lr_x,
        primal_lr_mode=config.primal_lr_mode,
        spectral_step_fraction=config.spectral_step_fraction,
        dual_lr=args.lr_y,
        dual_init=args.dual_init,
        dual_init_mode=config.dual_init_mode,
        dual_burn_in=0,
        max_iters=args.max_iters,
        primal_init="center_uniform",
        rho=0.05,
        seed=args.seed,
        verbose=False,
        dual_patience_threshold=args.dual_patience_threshold,
        dual_patience_every=args.dual_patience_every,
        step_callback=capture,
    )
    initialization_seconds = time.perf_counter() - initialization_started
    result = solver.optimize()
    final_iteration = int(solver.iterations_completed)
    if final_iteration not in samples:
        samples[final_iteration] = {
            "best_objective": float(solver.objective),
            "best_cut": -float(solver.objective),
            "solve_seconds": cumulative_step_seconds,
        }

    return {
        "samples": samples,
        "effective_primal_lr": float(solver.primal_lr),
        "effective_dual_init": float(solver.dual_init),
        "initialization_seconds": initialization_seconds,
        "runtime_seconds": float(result.runtime),
        "stop_reason": str(result.stop_reason),
    }


def merge_instance_rows(
    gset_id: int,
    n: int,
    m: int,
    checkpoints: Sequence[int],
    results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    iterations = sorted(
        {
            iteration
            for config in CONFIGURATIONS
            for iteration in results[config.key]["samples"]
        }
    )
    for iteration in iterations:
        row: Dict[str, Any] = {
            "gset_id": gset_id,
            "n": n,
            "m": m,
            "iteration": iteration,
        }
        for config in CONFIGURATIONS:
            sample = results[config.key]["samples"].get(iteration)
            row[f"{config.key}_best_objective"] = (
                sample["best_objective"] if sample is not None else ""
            )
            row[f"{config.key}_best_cut"] = sample["best_cut"] if sample is not None else ""
            row[f"{config.key}_solve_seconds"] = (
                sample["solve_seconds"] if sample is not None else ""
            )
        rows.append(row)
    return rows


def make_summary_row(
    gset_id: int,
    n: int,
    m: int,
    results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {"gset_id": gset_id, "n": n, "m": m}
    for config in CONFIGURATIONS:
        final = results[config.key]["samples"][max(results[config.key]["samples"])]
        row[f"{config.key}_final_cut"] = final["best_cut"]
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean(values: Iterable[float]) -> float:
    return statistics.fmean(values)


def win_tie_loss(gains: Sequence[float], tolerance: float = 1e-9) -> str:
    wins = sum(value > tolerance for value in gains)
    ties = sum(abs(value) <= tolerance for value in gains)
    losses = sum(value < -tolerance for value in gains)
    return f"{wins}/{ties}/{losses}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(
    path: Path,
    args: argparse.Namespace,
    selected_ids: Sequence[int],
    checkpoints: Sequence[int],
    timeseries_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    total_seconds: float,
) -> None:
    selection_note = (
        "- 实例选择：全部可用 Gset 实例"
        if args.instance_count is None and not args.gset_ids
        else f"- 随机抽样种子：{args.selection_seed}"
    )
    config_rows = [
        [
            config.label,
            config.primal_lr_mode,
            f"{args.lr_x:g}" if config.primal_lr_mode == "configured" else "spectral",
            config.dual_init_mode,
            f"{args.dual_init:g}" if config.dual_init_mode == "configured" else "spectral",
            f"{config.spectral_step_fraction:g}"
            if config.primal_lr_mode == "spectral"
            else "-",
        ]
        for config in CONFIGURATIONS
    ]

    final_rows = []
    for config in CONFIGURATIONS:
        cuts = [float(row[f"{config.key}_final_cut"]) for row in summary_rows]
        if config.key == "control":
            gain = "-"
            wtl = "-"
        else:
            gains = [
                float(row[f"{config.key}_final_cut"])
                - float(row["control_final_cut"])
                for row in summary_rows
            ]
            gain = f"{mean(gains):.3f}"
            wtl = win_tie_loss(gains)
        final_rows.append(
            [
                config.label,
                f"{mean(cuts):.3f}",
                f"{statistics.median(cuts):.3f}",
                gain,
                wtl,
            ]
        )

    trajectory_rows = []
    trajectory_iterations = sorted({int(row["iteration"]) for row in timeseries_rows})
    for iteration in trajectory_iterations:
        rows_at_iteration = [
            row for row in timeseries_rows if int(row["iteration"]) == iteration
        ]
        if not rows_at_iteration:
            continue
        values = []
        for config in CONFIGURATIONS:
            available = [
                float(row[f"{config.key}_best_cut"])
                for row in rows_at_iteration
                if row[f"{config.key}_best_cut"] != ""
            ]
            values.append(f"{mean(available):.3f}" if available else "-")
        trajectory_rows.append([str(iteration), *values])

    lines = [
        "# lr_x 与 dual_init 对 Gset Max-Cut 求解的影响",
        "",
        "## 实验设置",
        "",
        f"- 完成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Gset 实例数：{len(selected_ids)}",
        selection_note,
        f"- 求解器种子：{args.seed}",
        f"- 公共参数：`batch={args.batch}`，`lr_y={args.lr_y:g}`，"
        f"`max_iters={args.max_iters}`，`primal_init=center_uniform`，`rho=0.05`",
        f"- dual patience：每 {args.dual_patience_every} 次迭代检查 dual 均值下降量，"
        f"小于 `{args.dual_patience_threshold:g}` 时停止",
        f"- 记录间隔：每 {args.sample_every} 次迭代；dual patience 触发时额外记录实际停止迭代",
        f"- 抽取实例：{', '.join(f'G{item}' for item in selected_ids)}",
        f"- 总耗时：{total_seconds:.3f} 秒",
        "",
        markdown_table(
            ["组别", "lr_x 模式", "配置 lr_x", "dual_init 模式", "配置 dual_init", "谱步长比例"],
            config_rows,
        ),
        "",
        "Max-Cut 在求解器内部表示为最小化负 cut。CSV 同时保存"
        " `best_objective`（越小越好）和 `best_cut=-best_objective`（越大越好）；"
        "本报告使用更直观的 `best_cut`。四组在每个实例上使用相同的随机初始种子。",
        "",
        "## 最终结果",
        "",
        markdown_table(
            ["组别", "平均最终 best_cut", "中位最终 best_cut", "平均增益 vs 对照", "胜/平/负 vs 对照"],
            final_rows,
        ),
        "",
        "## 收敛过程",
        "",
        markdown_table(
            ["迭代"] + [f"{config.label}平均 best_cut" for config in CONFIGURATIONS],
            trajectory_rows,
        ),
        "",
        "## 输出文件",
        "",
        "- `timeseries.csv`：逐实例、逐检查点的四组最优目标值和累计求解时间；提前停止的组留空后续检查点。",
        "- `summary.csv`：仅包含 `gset_id,n,m` 和四组最终 `cut`。",
        "- `report.md`：本实验报告。",
        "",
        "## 解读说明",
        "",
        "结果是固定求解器种子下的配对比较；胜/平/负按同一实例的最终 `best_cut` 计算。"
        "它用于隔离参数模式的影响，不估计多随机种子下的方差。",
        "",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    selected_ids = select_gset_ids(args)
    checkpoints = checkpoint_iterations(args.max_iters, args.sample_every)
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timeseries_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    experiment_started = time.perf_counter()

    print(
        f"Selected {len(selected_ids)} instances with selection_seed={args.selection_seed}: "
        + " ".join(f"G{item}" for item in selected_ids),
        flush=True,
    )
    for index, gset_id in enumerate(selected_ids, start=1):
        graph = parse_gset(str(gset_id))
        data = generate_max_cut(graph)
        results: Dict[str, Dict[str, Any]] = {}
        for config in CONFIGURATIONS:
            started = time.perf_counter()
            results[config.key] = run_configuration(
                data, config, args, checkpoints
            )
            final_cut = results[config.key]["samples"][max(results[config.key]["samples"])]
            final_cut = final_cut["best_cut"]
            print(
                f"[{index:02d}/{len(selected_ids):02d}] G{gset_id} "
                f"{config.key}: final_best_cut={final_cut:.6g} "
                f"elapsed={time.perf_counter() - started:.3f}s",
                flush=True,
            )

        timeseries_rows.extend(
            merge_instance_rows(
                gset_id,
                graph.number_of_nodes(),
                graph.number_of_edges(),
                checkpoints,
                results,
            )
        )
        summary_rows.append(
            make_summary_row(
                gset_id,
                graph.number_of_nodes(),
                graph.number_of_edges(),
                results,
            )
        )
        write_csv(output_dir / "timeseries.csv", timeseries_rows)
        write_csv(output_dir / "summary.csv", summary_rows)

    total_seconds = time.perf_counter() - experiment_started
    write_report(
        output_dir / "report.md",
        args,
        selected_ids,
        checkpoints,
        timeseries_rows,
        summary_rows,
        total_seconds,
    )
    print(f"Saved {output_dir / 'timeseries.csv'}")
    print(f"Saved {output_dir / 'summary.csv'}")
    print(f"Saved {output_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
