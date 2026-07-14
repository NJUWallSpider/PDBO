import argparse
import os
import sys
import time

import numpy as np

from pdbo import (
    PDBOSolver,
    evaluate_labs_bits,
    generate_labs,
    generate_max_cut,
    generate_mis,
    parse_gset,
    random_graph,
    refine_binary_incumbent,
)
from utils import fix_seed


def build_parser():
    parser = argparse.ArgumentParser(description="Run PDBO on QUBO problems.")
    parser.add_argument("--task", choices=["mis", "mc", "labs"], default="mis")
    parser.add_argument("--graph", choices=["reg", "Gset"], default="reg")
    parser.add_argument("--Gset_id", type=int, default=66)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--labs_n", type=int, default=47)
    parser.add_argument("--labs_penalty", type=float, default=10000)
    parser.add_argument("--d", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--lr_x", type=float, default=0.02)
    parser.add_argument("--lr_y", type=float, default=0.02)
    parser.add_argument("--delta", type=float, default=1e-8)
    parser.add_argument("--dual_init", type=float, default=5)
    parser.add_argument("--primal_init", choices=["uniform", "half", "binary"], default="uniform")
    parser.add_argument("--restart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--penalty", type=float, default=4)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timelimit", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--check_every", type=int, default=100)
    parser.add_argument("--quadratic_backend", choices=["edge", "sparse"], default="sparse")
    parser.add_argument("--g_type", choices=["quadratic", "absolute"], default="quadratic")
    parser.add_argument("--rounding_samples", type=int, default=0)
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refine_max_passes", type=int, default=None)
    return parser


def result_path(args):
    solver_dir = "pdbo" if args.g_type == "quadratic" else f"pdbo_{args.g_type}"
    if args.task == "labs":
        result_dir = f"./result/{solver_dir}/labs/"
        path = f"n={args.labs_n}_p={args.labs_penalty}_s={args.seed}.txt"
    else:
        result_dir = f"./result/{solver_dir}/{args.task}/{args.graph}/"
        if args.graph == "Gset":
            path = f"G{args.Gset_id}.txt"
        else:
            path = f"n={args.n}d={args.d}s={args.seed}.txt"

    os.makedirs(result_dir, exist_ok=True)
    return os.path.join(result_dir, path)


def load_problem(args):
    if args.task == "labs":
        return None, generate_labs(args.labs_n, args.labs_penalty)

    if args.graph == "Gset":
        graph = parse_gset(f"{args.Gset_id}")
    else:
        graph = random_graph(n=args.n, d=args.d, seed=args.seed)

    if args.task == "mis":
        return graph, generate_mis(graph, args.penalty)
    return graph, generate_max_cut(graph)


def build_solver(args, data):
    common = {
        "batch_size": args.batch,
        "max_iters": args.max_iters,
        "primal_lr": args.lr_x,
        "dual_lr": args.lr_y,
        "dual_init": args.dual_init,
        "verbose": args.verbose,
        "seed": args.seed,
    }

    return PDBOSolver(
        n_vars=data["num_vars"],
        Q_indices=data["Q_indices"],
        Q_values=data["Q_values"],
        c=data["c"],
        timelimit=args.timelimit,
        primal_init=args.primal_init,
        restart=args.restart,
        tolerance=args.delta,
        patience=args.patience,
        min_delta=args.min_delta,
        check_every=args.check_every,
        quadratic_backend=args.quadratic_backend,
        rounding_samples=args.rounding_samples,
        g_type=args.g_type,
        **common,
    )


def write_result(path, args, data, solver, solving_time, refinement=None):
    objective_offset = data.get("objective_offset", 0.0)
    incumbents = [value + objective_offset for value in solver.objVal_record]
    with open(path, "w") as f:
        f.write("incumbents:" + str(incumbents) + "\n")
        f.write("timing:" + str(solver.timing_record) + "\n")
        f.write("total time:" + str(solving_time) + "\n")
        f.write("stop reason:" + str(solver.stop_reason) + "\n")
        f.write("perturbations:" + str(getattr(solver, "perturbation_count", 0)) + "\n")
        if refinement is not None:
            f.write("refined objective:" + str(refinement.objective) + "\n")
            f.write("refinement steps:" + str(refinement.steps) + "\n")
            f.write("refinement time:" + str(refinement.seconds) + "\n")
            f.write("refined solution:" + "".join(str(int(x)) for x in refinement.bits) + "\n")
        if args.task == "labs":
            labs_bits = np.array(solver.incumbent[: args.labs_n], dtype=np.int32)
            f.write("labs energy:" + str(evaluate_labs_bits(labs_bits)) + "\n")
            f.write("solution:" + "".join(str(int(x)) for x in labs_bits) + "\n")


def main():
    args = build_parser().parse_args()
    fix_seed(args.seed)
    out_path = result_path(args)

    if os.path.exists(out_path) and args.save:
        print("PASS", out_path)
        return 0

    _, data = load_problem(args)
    solver = build_solver(args, data)

    start = time.perf_counter()
    solver.optimize()
    solving_time = time.perf_counter() - start

    refinement = None
    if args.refine:
        refinement = refine_binary_incumbent(
            args.task,
            np.asarray(solver.incumbent, dtype=np.int32),
            data,
            max_passes=args.refine_max_passes,
        )

    if args.save:
        write_result(out_path, args, data, solver, solving_time, refinement)

    message = f"best={float(solver.objVal)} time={solving_time:.6f}s stop={getattr(solver, 'stop_reason', None)}"
    if refinement is not None:
        message += f" refined_best={refinement.objective} refine_time={refinement.seconds:.6f}s"
    print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
