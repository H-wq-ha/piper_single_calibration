import argparse
from pathlib import Path


def main(argv=None):
    from .opencv_backend import DEFAULT_METHOD, METHODS
    parser = argparse.ArgumentParser(description="Recorded Piper/RealSense hand-eye calibration")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("solve", help="Solve and evaluate a recorded session")
    p.add_argument("session")
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=[*METHODS, "auto"], default=DEFAULT_METHOD)
    p.add_argument("--auto-split", action="store_true", help="Override recorded roles with contiguous 60/20/20 split")
    p = sub.add_parser("capture", help="Record camera + robot feedback; no motion commands")
    p.add_argument("--mode", choices=["eye_to_hand", "eye_in_hand"], default="eye_to_hand")
    p.add_argument("--intrinsics", default="data/result/calibration.yaml")
    p.add_argument("--output", default="data/sessions")
    p.add_argument("--can-port", default="can0")
    p.add_argument("--enable-arm", action="store_true")
    p.add_argument("--stationary-window", type=float, default=.5)
    p.add_argument("--stationary-mm", type=float, default=.5)
    p.add_argument("--stationary-deg", type=float, default=.15)
    p.add_argument("--max-rmse-px", type=float, default=.8)
    p.add_argument("--min-corners", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "solve":
        from .storage import load_session
        from .pipeline import run_evaluation, split_samples
        meta, samples = load_session(args.session)
        if args.auto_split:
            for role, subset in split_samples(samples).items():
                for sample in subset:
                    sample["role"] = role
        elif not {"train", "development", "test"}.issubset({s.get("role") for s in samples}):
            parser.error("Record S/V/T samples, or explicitly use --auto-split")
        result = run_evaluation(meta, samples, args.output, method=args.method)
        print("Selected:", result["selected_method"], "results:", Path(args.output).resolve())
    elif args.command == "capture":
        from .capture import capture
        capture(args)


if __name__ == "__main__":
    main()
