# /mnt/data/cli.py
import argparse, json, yaml
from render import run_session

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True)
    ap.add_argument("--bpm", type=float)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--dur", type=float, default=84.0)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("logging", {}).get("level", "INFO") == "INFO":
        print(f"[Σ] launching style={args.style} bpm={args.bpm or 'default'} seed={args.seed} dur={args.dur}s")

    plan = {"style": args.style, "bpm": args.bpm, "seed": args.seed, "duration_s": args.dur}
    rep = run_session(plan, args.config)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()