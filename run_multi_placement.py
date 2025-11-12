
import argparse, subprocess, sys, pathlib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", required=True, help="Folder of JSON cells")
    ap.add_argument("--timesteps", type=int, default=100000)
    ap.add_argument("--output-dir", default="./out_multi")
    ap.add_argument("--python", default=sys.executable, help="Python interpreter to use")
    ap.add_argument("--script", default="transistor_placement.py", help="Path to transistor_placement.py")
    args = ap.parse_args()

    cmd = [
        args.python, args.script,
        "--env-dir", args.env_dir,
        "--timesteps", str(args.timesteps),
        "--output-dir", args.output_dir,
    ]
    print(">>", " ".join(cmd))
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
