#!/usr/bin/env python3
"""Shard usable COLMAP dog scenes across GPUs and run the full preprocess pipeline.

Usable scene = <root>/<dog>/colmap with sparse/0/{cameras,images,points3D}.txt and
>= --min_img images. Each GPU gets one worker process that loops its shard, running the
six stages per scene (model reloads per scene are negligible vs the per-scene fit/mask
compute). A stage failure logs and skips the rest of that scene; other scenes continue.
"""
import argparse
import glob
import multiprocessing as mp
import os
import subprocess
import sys

STAGES = [  # (script, extra args after --scene_dir)
    ("colmap_to_cameras.py", []),
    ("extract_masks.py", ["--device", "cuda"]),
    ("crop_dogs.py", []),
    ("barc_infer.py", ["--device", "cuda"]),
    ("fit_smal.py", ["--init_barc", "--device", "cuda"]),
    ("build_manifest.py", []),
]


def usable(colmap_dir, min_img):
    sp = os.path.join(colmap_dir, "sparse", "0")
    if not all(os.path.exists(os.path.join(sp, f))
               for f in ("cameras.txt", "images.txt", "points3D.txt")):
        return False
    return len(glob.glob(os.path.join(colmap_dir, "images", "*.jpg"))) >= min_img


def worker(scenes, gpu, py, pre_dir, logdir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    with open(os.path.join(logdir, f"gpu{gpu}.log"), "w") as log:
        for i, sc in enumerate(scenes):
            name = os.path.basename(os.path.dirname(sc))
            for script, extra in STAGES:
                cmd = [py, os.path.join(pre_dir, script), "--scene_dir", sc] + extra
                log.write(f"\n=== [{i+1}/{len(scenes)} {name}] {script} (gpu{gpu}) ===\n")
                log.flush()
                rc = subprocess.run(cmd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    log.write(f"[FAIL rc={rc}] {name}/{script} -> skip rest of scene\n")
                    log.flush()
                    break
        log.write("\nWORKER_DONE\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="parent of <dog>/colmap scenes")
    ap.add_argument("--gpus", default="1,2,3,4,5")
    ap.add_argument("--min_img", type=int, default=50)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--logdir", default="exps/preprocess_logs")
    ap.add_argument("--stages", default=None,
                    help="comma list of stage script stems to run (default: all). "
                         "e.g. barc_infer,fit_smal to re-run only those.")
    args = ap.parse_args()

    global STAGES
    if args.stages:
        want = set(args.stages.split(","))
        STAGES = [(s, e) for (s, e) in STAGES if s[:-3] in want]
        if not STAGES:
            sys.exit(f"no stages matched {args.stages}")

    gpus = [int(g) for g in args.gpus.split(",")]
    colmaps = sorted(glob.glob(os.path.join(args.root, "*", "colmap")))
    scenes = [c for c in colmaps if usable(c, args.min_img)]
    os.makedirs(args.logdir, exist_ok=True)
    shards = {g: scenes[i::len(gpus)] for i, g in enumerate(gpus)}
    print(f"{len(scenes)}/{len(colmaps)} usable scenes across {len(gpus)} GPUs", flush=True)
    for g in gpus:
        print(f"  gpu{g}: {len(shards[g])} scenes", flush=True)
    with open(os.path.join(args.logdir, "scenes.txt"), "w") as f:
        f.write("\n".join(scenes) + "\n")

    pre_dir = os.path.dirname(os.path.abspath(__file__))
    procs = [mp.Process(target=worker, args=(shards[g], g, args.py, pre_dir, args.logdir))
             for g in gpus]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("ALL WORKERS DONE", flush=True)


if __name__ == "__main__":
    main()
