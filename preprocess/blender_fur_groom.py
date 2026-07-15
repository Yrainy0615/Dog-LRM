#!/usr/bin/env python3
"""v6 stage-1: headless Blender groom on the canonical D-SMAL template.

Runs under Blender's bundled python:
  .blender/blender-4.2*/blender --background --python preprocess/blender_fur_groom.py -- \
      --inp synth_fur/blender_input.npz --out synth_fur/canonical_density.npz --grow_hair

Authors a per-vertex fur DENSITY field on the mean D-SMAL template from the measured
length profile (L_geo): low where every dog measures ~no fur (eyes/nose = true bald),
scaled by length elsewhere, with procedural value noise for natural part-boundary
falloff. The field is realized as a particle-hair system (density-weighted) to confirm
Blender places strands where the field says, then the field (and, with --grow_hair, the
evaluated hair-root positions) is exported. Downstream `export_fur_gt.py` samples this
field at the fur model's roots to make per-root density GT for the v6 density head.

Pure stdlib + numpy (Blender ships numpy); no scene assets, fully reproducible headless.
"""
import sys

import numpy as np

try:
    import bpy
except ImportError:
    sys.exit("must run inside Blender: blender --background --python this.py -- ...")


def argv_after_ddash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="synth_fur/blender_input.npz")
    ap.add_argument("--out", default="synth_fur/canonical_density.npz")
    ap.add_argument("--count", type=int, default=26000, help="particle-hair count (validation)")
    ap.add_argument("--noise", type=float, default=0.10, help="procedural density noise amplitude")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grow_hair", action="store_true", help="grow + read back hair roots (validation)")
    return ap.parse_args(argv_after_ddash())


def density_from_length(L_geo, w_ear, noise_amp, seed):
    """Per-vertex fur PRESENCE gate in [0,1]: ~1 wherever fur is measured, smoothly ->0 only
    on genuinely short/bald regions (eyes/nose, where every dog measures ~no fur). This is a
    baldness gate (multiplies opacity), NOT a length scale -- it must stay ~1 over the coat
    so it doesn't dim well-furred regions. Length-driven, not part-driven, so it does NOT
    zero ears (real ear fur exists) -- avoids the v3 'blanket ear cut kills fluffy ears' bug."""
    nz = L_geo[L_geo > 1e-4]
    lo = np.percentile(nz, 8) if nz.size else 0.0                   # below ~p8 of fur -> fades out
    d = np.clip(L_geo / max(lo, 1e-6), 0.0, 1.0)                    # 1 over most of the coat
    d = d * d * (3 - 2 * d)                                          # smoothstep transition
    rng = np.random.default_rng(seed)
    d = np.clip(d + noise_amp * (rng.random(d.shape) - 0.5), 0.0, 1.0)
    d[L_geo < 1e-4] = 0.0                                            # hard bald where never measured
    return d.astype(np.float32)


def main():
    args = parse_args()
    z = np.load(args.inp)
    verts, faces = z["verts"].astype(np.float32), z["faces"].astype(np.int32)
    L_geo, w_ear = z["L_geo"], z["w_ear"]
    density = density_from_length(L_geo, w_ear, args.noise, args.seed)
    print(f"[groom] verts {verts.shape} faces {faces.shape} | density "
          f"mean {density.mean():.3f} zeros {int((density < 1e-3).sum())}", flush=True)

    # fresh scene -> mesh object
    bpy.ops.wm.read_factory_settings(use_empty=True)
    mesh = bpy.data.meshes.new("dsmal")
    mesh.from_pydata(verts.tolist(), [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new("dsmal", mesh)
    bpy.context.collection.objects.link(obj)

    # density vertex group (drives hair placement, and is the exported GT)
    vg = obj.vertex_groups.new(name="density")
    for i, w in enumerate(density):
        vg.add([i], float(w), 'REPLACE')

    hair_roots = None
    if args.grow_hair:
        bpy.context.view_layer.objects.active = obj
        psm = obj.modifiers.new("hair", 'PARTICLE_SYSTEM')
        psys = obj.particle_systems[0]
        ps = psys.settings
        ps.type = 'HAIR'
        ps.count = args.count
        ps.hair_length = 0.05
        ps.use_advanced_hair = True
        psys.vertex_group_density = "density"                       # density field drives placement
        psys.vertex_group_length = "density"
        deps = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(deps)
        pe = obj_eval.particle_systems[0].particles
        n = len(pe)
        if n:
            co = np.empty(n * 3, dtype=np.float32)
            pe.foreach_get("location", co)                          # emission roots
            hair_roots = co.reshape(n, 3)
        print(f"[groom] grew {n} hairs (density-weighted placement)", flush=True)

    out = dict(density=density, verts=verts, faces=faces,
               L_geo=L_geo.astype(np.float32), w_ear=w_ear.astype(np.float32))
    if hair_roots is not None:
        out["hair_roots"] = hair_roots
    np.savez(args.out, **out)
    print(f"[groom] saved -> {args.out}", flush=True)


main()
