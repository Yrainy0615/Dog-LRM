#!/usr/bin/env python3
"""DiffLocks-style paired (image, 3D-strand) dataset on the D-SMAL template.

v2: Blender 4.2 CURVES hair system. ALL strand geometry is synthesized in numpy
(comb flow field + curl + droop + clump + frizz -- the FurV6Flow groom recipe), the
Curves object is only the render carrier, so exported GT == rendered geometry EXACTLY
(the old particle-hair path had straight normal-direction parents: kink/clump lived
only in render-time children, and hair_keys writes don't survive depsgraph re-eval).

  .blender/blender-4.2.21-linux-x64/blender --background --python preprocess/blender_fur_dataset.py -- \
      --inp synth_fur/blender_input.npz --palette synth_fur/coat_palette.npz --out synth_fur/dataset [--smoke]
"""
import sys, os, math
import numpy as np
try:
    import bpy, mathutils
except ImportError:
    sys.exit("run inside Blender")


def args_():
    import argparse
    a = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="synth_fur/blender_input.npz")
    ap.add_argument("--palette", default="synth_fur/coat_palette.npz")
    ap.add_argument("--out", default="synth_fur/dataset")
    ap.add_argument("--views", type=int, default=8)
    ap.add_argument("--res", type=int, default=448)
    ap.add_argument("--count", type=int, default=250000, help="rendered strands per groom")
    ap.add_argument("--gt_count", type=int, default=40000, help="strands saved as GT")
    ap.add_argument("--K", type=int, default=12, help="points per strand")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--per_family", type=int, default=3, help="param jitters per (family,colour)")
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args(a)


# family -> (BODY hair length in BU (1 BU ~ 35cm), curl_amp_frac, curl_freq); jittered per groom
STYLES = {"short": (0.04, 0.0, 0.0), "long": (0.12, 0.0, 0.0),
          "curly": (0.085, 0.55, 3.0), "wavy": (0.10, 0.25, 1.6)}


def jitter_style(family, rng):
    length, kamp, kfreq = STYLES[family]
    length = length * rng.uniform(0.7, 1.25)
    if kamp > 0:
        kamp = min(1.0, kamp * rng.uniform(0.7, 1.3)); kfreq = kfreq * rng.uniform(0.7, 1.3)
    elif rng.uniform() < 0.3:                                # some straight coats get mild waviness
        kamp = rng.uniform(0.05, 0.15); kfreq = rng.uniform(0.8, 1.6)
    return length, kamp, kfreq


def density_field(L_geo):
    nz = L_geo[L_geo > 1e-4]; lo = np.percentile(nz, 8) if nz.size else 0.0
    d = np.clip(L_geo / max(lo, 1e-6), 0, 1); d = d*d*(3-2*d); d[L_geo < 1e-4] = 0.0
    return d.astype(np.float32)


def srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c/12.92, ((c+0.055)/1.055)**2.4)


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]]-verts[faces[:, 0]], verts[faces[:, 2]]-verts[faces[:, 0]])
    n = np.zeros_like(verts)
    np.add.at(n, faces.ravel(), np.repeat(fn, 3, 0))
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


def chunked_nn(a, b, chunk=8192):
    out = np.empty(len(a), np.int64)
    for i in range(0, len(a), chunk):
        d = ((a[i:i+chunk, None] - b[None]) ** 2).sum(-1)
        out[i:i+chunk] = d.argmin(1)
    return out


def grow_strands(verts, faces, vn, density, lengthw, params, K, n, rng):
    """full numpy groom: roots + comb flow + droop + curl + clump + frizz -> [n,K,3]"""
    length, kamp, kfreq = params
    tri = verts[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
    fd = density[faces].mean(1) * area
    fi = rng.choice(len(faces), n, p=fd/fd.sum())
    r = rng.random((n, 2)); su = np.sqrt(r[:, 0])
    b = np.stack([1-su, su*(1-r[:, 1]), su*r[:, 1]], 1)
    roots = (tri[fi] * b[..., None]).sum(1)
    nrm = vn[faces[fi]]; nrm = (nrm * b[..., None]).sum(1)
    nrm /= np.clip(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9, None)
    lw = (lengthw[faces[fi]] * b).sum(1)
    ref = np.percentile(lengthw[lengthw > 0.05], 70)                           # body-level weight -> 1.0
    L = length * np.clip(lw / max(ref, 1e-6), 0.08, 1.6) * rng.uniform(0.85, 1.15, n)

    # comb direction: head->tail + down flow projected on surface (v6-flow), azimuth jitter
    flow = np.array([-1.0, 0.0, -0.6]); flow /= np.linalg.norm(flow)
    down = np.array([0.0, 0.0, -1.0])
    t = flow[None] - (nrm @ flow)[:, None] * nrm
    tl = np.linalg.norm(t, axis=1, keepdims=True)
    t = np.where(tl > 1e-6, t / np.clip(tl, 1e-9, None), np.array([[-1.0, 0, 0]]))
    az = rng.normal(0, 0.15, n)
    bvec = np.cross(nrm, t)
    t = t * np.cos(az)[:, None] + bvec * np.sin(az)[:, None]
    tmix = np.clip(rng.normal(0.65, 0.08, n), 0.45, 0.85)[:, None]
    d = tmix * t + (1 - tmix) * nrm
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    # integrate polyline: droop bends toward gravity along the strand
    frac = (np.arange(1, K) / (K - 1))[None, :, None]                          # [1,K-1,1]
    droop = np.clip(rng.normal(0.45, 0.1, n), 0.2, 0.8)
    droop = (droop * np.clip(L / 0.08, 0.6, 2.4))[:, None, None]               # long fur HANGS, short stays lofty
    dirs = d[:, None] + droop * frac**2 * down[None, None]
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    seg = (L / (K - 1))[:, None, None]
    pts = np.concatenate([roots[:, None], roots[:, None] + np.cumsum(dirs * seg, 1)], 1)   # [n,K,3]

    # curl: helix around the comb direction, amplitude grows toward the tip
    if kamp > 0:
        e1 = np.cross(d, down[None]); e1 /= np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9, None)
        e2 = np.cross(d, e1)
        ph = rng.uniform(0, 2*np.pi, n)[:, None]
        th = 2*np.pi*kfreq*np.arange(K)[None, :]/(K-1) + ph
        amp = (kamp * L * 0.35)[:, None] * (np.arange(K)[None, :]/(K-1))
        pts += (np.cos(th)*amp)[..., None]*e1[:, None] + (np.sin(th)*amp)[..., None]*e2[:, None]

    # clump: pull toward the nearest guide strand, stronger toward the tip (fur tufts)
    gsel = rng.choice(n, max(n//15, 1), replace=False)
    gid = gsel[chunked_nn(roots, roots[gsel])]
    cf = np.clip(rng.normal(0.35, 0.08, n), 0.1, 0.6)[:, None, None]
    w = cf * (np.arange(K)[None, :, None]/(K-1))**1.2
    pts = pts + w * (pts[gid] - pts)

    # frizz: small smooth per-strand wobble (absolute cap so long coats don't go wild)
    fz = np.minimum(0.06 * L, 0.006)[:, None]
    ph1, ph2 = rng.uniform(0, 2*np.pi, (2, n, 1))
    wob = np.sin(np.arange(K)[None]*1.7 + ph1)*fz, np.cos(np.arange(K)[None]*2.3 + ph2)*fz
    e1 = np.cross(d, down[None]); e1 /= np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9, None)
    e2 = np.cross(d, e1)
    pts += wob[0][..., None]*e1[:, None] + wob[1][..., None]*e2[:, None]
    pts[:, 0] = roots                                                          # roots stay on the surface
    return pts.astype(np.float32)


def make_body(verts, faces, rgb_lin):
    for o in list(bpy.data.objects):
        if o.type in ('MESH', 'CURVES'): bpy.data.objects.remove(o, do_unlink=True)
    mesh = bpy.data.meshes.new("dsmal"); mesh.from_pydata(verts.tolist(), [], faces.tolist()); mesh.update()
    for p in mesh.polygons: p.use_smooth = True
    obj = bpy.data.objects.new("dsmal", mesh); bpy.context.collection.objects.link(obj)
    m = bpy.data.materials.new("skin"); m.use_nodes = True                     # darker undercoat skin
    m.node_tree.nodes.get("Principled BSDF").inputs["Base Color"].default_value = (*(rgb_lin*0.7), 1)
    obj.data.materials.append(m)
    return obj


def hair_material(rgb_lin):
    m = bpy.data.materials.new("hair"); m.use_nodes = True; nt = m.node_tree
    for n in list(nt.nodes): nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    h = nt.nodes.new('ShaderNodeBsdfHairPrincipled')
    try: h.parametrization = 'COLOR'
    except Exception: pass
    for k, v in [("Roughness", 0.35), ("Radial Roughness", 0.25),
                 ("Random Color", 0.2), ("Random Roughness", 0.35), ("Coat", 0.15)]:
        if k in h.inputs:
            try: h.inputs[k].default_value = v
            except Exception: pass
    tex = nt.nodes.new('ShaderNodeTexNoise'); tex.inputs["Scale"].default_value = 2.0
    co = nt.nodes.new('ShaderNodeTexCoord')
    mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'
    mix.inputs["A"].default_value = (*rgb_lin, 1)
    mix.inputs["B"].default_value = (*(rgb_lin * 0.55), 1)
    nt.links.new(co.outputs["Object"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Fac"], mix.inputs["Factor"])
    if "Color" in h.inputs:
        nt.links.new(mix.outputs["Result"], h.inputs["Color"])
    nt.links.new(h.outputs[0], out.inputs["Surface"])
    return m


def make_fur(strands, rgb_lin, diag):
    n, K = strands.shape[:2]
    cu = bpy.data.hair_curves.new("fur")
    cu.add_curves([K] * n)
    cu.points.foreach_set("position", strands.reshape(-1).astype(np.float32))
    rr, tr = 0.0014 * diag / 2.4, 0.0004 * diag / 2.4                          # ~tutorial 0.001-0.003 BU widths
    rad = np.repeat(np.linspace(rr, tr, K, dtype=np.float32)[None], n, 0)
    attr = cu.attributes.new("radius", "FLOAT", "POINT")
    attr.data.foreach_set("value", rad.reshape(-1))
    cu.materials.append(hair_material(rgb_lin))
    obj = bpy.data.objects.new("fur", cu); bpy.context.collection.objects.link(obj)
    return obj


def setup_render(res, samples):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'; sc.cycles.samples = samples
    try: sc.cycles.device = 'GPU'
    except Exception: pass
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = 'PNG'
    w = bpy.data.worlds.new("w"); w.use_nodes = True
    bg = w.node_tree.nodes["Background"]; bg.inputs[0].default_value = (0.82, 0.82, 0.83, 1); bg.inputs[1].default_value = 0.9
    sc.world = w
    sc.view_settings.view_transform = 'Standard'
    lights = []
    for ang, en in [((50, 0, 40), 1.6), ((60, 0, -120), 1.0), ((30, 0, 180), 0.8)]:
        ld = bpy.data.lights.new("L", 'SUN'); ld.energy = en
        lo = bpy.data.objects.new("L", ld); bpy.context.collection.objects.link(lo)
        lo.rotation_euler = tuple(math.radians(x) for x in ang)
        lights.append((ld, en))
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); bpy.context.collection.objects.link(cam); sc.camera = cam
    return cam, lights, bg


def jitter_scene(lights, bg, rng):
    for ld, base in lights:
        ld.energy = base * rng.uniform(0.7, 1.3)
    tint = 0.80 + rng.uniform(-0.04, 0.06)
    bg.inputs[0].default_value = (tint, tint, tint + rng.uniform(-0.01, 0.02), 1)
    bg.inputs[1].default_value = rng.uniform(0.75, 1.0)


def look_at(cam, eye, target):
    cam.location = eye
    cam.rotation_euler = (mathutils.Vector(target)-mathutils.Vector(eye)).to_track_quat('-Z', 'Y').to_euler()


def cam_K(cam, res):
    bpy.context.view_layer.update()
    fx = cam.data.lens / cam.data.sensor_width * res
    return np.array([[fx, 0, res/2], [0, fx, res/2], [0, 0, 1]], np.float32), np.array(cam.matrix_world, np.float32)


def main():
    A = args_(); z = np.load(A.inp); pal = np.load(A.palette)
    verts, faces = z["verts"].astype(np.float32), z["faces"].astype(np.int64)
    vn = vertex_normals(verts, faces)
    density = density_field(z["L_geo"])
    lengthw = (z["L_geo"] / max(float(z["L_geo"].max()), 1e-6)).astype(np.float32)
    lengthw *= (1 - 0.85 * np.clip(z["w_face"], 0, 1))                 # muzzle/face fur is VERY short
    lengthw *= (1 - 0.5 * np.clip(z["w_ear"], 0, 1))
    lsf = min(1.0 / max(float(np.percentile(lengthw[lengthw > 0.05], 70)), 0.3), 1.2)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    os.makedirs(A.out, exist_ok=True)
    ctr = (verts.max(0)+verts.min(0))/2
    rad = diag*1.5                                                     # 50mm lens needs >=1.4xdiag to fit
    cam, lights, bgnode = setup_render(A.res, A.samples)

    rng = np.random.default_rng(0)
    grooms = []
    for style in (["short", "long", "curly"] if A.smoke else STYLES):
        cols = pal[style] if style in pal.files else np.array([[0.6, 0.5, 0.4]])
        for ci, c in enumerate(cols[:1] if A.smoke else cols):
            for ji in range(1 if A.smoke else A.per_family):
                grooms.append((style, ci, ji, jitter_style(style, rng), srgb_to_lin(c)))
    nview = 4 if A.smoke else A.views
    index = []
    for gi, (style, ci, ji, params, rgb_lin) in enumerate(grooms):
        grng = np.random.default_rng(100 + gi)
        strands = grow_strands(verts, faces, vn, density, lengthw,
                               (params[0]*lsf, params[1], params[2]), A.K, A.count, grng)
        make_body(verts, faces, rgb_lin)
        make_fur(strands, rgb_lin, diag)
        jitter_scene(lights, bgnode, rng)
        name = f"{style}_{ci}_{ji}"; cams = []
        for vi in range(nview):
            az = math.pi/2 + 2*math.pi*vi/nview; el = math.radians(10 + 8*math.sin(vi))
            eye = (ctr[0]+rad*math.cos(el)*math.cos(az), ctr[1]+rad*math.cos(el)*math.sin(az), ctr[2]+rad*math.sin(el))
            look_at(cam, eye, tuple(ctr)); K, c2w = cam_K(cam, A.res)
            fp = os.path.join(A.out, f"{name}_v{vi}.png"); bpy.context.scene.render.filepath = fp
            bpy.ops.render.render(write_still=True)
            cams.append((K, c2w, os.path.basename(fp)))
        keep = grng.choice(A.count, min(A.gt_count, A.count), replace=False)
        gt = strands[keep]
        np.savez(os.path.join(A.out, f"{name}.npz"), strands=gt, roots=gt[:, 0].copy(), style=style,
                 style_params=np.array(params, np.float32),
                 rgb=(rgb_lin).astype(np.float32), Ks=np.stack([c[0] for c in cams]),
                 c2ws=np.stack([c[1] for c in cams]), imgs=np.array([c[2] for c in cams]))
        index.append(name)
        print(f"[data] {name}: {gt.shape[0]}x{gt.shape[1]} strand GT ({A.count} rendered), {nview} views", flush=True)
    with open(os.path.join(A.out, "index.txt"), "w") as f: f.write("\n".join(index))
    print(f"[data] DONE {len(index)} grooms -> {A.out}", flush=True)


main()
