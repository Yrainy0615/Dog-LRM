"""v6 stage-2: cloth-sim draped-strand GT on the canonical D-SMAL (with body collision).
  blender --background --python preprocess/blender_drape_gt.py -- --out synth_fur/drape_raw.npz
Reads back BOTH the undraped strands (d0/length) and the gravity+collision draped strands
[M,K,3] plus emission roots, for fitting per-root (droop,gamma) supervision targets."""
import sys, numpy as np
try: import bpy
except ImportError: sys.exit("run inside blender")
A=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
import argparse; ap=argparse.ArgumentParser()
ap.add_argument('--inp',default='synth_fur/blender_input.npz'); ap.add_argument('--out',default='synth_fur/drape_raw.npz')
ap.add_argument('--count',type=int,default=20000); ap.add_argument('--K',type=int,default=11)
ap.add_argument('--frames',type=int,default=45)
ap.add_argument('--collision',action='store_true',help='body collision (causes lateral parting our planar droop cannot fit; default off)')
ap.add_argument('--bend',type=float,default=0.05,help='cloth bending stiffness (material knob; high=stiff fur)')
ap.add_argument('--pin',type=float,default=1.0)
ap.add_argument('--mass',type=float,default=0.3)
ap.add_argument('--hair_len',type=float,default=0.0,help='override uniform hair length (>max L_geo gives trim headroom); 0=use max L_geo')
a=ap.parse_args(A)
z=np.load(a.inp); v=z['verts'].astype('float32'); f=z['faces'].astype('int32'); L=z['L_geo'].astype('float32')
nz=L[L>1e-4]; lo=np.percentile(nz,8)
dens=np.clip(L/max(lo,1e-6),0,1); dens=dens*dens*(3-2*dens); dens[L<1e-4]=0
Ln=np.clip(L/L.max(),0,1)

def setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    me=bpy.data.meshes.new('d'); me.from_pydata(v.tolist(),[],f.tolist()); me.update()
    ob=bpy.data.objects.new('d',me); bpy.context.collection.objects.link(ob)
    bpy.context.view_layer.objects.active=ob
    for nm,w in [('dens',dens),('len',Ln)]:
        g=ob.vertex_groups.new(name=nm)
        for i,x in enumerate(w): g.add([i],float(x),'REPLACE')
    if a.collision:
        ob.modifiers.new('col','COLLISION')                   # body = collision surface (optional)
    ob.modifiers.new('h','PARTICLE_SYSTEM'); ps=ob.particle_systems[0]; s=ps.settings
    s.type='HAIR'; s.count=a.count; s.hair_length=a.hair_len or float(L.max()); s.hair_step=a.K-1
    s.use_advanced_hair=True; ps.vertex_group_density='dens'; ps.vertex_group_length='len'
    return ob,ps,s

def readback(ob):
    deps=bpy.context.evaluated_depsgraph_get(); oe=ob.evaluated_get(deps)
    parts=oe.particle_systems[0].particles
    return np.array([[list(k.co) for k in p.hair_keys] for p in parts],dtype='float32')

ob,ps,s=setup(); S0=readback(ob)                              # undraped (d0/length)
ps.use_hair_dynamics=True
cs=ps.cloth.settings; cs.quality=5; cs.mass=a.mass; cs.bending_stiffness=a.bend; cs.pin_stiffness=a.pin
try: ps.cloth.collision_settings.use_collision=True
except Exception as e: print('collision settings:',e)
sc=bpy.context.scene; sc.frame_start=1; sc.frame_end=a.frames
for fr in range(1,a.frames+1): sc.frame_set(fr)
S=readback(ob); roots=S[:,0].copy()
print('[drape] strands',S.shape,'undraped',S0.shape)
np.savez(a.out, drape=S, undraped=S0, roots=roots)
print('[drape] saved',a.out)
