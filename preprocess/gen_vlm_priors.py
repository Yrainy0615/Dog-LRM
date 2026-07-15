"""P2: expand my per-dog VLM annotations (from exps/_p2_sheet_*.png contact sheets,
annotated 2026-06-12) into per-dog prior jsons compatible with vlm_prior_bear.json.

Compact schema per dog: (bbox_diag_cm, body_len_cm, curl_class, overrides{}).
Region->joint expansion mirrors the bear annotation layout; region defaults are
ratios of body length. Stiffness comes from the curl class (for later dynamics).
"""
import json
import os

# (diag_cm, body_cm, curl, overrides)
ANN = {
    "00003-nara":      (75, 2.5, "short_smooth", {}),
    "00010-hanabi":    (55, 6.0, "wavy", {}),
    "00013-kota":      (50, 4.0, "curly", {}),
    "00027-kuma":      (45, 2.5, "short_smooth", {}),
    "00029-oto":       (48, 4.5, "wavy", {}),
    "00031-itsuki":    (85, 6.0, "long_straight", {"tail": 9.0}),
    "00035-ten":       (40, 1.2, "short_smooth", {}),
    "00037-runon":     (55, 4.0, "wire", {"face": 3.0, "muzzle": 3.0, "ears": 1.5}),
    "00039-reon":      (45, 4.0, "long_straight", {"ears": 6.0, "tail": 7.0}),
    "00041-niko":      (42, 5.0, "double_coat", {"tail": 7.0}),
    "00048-tenten":    (45, 6.0, "double_coat", {"tail": 8.0}),
    "00052-kotaro":    (42, 4.0, "double_coat", {}),
    "00062-bear":      (70, 5.5, "wavy", {"face": 2.0, "muzzle": 1.5, "ears": 5.0, "tail": 6.5}),
    "00065-paul":      (65, 5.0, "curly", {"note": "wears clothes on torso"}),
    "00077-kinu":      (50, 5.0, "double_coat", {"tail": 7.0}),
    "00081-yamato":    (55, 4.5, "long_straight", {}),
    "00082-yamato":    (60, 5.0, "long_straight", {}),
    "00085-kotori":     (65, 2.5, "short_smooth", {}),
    "00087-tiara":     (45, 4.0, "wavy", {}),
    "00089-huu":       (50, 6.0, "long_straight", {}),
    "00095-toko":      (55, 4.5, "long_straight", {}),
    "00097-rinto":     (50, 4.0, "long_straight", {}),
    "00100-kokoa":     (52, 5.0, "curly", {}),
    "00104-milk":      (48, 5.0, "curly", {}),
    "00105-tsubu":     (40, 4.0, "long_straight", {}),
    "00106-tsubu":     (42, 4.0, "long_straight", {}),
    "00107-kotsubu":    (38, 1.5, "short_smooth", {}),
    "00110-choko":     (42, 5.0, "double_coat", {}),
    "00117-kanon":     (55, 6.0, "wavy", {}),
    "00126-sara":       (55, 3.0, "short_smooth", {}),
    "00127-pon":       (45, 5.5, "double_coat", {}),
    "00130-komugi":    (50, 4.0, "wire", {}),
    "00134-nagomi":    (45, 5.0, "long_straight", {}),
    "00139-ten":       (55, 4.0, "long_straight", {}),
    "00140-uru":       (55, 2.0, "short_smooth", {}),
    "00146-sora":       (48, 4.5, "curly", {}),
    "00148-uta":       (48, 5.0, "wavy", {}),
    "00153-tekuno":    (45, 4.0, "long_straight", {}),
    "00168-rubi":      (55, 6.0, "double_coat", {}),
    "00169-tono":      (80, 7.0, "double_coat", {}),
    "00174-tete":      (48, 4.0, "curly", {}),
    "00179-runrun":    (52, 4.5, "curly", {}),
    "00182-ann":       (48, 4.0, "curly", {}),
    "00185-ragua":     (75, 7.0, "wavy", {}),
    "00188-willy":     (80, 1.0, "short_smooth", {}),
    "00190-kohaku":    (40, 3.0, "wavy", {}),
    "00195-kurizo":    (70, 8.0, "double_coat", {}),
    "00198-suzu":      (40, 4.0, "long_straight", {}),
    "00200-uni":       (55, 5.0, "curly", {}),
    "00201-kintarou":  (45, 6.0, "long_straight", {}),
    "00202-momotarou": (45, 5.0, "long_straight", {}),
    "00203-chachamaru":(45, 4.0, "long_straight", {"ears": 6.0}),
    "00205-akubi":     (60, 1.0, "short_smooth", {}),
    "00208-rinda":     (50, 1.2, "short_smooth", {}),
    "00210-hikaru":    (52, 1.2, "short_smooth", {}),
    "00211-pon":       (75, 1.5, "short_smooth", {}),
    "00215-teo":       (60, 6.0, "wavy", {"ears": 8.0}),
    "00222-hal":       (50, 5.0, "curly", {}),
    "00224-ann":       (38, 3.5, "long_straight", {}),
    "00228-kick":      (55, 5.0, "wavy", {}),
    "00231-hanna":     (90, 6.0, "long_straight", {"tail": 9.0}),
    "00234-sowara":    (90, 6.0, "long_straight", {"tail": 9.0}),
    "00235-sowara":    (90, 6.0, "long_straight", {"tail": 9.0}),
    "00239-pearl":     (38, 2.0, "short_smooth", {}),
    "00241-puchi":     (40, 4.0, "wavy", {}),
    "00248-hawl":      (40, 3.5, "long_straight", {}),
    "00250-lapisLazuli":(35, 1.5, "short_smooth", {}),
    "00256-nagi":      (55, 4.5, "curly", {}),
    "00260-momo":      (40, 3.5, "double_coat", {}),
}

STIFFNESS = {"short_smooth": 0.55, "double_coat": 0.40, "long_straight": 0.28,
             "wavy": 0.30, "curly": 0.38, "wire": 0.65}
SPINE, NECK, SKULL, MUZZLE = list(range(1, 7)), [15], [16], [32]
EARS = [33, 34]
LEG_U, LEG_M, PAW = [8, 12, 18, 22], [9, 13, 19, 23], [10, 14, 20, 24]
TAIL = list(range(25, 32))


def expand(dog, diag, body, curl, ov):
    r = dict(face=min(2.0, body * 0.35), muzzle=min(1.5, body * 0.25),
             ears=body * 0.7, leg_u=body * 0.75, leg_m=body * 0.55,
             paw=min(1.5, body * 0.3),
             tail=body * (1.5 if curl == "short_smooth" else 1.3))
    r.update({k: v for k, v in ov.items() if k != "note"})
    jl = {}
    for js, cm in ((SPINE, body), (NECK, body * 1.05), (SKULL, r["face"]),
                   (MUZZLE, r["muzzle"]), (EARS, r["ears"]), (LEG_U, r["leg_u"]),
                   (LEG_M, r["leg_m"]), (PAW, r["paw"]), (TAIL, r["tail"])):
        for j in js:
            jl[str(j)] = round(cm, 2)
    return dict(dog=dog, dog_bbox_diag_cm=float(diag), default_cm=round(body, 2),
                curl_class=curl, stiffness=STIFFNESS[curl],
                joint_lengths_cm=jl, note=ov.get("note", ""))


def main():
    out_dir = "exps/vlm_priors"
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for dog, (diag, body, curl, ov) in ANN.items():
        json.dump(expand(dog, diag, body, curl, ov),
                  open(os.path.join(out_dir, f"{dog}.json"), "w"), indent=1)
        counts[curl] = counts.get(curl, 0) + 1
    print(f"{len(ANN)} priors -> {out_dir}; curl distribution: {counts}")


if __name__ == "__main__":
    main()
