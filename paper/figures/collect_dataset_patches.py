"""Copy the 22 example patches (one per class, official TRAIN split) used in Fig. dataset into
figures/dataset_patches/<Class>.png and record their provenance. Illustration only; not used in any result."""
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
SCR = Path(sys.argv[1])  # scratchpad holding cand/ (from the GPU PC) and mendeley_dl/ (from Mendeley)
OUT = HERE / "dataset_patches"
OUT.mkdir(exist_ok=True)

# class -> (source file, micrograph_vertex in the official train split)
PICK = {
    "Adenovirus": "cand/train_Adenovirus_3_27_5.png", "Astrovirus": "mendeley_dl/Astrovirus/13_1.tif",
    "CCHF": "cand/train_CCHF_2_19_2.png", "Cowpox": "mendeley_dl/Cowpox/19_1.tif",
    "Ebola": "cand/train_Ebola_4_38_6.png", "Influenza": "cand/train_Influenza_0_11_16.png",
    "Lassa": "mendeley_dl/Lassa/11_1.tif", "Marburg": "cand/train_Marburg_1_22_1.png",
    "Nipah virus": "mendeley_dl/Nipahvirus/14_1.tif", "Norovirus": "cand/train_Norovirus_1_24_16.png",
    "Orf": "cand/train_Orf_0_22_1.png", "Papilloma": "cand/train_Papilloma_4_16_14.png",
    "Rift Valley": "cand/train_RiftValley_1_40_5.png", "Rotavirus": "cand/train_Rotavirus_1_12_4.png",
    "Dengue": "cand/train_Dengue_2_12_1.png", "Guanarito": "cand/train_Guanarito_0_4_1.png",
    "LCM": "cand/train_LCM_2_18_1.png", "Machupo": "mendeley_dl/Machupo/20_1.tif",
    "Pseudocowpox": "mendeley_dl/Pseudocowpox/18_1.tif", "Sapovirus": "cand/train_Sapovirus_4_4_2.png",
    "TBE": "cand/train_TBE_2_3_1.png", "WestNile": "cand/train_WestNile_3_3_1.png",
}
prov = {}
for cls, rel in PICK.items():
    src = SCR / rel
    im = Image.open(src).convert("L")
    assert im.size == (256, 256), (cls, im.size)
    im.save(OUT / f"{cls.replace(' ', '')}.png")
    stem = Path(rel).stem
    patch = stem if rel.startswith("mendeley_dl") else "_".join(stem.split("_")[-2:])
    prov[cls] = {"split": "train", "patch": f"{patch}.tif",
                 "source": "Mendeley public API (sha256 verified)" if rel.startswith("mendeley_dl") else "GPU PC copy of the dataset"}
json.dump(prov, open(OUT / "SOURCES.json", "w"), indent=1)
print(len(prov), "patches written")
