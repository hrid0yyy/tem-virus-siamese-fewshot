"""Dataset constants.

Folder names and per-class information were verified against the Mendeley Data
file listing of dataset 10.17632/x4dwwfwtw3.3 and the authors' MATLAB / Keras code
(dataset code.zip and Mendeley dataset kxsvzhcfgs v2).
"""

# Name of the official pre-processed patch set (1 nm/pixel, 256x256 crops).
PATCH_ROOT_NAME = "context_virus_1nm_256x256"
RAW_ROOT_NAME = "context_virus_RAW"
EXCLUDED_DIR = "_EXCLUDED"            # sub-folder holding the 8 classes excluded in the paper
SPLITS = ("train", "validation", "test")
AUGMENTED_SPLIT = "augmented_train"   # authors' balanced (736/class) training set, 14 classes only
PATCH_SIZE = 256

# The 14 classes used for classification in Matuszewski & Sintorn (2021).
# Order == the authors' CLASSES_NAMES order (alphabetical folder glob), so that
# confusion matrices are directly comparable with their published ones.
BASE_CLASSES = [
    "Adenovirus", "Astrovirus", "CCHF", "Cowpox", "Ebola", "Influenza", "Lassa",
    "Marburg", "Nipah virus", "Norovirus", "Orf", "Papilloma", "Rift Valley", "Rotavirus",
]

# The 8 classes the paper excluded (too few images/particles). Stored under _EXCLUDED/.
NOVEL_CLASSES = ["Dengue", "Guanarito", "LCM", "Machupo", "Pseudocowpox", "Sapovirus", "TBE", "WestNile"]

ALL_CLASSES = BASE_CLASSES + NOVEL_CLASSES  # base first, then novel (convenient for GFSL indexing)

# Average particle size [nm] from the authors' getAllVirusData.m (reference only).
VIRUS_SIZE_NM = {
    "Adenovirus": 80, "Astrovirus": 25, "CCHF": 120, "Cowpox": 270, "Dengue": 45, "Ebola": 80,
    "Guanarito": 140, "Influenza": 110, "Lassa": 140, "LCM": 120, "Machupo": 120, "Marburg": 80,
    "Nipah virus": 95, "Norovirus": 30, "Orf": 145, "Papilloma": 55, "Pseudocowpox": 145,
    "Rift Valley": 90, "Rotavirus": 80, "Sapovirus": 30, "TBE": 50, "WestNile": 50,
}

# Virus family (general virology reference; used only for grouped error analysis).
VIRUS_FAMILY = {
    "Adenovirus": "Adenoviridae", "Astrovirus": "Astroviridae", "CCHF": "Nairoviridae",
    "Cowpox": "Poxviridae", "Dengue": "Flaviviridae", "Ebola": "Filoviridae",
    "Guanarito": "Arenaviridae", "Influenza": "Orthomyxoviridae", "Lassa": "Arenaviridae",
    "LCM": "Arenaviridae", "Machupo": "Arenaviridae", "Marburg": "Filoviridae",
    "Nipah virus": "Paramyxoviridae", "Norovirus": "Caliciviridae", "Orf": "Poxviridae",
    "Papilloma": "Papillomaviridae", "Pseudocowpox": "Poxviridae", "Rift Valley": "Phenuiviridae",
    "Rotavirus": "Reoviridae", "Sapovirus": "Caliciviridae", "TBE": "Flaviviridae",
    "WestNile": "Flaviviridae",
}

# Pseudo-novel class sets for feasibility test F2/F3 (hidden from encoder training).
# F2: classes without an obvious look-alike among the remaining base classes.
# F3: classes WITH a look-alike left in training (Marburg~Ebola, Orf~Cowpox (Poxviridae),
#     Adenovirus~Rift Valley (confused in the paper), CCHF~Rift Valley (both Bunyavirales)).
HIDDEN_PRESETS = {
    "f2": ["Astrovirus", "Influenza", "Papilloma", "Rotavirus"],
    "f3": ["Marburg", "Orf", "Adenovirus", "CCHF"],
}

MENDELEY_DATASET_ID = "x4dwwfwtw3"
MENDELEY_VERSION = 3
MENDELEY_ZIP_URL = f"https://data.mendeley.com/public-api/zip/{MENDELEY_DATASET_ID}/download/{MENDELEY_VERSION}"


def resolve_classes(spec):
    """Turn a class spec into an ordered list of class folder names.

    spec: 'base14' | 'novel8' | 'all22' | comma-separated class names | list of names.
    """
    if isinstance(spec, (list, tuple)):
        names = list(spec)
    else:
        key = spec.strip().lower()
        if key == "base14":
            return list(BASE_CLASSES)
        if key == "novel8":
            return list(NOVEL_CLASSES)
        if key == "all22":
            return list(ALL_CLASSES)
        if key.startswith("preset:"):
            return list(HIDDEN_PRESETS[key.split(":", 1)[1]])
        names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in ALL_CLASSES]
    if unknown:
        raise ValueError(f"Unknown class name(s): {unknown}. Valid: {ALL_CLASSES}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate class names in {names}")
    return names
