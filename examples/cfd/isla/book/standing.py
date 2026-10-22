# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The book's one reader of the standing artifacts.

Every chapter that quotes how the three architectures compare reads it through this module,
so the artifact behind each dataset, the ratio convention and the seed rule are written once.

The standing artifacts live in ``results/char/<file>.json`` and share one schema::

    {"protocols": {"native": label, "area": label, "posed": label},
     "fields":    {key: label},
     "isla_arm":  arm name,
     "arms":      {arm name: {protocol: {field: {"mean": x, "per_seed": {seed: x},
                                                 "n_cases": n, "n_seeds": n}}}}}

Conventions (STYLE.md):

* a ratio is *baseline ÷ ISLA*, so a value above one favors ISLA;
* a *shift* is an arm's error under a protocol divided by the same arm's native error;
* the seed rule: one arm leads another on a field only if every seed of the one sits below every
  seed of the other *and* the seed means differ by at least the tie band (3%, the tie band of the
  rule that selected the reference configurations); otherwise the two are *level*.

``DATASETS`` is the registry. When a rerun lands (a new artifact file), change its entry here and
every chapter follows.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "results"
CHAR = RES / "char"

ARCHS = ("ISLA", "GeoTransolver", "Transolver")
BASELINES = ("GeoTransolver", "Transolver")

#: How each architecture's reference configuration is written in prose and captions.
CONFIG = {
    "ISLA": "ISLA (width 512, 12 layers, 512 slices)",
    "GeoTransolver": "GeoTransolver (width 256)",
    "Transolver": "Transolver (width 512, 16 layers)",
}

#: Seed means closer than this fraction are level, whatever the seeds do (the selection rule's tie band).
TIE_BAND = 0.03

#: The form of ISLA every standing's ISLA checkpoints (and every trained ISLA the book shows) are of, as captions name it.
FORM_TAG = "nearest-token form"

#: The one sentence that says what that form is, for a chapter or a caption that needs it whole.
FORM_NOTE = (
    f"The ISLA checkpoints of the standing are of the {FORM_TAG}: the normalized slice reads, one boundary schema shared by every condition, boundary tokens drawn by the datapipe, each prediction point's wall "
    "distance and direction from the nearest sampled boundary token, on AirfRANS, Tensile2d and the lug the stored signed "
    "distance read as a prediction-point input, a length unit from the given cells' centroids alone, and no declared scales "
    "(@sec-checkpoint-form)."
)

#: The protocol names as the book writes them.
PROTOCOL = {
    "native": "native mesh",
    "area": "uniform point cloud",
    "posed": "rotated and translated",
}


@dataclass(frozen=True)
class Dataset:
    """One dataset and task of the standing."""

    key: str
    name: str  # as written in a cross-architecture sentence
    artifact: str | None  # file under results/char/, None while no standing exists
    headline: tuple  # the fields the headline figure reads, in order
    protocols: tuple = (
        "native",
        "posed",
    )  # the protocols that mean something on this task
    section: str = ""  # the chapter section that holds the standing
    reference: bool = (
        True  # False: ISLA's arm is from checkpoints without the normalized slice reads
    )
    provisional: dict = field(
        default_factory=dict
    )  # field -> why its ratios are given without a verdict although the inputs are fair
    notes: dict = field(default_factory=dict)
    # the artifact (file under results/char/) or arm of ISLA at the current class on this task, None until it is scored; a standing
    # table then shows its one ISLA row (native, area and posed seed means and per-seed values) beside the nearest-token-form row
    current: str | None = None

    @property
    def short(self) -> str:
        """The name without its task or rung qualifier ("AirfRANS", "HiLiftAeroML", "DrivAerML Surface")."""
        return (
            self.name.split(" (")[0].replace(" interior task", "").replace(" task", "")
        )

    @property
    def in_sentence(self) -> str:
        """The name as it reads after a preposition: 'the DrivAerML Surface task', 'SHIFT-SUV'."""
        return ("the " + self.name) if self.name.endswith("task") else self.name


#: The (dataset key, architecture) baseline arms that read the freestream as one constant physical vector in m/s, where ISLA reads
#: its unit direction: none. Every baseline arm of a flow standing reads the unit direction (FAIR_INPUT of
#: results/char/a2r_protocol_artifacts_2026-10-17.py and results/char/hilift_a2_2026-10-20.py, each replaced arm kept under
#: raw_drive_<arch>; AirfRANS: results/char/airfrans_a2_2026-10-18.py).
CONSTANT_DRIVE = ()

DATASETS = {
    d.key: d
    for d in (
        Dataset(
            "drivaerml_surface",
            "DrivAerML Surface task",
            "drivaerml_surface_a2.json",
            ("pressure_l2", "wss_l2"),
            ("native", "area", "posed"),
            "sec-scoreboard-200k",
        ),
        Dataset(
            "drivaerml_volume",
            "DrivAerML Volume task",
            "drivaerml_volume_a2.json",
            ("pressure_l2", "velocity_l2", "nut_l2"),
            ("native", "posed"),
            "sec-interior-results",
        ),
        # ISLA's arm is the normalized-read rerun (a2r_hl_isla) and the baselines read U_inf_dir (UDRV-HL: udrv_hl_gt, udrv_hl_t), every
        # arm on ISLA's 200,000-cell set (results/char/hilift_a2_2026-10-20.py); the CHAR record's raw-read ISLA arm and the baselines
        # reading U_inf, the physical freestream vector in inch/s, are kept in the file (raw_read_isla, raw_drive_<arch>, char_baselines);
        # no current-class row: the class's standing is the open question of 20 (sec-oq-class-standing)
        Dataset(
            "hilift",
            "HiLiftAeroML (510 training cases)",
            "hilift_surface_a2.json",
            ("pressure_l2", "tau_wall_l2"),
            ("native", "area", "posed"),
            "sec-de-result",
        ),
        Dataset(
            "shift_suv",
            "SHIFT-SUV",
            "shift_suv_surface_a2.json",
            ("pressure_l2", "wss_l2"),
            ("native", "area", "posed"),
            "sec-shift-suv-standing",
        ),
        # the baselines of this file read U_inf_dir and the standardized inlet speed (DRIVE-FIX, campaigns_2026-10-19/drive_fix/);
        # they run AirfRANS embedded in 3-D, and their velocity_l2 equals the in-plane error to within 4e-6 relative on the native mesh
        # (AF-PLANAR, campaigns_2026-10-19/drive_fix2/planar/), so velocity is compared like the other fields
        Dataset(
            "airfrans",
            "AirfRANS interior task",
            "airfrans_a2.json",
            ("pressure_l2", "velocity_l2", "nut_l2"),
            ("native", "posed"),
            "sec-interior-airfrans",
            provisional={},
        ),
        Dataset("aerozoo", "AeroZoo", None, (), ("native",), "sec-interior-aerozoo"),
        Dataset(
            "tensile2d",
            "PLAID Tensile2d",
            "tensile2d_a2.json",
            ("stress_l2", "displacement_l2", "q_l2"),
            ("native", "posed"),
            "sec-interior-tensile2d",
        ),
        Dataset(
            "lug",
            "GINOT bracket lug",
            "lug_a2.json",
            ("von_mises_l2",),
            ("native", "posed"),
            "sec-interior-lug",
        ),
        Dataset(
            "deepjeb",
            "DeepJEB++",
            "deepjeb_surface_a2.json",
            ("von_mises_l2", "displacement_l2"),
            ("native", "posed"),
            "sec-surface-deepjeb",
        ),
    )
}


def _arch(arm_name: str) -> str:
    """The architecture of an arm: the first word of its label."""
    return arm_name.split()[0]


class Standing:
    """The standing of the three reference configurations on one dataset."""

    def __init__(self, key: str):
        self.ds = DATASETS[key]
        path = CHAR / self.ds.artifact
        if not path.exists():
            raise FileNotFoundError(f"{self.ds.name}: no standing artifact at {path}")
        self.path = path
        self.raw = json.load(open(path))
        self.fields = self.raw["fields"]  # key -> label
        self.arms = {_arch(name): arm for name, arm in self.raw["arms"].items()}
        self.arm_names = {_arch(name): name for name in self.raw["arms"]}

    @property
    def source(self) -> str:
        """The artifact path as captions print it."""
        return f"results/char/{self.ds.artifact}"

    def has(self, arch: str) -> bool:
        """Whether the artifact holds an arm of this architecture."""
        return arch in self.arms

    def cell(self, arch: str, f: str, protocol: str = "native") -> dict:
        """One arm's record on one field under one protocol: {mean, per_seed, n_cases, n_seeds}."""
        return self.arms[arch][protocol][f]

    def err(self, arch: str, f: str, protocol: str = "native") -> float:
        """Seed-mean error."""
        return self.cell(arch, f, protocol)["mean"]

    def seeds(self, arch: str, f: str, protocol: str = "native") -> list:
        """Per-seed errors."""
        return list(self.cell(arch, f, protocol)["per_seed"].values())

    def ratio(self, arch: str, f: str, protocol: str = "native") -> float:
        """baseline ÷ ISLA under one protocol (seed means)."""
        return self.err(arch, f, protocol) / self.err("ISLA", f, protocol)

    def shift(self, arch: str, f: str, protocol: str) -> float:
        """An arm's error under a protocol over its native error."""
        return self.err(arch, f, protocol) / self.err(arch, f, "native")

    def compared(self, arch: str, f: str) -> bool:
        """Whether a verdict is drawn on this field against this architecture (the field is not ``provisional``)."""
        return arch == "ISLA" or f not in self.ds.provisional

    def verdict(self, arch: str, f: str, protocol: str = "native") -> str:
        """ISLA against one baseline by the seed rule: 'leads', 'trails' or 'is level with'; 'is not compared with'
        where the field is ``provisional``."""
        if not self.compared(arch, f):
            return "is not compared with"
        b, i = self.seeds(arch, f, protocol), self.seeds("ISLA", f, protocol)
        if abs(self.ratio(arch, f, protocol) - 1) < TIE_BAND:
            return "is level with"
        if min(b) > max(i):
            return "leads"
        if max(b) < min(i):
            return "trails"
        return "is level with"

    def table(self, label: str, setting: str, protocols=None, fields=None) -> str:
        """The headline table: one row per architecture, one column group per protocol, ratio rows.

        Returns markdown with a Quarto caption; print it from an ``output: asis`` cell."""
        protocols = tuple(protocols or self.ds.protocols)
        fields = tuple(fields or self.fields)
        flab = " / ".join(self.fields[f] for f in fields)
        out = [
            "| | " + " | ".join(f"{PROTOCOL[p]}: {flab}" for p in protocols) + " |",
            "|---|" + "---:|" * len(protocols),
        ]
        for arch in ARCHS:
            if not self.has(arch):
                continue
            out.append(
                f"| {CONFIG[arch]} | "
                + " | ".join(
                    " / ".join(f"{self.err(arch, f, p):.4f}" for f in fields)
                    for p in protocols
                )
                + " |"
            )
        for arch in BASELINES:
            if self.has(arch):
                out.append(
                    f"| {arch} ÷ ISLA | "
                    + " | ".join(
                        " / ".join(
                            fmt_ratio(self.ratio(arch, f, p))
                            + ("§" if f in self.ds.provisional else "")
                            for f in fields
                        )
                        for p in protocols
                    )
                    + " |"
                )
        cap = (
            f": {setting}: the architectures at their reference configurations, {flab}, "
            f"relative L2 error (@sec-metrics), seed means over two seeds, float32; ISLA checkpoints of the {FORM_TAG}"
            + ("" if self.ds.reference else ", with the raw slice reads")
            + " (@sec-checkpoint-form); "
            + "each column scores the same checkpoint of each seed on the test cases as presented: "
            + ", ".join(f"*{PROTOCOL[p]}*" for p in protocols)
            + ". How to read this: lower is better; "
            + "a ratio row is baseline ÷ ISLA, so a value above one favors ISLA. "
            + "".join(
                f"§ {self.fields[f]} is given without a verdict: {why}. "
                for f, why in self.ds.provisional.items()
                if f in fields
            )
            + f"Data: {self.source}. {{#{label}}}"
        )
        return "\n".join(out) + "\n\n" + cap + "\n"


def fmt_ratio(r: float) -> str:
    """A ratio as the book writes it: two decimals below 2, one decimal below 10, none above."""
    return f"{r:.0f}" if r >= 10 else f"{r:.1f}" if r >= 2 else f"{r:.2f}"


def fmt_pct(r: float) -> str:
    """A ratio near one as a percentage difference: 1.07 -> '7%'."""
    return f"{abs(r - 1) * 100:.0f}%"


def unit_drive_control(scored_set: str = "registered"):
    """{field: error with the unit direction ÷ error with the constant physical vector}: Transolver at its reference configuration
    on the DrivAerML Surface task, the same lanes retrained reading U_inf_dir (DRIVE-FIX, r80ud_dr_t), seed means. ``scored_set``:
    "registered", both arms on their own 100,000-cell draw, the set the control's band was registered on; "isla", both re-scored on
    ISLA's 200,000-cell set (T-SCORESET, campaigns_2026-10-19/drive_fix2/scoreset/)."""
    ref = Standing("drivaerml_surface")
    raw = ref.raw.get(
        "raw_drive_transolver"
    )  # the physical-vector arm, once a unit-direction arm replaces it (FAIR_INPUT)
    if scored_set == "registered":
        rec = json.load(
            open(
                RES
                / "campaigns_2026-10-19"
                / "drive_fix"
                / "drivefix_scores_2026-10-19.json"
            )
        )
        run = "evals_r80ud/r80ud_dr_t_h512_l16_c100k_seed{}/native"
        arm = ref.raw.get("transolver_100k_scoring") or raw or ref.arms["Transolver"]
    else:
        rec = json.load(
            open(
                RES
                / "campaigns_2026-10-19"
                / "drive_fix2"
                / "scoreset"
                / "tset200k_scores.json"
            )
        )["files"]
        run = "evals_tset200k/r80ud_dr_t_h512_l16_c100k_seed{}/native"
        arm = raw or ref.arms["Transolver"]
    return {
        f: sum(rec[run.format(s)]["metrics"][f] for s in (42, 43))
        / 2
        / arm["native"][f]["mean"]
        for f in ref.ds.headline
    }


def unit_drive(key: str, arch: str) -> dict:
    """{field: native error reading the unit direction ÷ native error reading the physical vector}, seed means, for a baseline arm
    that FAIR_INPUT replaced (the standing's arm over its ``raw_drive_<arch>``); empty if the standing keeps no such arm."""
    s = Standing(key)
    raw = s.raw.get(f"raw_drive_{arch.lower()}")
    return (
        {f: s.err(arch, f) / raw["native"][f]["mean"] for f in s.ds.headline}
        if raw
        else {}
    )


def available():
    """The datasets whose standing artifact exists, in registry order."""
    return [
        k for k, d in DATASETS.items() if d.artifact and (CHAR / d.artifact).exists()
    ]


# --- The standing as figures (index, 08-standing) -------------------------------------------------


def headline_rows():
    """(dataset key, field key, row label) for every headline field of every dataset with a standing."""
    out = []
    for k in available():
        s = Standing(k)
        for f in s.ds.headline:
            out.append((k, f, f"{s.ds.name}: {s.fields[f]}"))
    return out


def _seed_range(s, arch, f, protocol="native"):
    """The smallest and largest ratio over the seed pairs (baseline seed ÷ ISLA seed)."""
    r = [
        b / i for b in s.seeds(arch, f, protocol) for i in s.seeds("ISLA", f, protocol)
    ]
    return min(r), max(r)


def plot_native(ax, rows=None, note_flag="†"):
    """Forest plot of baseline ÷ ISLA on the native mesh: one row per (dataset, field), both baselines, the seed-pair
    range as a bar, log x. Returns the row labels drawn (with ``note_flag`` on rows whose ISLA arm is not the
    reference configuration)."""
    from palette import STYLE, GRID, INK2, MUTED, BASELINE, halo

    rows = rows or headline_rows()
    y = list(range(len(rows)))[::-1]
    labels = []
    for yy, (k, f, lab) in zip(y, rows):
        s = Standing(k)
        flag = ("" if s.ds.reference else f" {note_flag}") + (
            " §" if f in s.ds.provisional else ""
        )
        labels.append(lab + flag)
        for arch, key, dy in (
            ("GeoTransolver", "gt", 0.16),
            ("Transolver", "t", -0.16),
        ):
            if not s.has(arch):
                continue
            st = STYLE[key]
            lo, hi = _seed_range(s, arch, f)
            if s.compared(arch, f):
                ax.plot(
                    [lo, hi],
                    [yy + dy] * 2,
                    color=st["color"],
                    lw=4,
                    alpha=0.3,
                    solid_capstyle="butt",
                )
                ax.plot(
                    s.ratio(arch, f),
                    yy + dy,
                    marker=st["marker"],
                    color=st["color"],
                    ms=6.5,
                    mec="white",
                    mew=0.6,
                    ls="none",
                )
            else:  # a provisional field: hollow marks, no seed bar
                ax.plot(
                    s.ratio(arch, f),
                    yy + dy,
                    marker=st["marker"],
                    color="white",
                    ms=6.5,
                    mec=st["color"],
                    mew=1.3,
                    ls="none",
                )
    ax.axvline(1.0, color=INK2, lw=1.0)
    ax.axvspan(1.0 - TIE_BAND, 1.0 + TIE_BAND, color=GRID, alpha=0.8, lw=0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.6)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    return labels


def plot_shifts(ax, protocol="posed", keys=None, field=None):
    """Each architecture's error under ``protocol`` over its native error, one row per dataset (first headline field),
    log x."""
    from palette import STYLE, INK2

    keys = keys or [k for k in available() if protocol in DATASETS[k].protocols]
    y = list(range(len(keys)))[::-1]
    for yy, k in zip(y, keys):
        s = Standing(k)
        f = field or s.ds.headline[0]
        for arch, key, dy in (
            ("ISLA", "isla_ref", 0.0),
            ("GeoTransolver", "gt", 0.22),
            ("Transolver", "t", -0.22),
        ):
            if s.has(arch):
                st = STYLE[key]
                ax.plot(
                    s.shift(arch, f, protocol),
                    yy + dy,
                    marker=st["marker"],
                    color=st["color"],
                    ms=7,
                    mec="white",
                    mew=0.6,
                    ls="none",
                )
    ax.axvline(1.0, color=INK2, lw=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{DATASETS[k].name}: {Standing(k).fields[field or DATASETS[k].headline[0]]}"
            for k in keys
        ],
        fontsize=8.6,
    )
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.grid(axis="y", visible=False)
    return keys


# --- Cost of the reference configurations ------------------------------------------------------------
#: Per task and architecture: training peak memory (GiB, max over four GB300 ranks), epoch wall time (s) and parameter
#: count, from the one-epoch lanes of the reference configurations. The lanes ran with the raw slice reads; the
#: normalized reads add no parameter and no tensor of note, so the counts are exact for the standing's checkpoints (the
#: nearest-token form) and the cost representative.
COST_FILE = "char_lanes_cost_2026-10-04.json"
_COST_KEY = {
    "drivaerml_surface": "DrivAerML Surface",
    "drivaerml_volume": "DrivAerML Volume",
    "shift_suv": "Shift-SUV",
    "airfrans": "AirfRANS",
    "tensile2d": "Tensile2D",
    "lug": "lug",
    "deepjeb": "DeepJEB++",
}


def cost(key: str, arch: str):
    """{peak_gib, epoch_seconds, n_parameters, source} of one reference configuration on one task, or None."""
    c = json.load(open(CHAR / COST_FILE))["cost"].get(_COST_KEY.get(key, ""), {})
    return c.get(arch)


# --- The recipe's ISLA configurations -----------------------------------------------------------------
RECIPE = (
    Path(__file__).resolve().parents[2]
    / "external_aerodynamics"
    / "unified_external_aero_recipe"
)
#: The dataset file each ISLA configuration that leaves constants to its dataset trains on (@tbl-data-files).
ISLA_DATASET = {
    "isla_volume_aerozoo": "aerozoo_uvolume",
    "isla_volume_airfrans": "airfrans_uvolume_2d",
}


def isla_constants(name: str) -> dict:
    """The ``isla_constants`` block of an ISLA configuration's dataset file (``{}`` for one that leaves none to it)."""
    import yaml

    if name not in ISLA_DATASET:
        return {}
    return yaml.safe_load(
        (RECIPE / "datasets" / f"{ISLA_DATASET[name]}.yaml").read_text()
    )["isla_constants"]


def recipe_model(name: str) -> dict:
    """The ``model`` block of the recipe's ``conf/model/<name>.yaml`` with the constants it leaves to its dataset
    (``reynolds: null``, ``scale: null``) filled from the dataset file's ``isla_constants``, as train.py and infer.py
    fill them (``output_normalize.model_dataset_constants``); the class refuses the block without them."""
    import sys

    import yaml
    from omegaconf import OmegaConf

    if str(RECIPE / "src") not in sys.path:
        sys.path.insert(0, str(RECIPE / "src"))
    from output_normalize import model_dataset_constants

    block = yaml.safe_load(
        (RECIPE / "conf" / "model" / f"{name}.yaml").read_text()
    ).get("model", {})
    dataset = OmegaConf.create({"isla_constants": isla_constants(name)})
    return {**block, **model_dataset_constants(OmegaConf.create(block), dataset)}
