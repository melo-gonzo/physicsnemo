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

#: The protocol names as the book writes them.
PROTOCOL = {"native": "native mesh", "area": "uniform point cloud", "posed": "rotated and translated"}


@dataclass(frozen=True)
class Dataset:
    """One dataset and task of the standing."""

    key: str
    name: str  # as written in a cross-architecture sentence
    artifact: str | None  # file under results/char/, None while no standing exists
    headline: tuple  # the fields the headline figure reads, in order
    protocols: tuple = ("native", "posed")  # the protocols that mean something on this task
    section: str = ""  # the chapter section that holds the standing
    reference: bool = True  # False: ISLA's arm is from checkpoints without the normalized slice reads
    fair_drive: bool = True  # False: the baselines in this artifact read the freestream as a physical vector, not a unit direction
    drive_note: str = ""  # what the baselines read instead, for the one sentence the chapters print (results/drive_audit_2026-10-18/)
    notes: dict = field(default_factory=dict)


DATASETS = {
    d.key: d
    for d in (
        Dataset("drivaerml_surface", "DrivAerML Surface task", "drivaerml_surface_a2.json",
                ("pressure_l2", "wss_l2"), ("native", "area", "posed"), "sec-scoreboard-200k"),
        Dataset("drivaerml_volume", "DrivAerML Volume task", "drivaerml_volume_a2.json",
                ("pressure_l2", "velocity_l2", "nut_l2"), ("native", "posed"), "sec-interior-results"),
        # the normalized-read rerun replaces this file when its lanes are scored (hilift_surface_a2.json); in this file the
        # baselines read global_data.U_inf, the physical freestream vector in inch/s (recipe_v8 geotransolver_surface.yaml
        # global_embedding, transolver_surface.yaml fx), so no ISLA-vs-baseline verdict is drawn from it (fair_drive=False)
        Dataset("hilift", "HiLiftAeroML, 210-wing rung", "hilift_surface.json",
                ("pressure_l2", "tau_wall_l2"), ("native", "area", "posed"), "sec-de-result", reference=False,
                fair_drive=False, drive_note="the freestream as the physical velocity vector in inch/s (magnitude 2,680 on every "
                "case, turning with the angle of attack), where ISLA reads its unit direction"),
        Dataset("shift_suv", "SHIFT-SUV", "shift_suv_surface_a2.json",
                ("pressure_l2", "wss_l2"), ("native", "area", "posed"), "sec-shift-suv-standing"),
        # the baselines of this file read global_data.U_inf, the raw inlet velocity (results/drive_audit_2026-10-18/, F1)
        Dataset("airfrans", "AirfRANS interior task", "airfrans_a2.json",
                ("pressure_l2", "velocity_l2", "nut_l2"), ("native", "posed"), "sec-interior-airfrans",
                fair_drive=False, drive_note="the inlet velocity as the physical vector in m/s (31 to 94 m/s, at angles of "
                "-4.9 to 14.9 degrees), where ISLA reads its unit direction and the speed as a scalar"),
        Dataset("aerozoo", "AeroZoo", None, (), ("native",), "sec-interior-aerozoo"),
        Dataset("tensile2d", "PLAID Tensile2d", "tensile2d_a2.json",
                ("stress_l2", "displacement_l2", "q_l2"), ("native", "posed"), "sec-interior-tensile2d"),
        Dataset("lug", "GINOT bracket lug", "lug_a2.json", ("von_mises_l2",), ("native", "posed"), "sec-interior-lug"),
        Dataset("deepjeb", "DeepJEB++", "deepjeb_surface_a2.json",
                ("von_mises_l2", "displacement_l2"), ("native", "posed"), "sec-surface-deepjeb"),
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
        return arch in self.arms

    def cell(self, arch: str, f: str, protocol: str = "native") -> dict:
        return self.arms[arch][protocol][f]

    def err(self, arch: str, f: str, protocol: str = "native") -> float:
        """Seed-mean error."""
        return self.cell(arch, f, protocol)["mean"]

    def seeds(self, arch: str, f: str, protocol: str = "native") -> list:
        return list(self.cell(arch, f, protocol)["per_seed"].values())

    def ratio(self, arch: str, f: str, protocol: str = "native") -> float:
        """baseline ÷ ISLA under one protocol (seed means)."""
        return self.err(arch, f, protocol) / self.err("ISLA", f, protocol)

    def shift(self, arch: str, f: str, protocol: str) -> float:
        """An arm's error under a protocol over its native error."""
        return self.err(arch, f, protocol) / self.err(arch, f, "native")

    def verdict(self, arch: str, f: str, protocol: str = "native") -> str:
        """ISLA against one baseline by the seed rule: 'leads', 'trails' or 'is level with'; 'is not compared with'
        where the baselines' inputs differ from the protocol (``fair_drive`` False)."""
        if not self.ds.fair_drive and arch != "ISLA":
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
        out = ["| | " + " | ".join(f"{PROTOCOL[p]}: {flab}" for p in protocols) + " |",
               "|---|" + "---:|" * len(protocols)]
        for arch in ARCHS:
            if not self.has(arch):
                continue
            out.append(f"| {CONFIG[arch]} | "
                       + " | ".join(" / ".join(f"{self.err(arch, f, p):.4f}" for f in fields) for p in protocols) + " |")
        for arch in BASELINES:
            if self.has(arch):
                out.append(f"| {arch} ÷ ISLA | "
                           + " | ".join(" / ".join(fmt_ratio(self.ratio(arch, f, p)) for f in fields) for p in protocols) + " |")
        cap = (f": {setting}: the architectures at their reference configurations, {flab}, "
               f"relative L2 error (@sec-metrics), seed means over two seeds, float32; "
               + ", ".join(f"*{PROTOCOL[p]}*" for p in protocols)
               + " score one checkpoint per seed. How to read this: lower is better; a ratio row is baseline ÷ ISLA, "
                 f"so a value above one favors ISLA. Data: {self.source}. {{#{label}}}")
        return "\n".join(out) + "\n\n" + cap + "\n"


def fmt_ratio(r: float) -> str:
    """A ratio as the book writes it: two decimals below 2, one decimal below 10, none above."""
    return f"{r:.0f}" if r >= 10 else f"{r:.1f}" if r >= 2 else f"{r:.2f}"


def fmt_pct(r: float) -> str:
    """A ratio near one as a percentage difference: 1.07 -> '7%'."""
    return f"{abs(r - 1) * 100:.0f}%"


def available():
    """The datasets whose standing artifact exists, in registry order."""
    return [k for k, d in DATASETS.items() if d.artifact and (CHAR / d.artifact).exists()]


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
    r = [b / i for b in s.seeds(arch, f, protocol) for i in s.seeds("ISLA", f, protocol)]
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
        flag = ("" if s.ds.reference else f" {note_flag}") + ("" if s.ds.fair_drive else " ‡")
        labels.append(lab + flag)
        for arch, key, dy in (("GeoTransolver", "gt", 0.16), ("Transolver", "t", -0.16)):
            if not s.has(arch):
                continue
            st = STYLE[key]
            lo, hi = _seed_range(s, arch, f)
            if s.ds.fair_drive:
                ax.plot([lo, hi], [yy + dy] * 2, color=st["color"], lw=4, alpha=0.3, solid_capstyle="butt")
                ax.plot(s.ratio(arch, f), yy + dy, marker=st["marker"], color=st["color"], ms=6.5, mec="white", mew=0.6, ls="none")
            else:  # baselines fed the physical freestream vector: hollow marks, no seed bar
                ax.plot(s.ratio(arch, f), yy + dy, marker=st["marker"], color="white", ms=6.5, mec=st["color"], mew=1.3, ls="none")
    ax.axvline(1.0, color=INK2, lw=1.0)
    ax.axvspan(1.0 - TIE_BAND, 1.0 + TIE_BAND, color=GRID, alpha=0.8, lw=0)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.6)
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
        for arch, key, dy in (("ISLA", "isla_ref", 0.0), ("GeoTransolver", "gt", 0.22), ("Transolver", "t", -0.22)):
            if s.has(arch):
                st = STYLE[key]
                ax.plot(s.shift(arch, f, protocol), yy + dy, marker=st["marker"], color=st["color"], ms=7, mec="white", mew=0.6, ls="none")
    ax.axvline(1.0, color=INK2, lw=1.0)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels([f"{DATASETS[k].name}: {Standing(k).fields[field or DATASETS[k].headline[0]]}" for k in keys], fontsize=8.6)
    ax.set_ylim(-0.7, len(keys) - 0.3)
    ax.grid(axis="y", visible=False)
    return keys


# --- Cost of the reference configurations ------------------------------------------------------------
#: Per task and architecture: training peak memory (GiB, max over four GB300 ranks), epoch wall time (s) and parameter
#: count, from the one-epoch lanes of the reference configurations. The lanes ran with the raw slice reads; the
#: normalized reads add no parameter and no tensor of note, so the counts are exact and the cost representative.
COST_FILE = "char_lanes_cost_2026-10-04.json"
_COST_KEY = {"drivaerml_surface": "DrivAerML Surface", "drivaerml_volume": "DrivAerML Volume", "shift_suv": "Shift-SUV",
             "airfrans": "AirfRANS", "tensile2d": "Tensile2D", "lug": "lug", "deepjeb": "DeepJEB++"}


def cost(key: str, arch: str):
    """{peak_gib, epoch_seconds, n_parameters, source} of one reference configuration on one task, or None."""
    c = json.load(open(CHAR / COST_FILE))["cost"].get(_COST_KEY.get(key, ""), {})
    return c.get(arch)
