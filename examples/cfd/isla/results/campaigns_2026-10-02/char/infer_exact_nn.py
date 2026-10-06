"""infer.py with ISLA's nearest-boundary search forced to the exact difference-form torch backend (diagnostic, 2026-10-03).

The model exposes no option for the search backend (`geo_kernel` is the per-layer region kernel); geometry.py calls
physicsnemo's `knn` with its default (cuML on CUDA). This wrapper rebinds that name to `knn(..., implementation="torch")`,
which materializes |x_i - y_j| in the difference form (exact), then executes infer.py as __main__ (Hydra resolves
`config_path` relative to the main script, so the wrapper must sit next to infer.py and run it with runpy).
"""
import functools, os, runpy, sys
import physicsnemo.experimental.models.isla.geometry as geometry
from physicsnemo.nn.functional.neighbors.knn.knn import knn
geometry.knn = functools.partial(knn, implementation="torch")
infer_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "infer.py")
sys.argv[0] = infer_py
runpy.run_path(infer_py, run_name="__main__")
