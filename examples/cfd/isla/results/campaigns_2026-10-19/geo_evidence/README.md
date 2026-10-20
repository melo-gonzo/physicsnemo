# The exact-geometry evidence, reduced for the book

`geo_evidence.py` reads the GEO1 score records (`../../campaigns_2026-10-14/afu/`) and the GEO-CONT score records
(`../../campaigns_2026-10-18/geo_cont/scores/`) with the standing's ISLA arm (`../../char/{airfrans,lug,tensile2d}_a2.json`)
and writes `geo_evidence.json`: per arm, protocol, field and seed, and the ratios the book quotes (GEO1 over the standing's
arm; the GEO1 400-segment law and count effects; CONT / CONTROL; CONT / A2 and CONTROL / A2; posed / native per seed). Its
docstring defines every arm. Run: `python geo_evidence.py` (CPU, seconds).
