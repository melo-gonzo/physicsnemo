# Where the book's numbers come from

Every number in a chapter is computed, as the page renders, from a file under
`examples/cfd/isla/results/`: the standings from `results/char/<dataset>[_a2].json`,
the other studies from the artifact named in each table or figure caption. The dated
record of how each number was produced is the lab notebook, `18-notebook.qmd`.

This file used to hold a hand-maintained claims ledger. It stopped being kept current
in September 2026 and is retired; its last version is in git history
(`git log -- examples/cfd/isla/book/CLAIMS.md`).
