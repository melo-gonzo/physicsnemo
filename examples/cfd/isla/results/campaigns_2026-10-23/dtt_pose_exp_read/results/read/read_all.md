# DTT-POSE-EXP read (read_x.py)

## lug

Checks before reading: all pass

- dttposex_lug_dtt_seed42/native: epoch500 True, cases 600, ids=control True, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 6
- dttposex_lug_dtt_seed42/posed: epoch500 True, cases 600, ids=control None, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 6
- dttposex_lug_dtt_seed43/native: epoch500 True, cases 600, ids=control True, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 6
- dttposex_lug_dtt_seed43/posed: epoch500 True, cases 600, ids=control None, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 6

| field | arm s42 / s43 | control s42 / s43 | arm / control | D | 2 SE | N | flag | seed-matched | arm / ISLA | control / ISLA |
|---|---|---|---|---|---|---|---|---|---|---|
| von_mises_l2 | 0.00241021 / 0.00210274 | 0.00221933 / 0.00183431 | 1.1133 | +0.1073 | 0.2344 | **no change** | none | [1.0860, 1.1463] | 1.552 | 1.394 |

- von_mises_l2: posed / native arm [1.0001, 1.0014], control [1.5281, 1.9775]; posed N gain (ratio 0.6434); case median arm 0.0020694 control 0.0018664; arm better in 19.2% of cases; **paired: no paired label**; **P pass**

Q: pass: dttposex_lug_dtt_seed42/native 600/600 bitwise, pairs/pt 114.95; dttposex_lug_dtt_seed42/posed 600/600 bitwise, pairs/pt 114.95; dttposex_lug_dtt_seed43/native 600/600 bitwise, pairs/pt 114.95; dttposex_lug_dtt_seed43/posed 600/600 bitwise, pairs/pt 114.95

## af

Checks before reading: all pass

- dttposex_af_dtt_seed42/native: epoch500 True, cases 200, ids=control True, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 4
- dttposex_af_dtt_seed42/posed: epoch500 True, cases 200, ids=control None, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 4
- dttposex_af_dtt_seed43/native: epoch500 True, cases 200, ids=control True, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 4
- dttposex_af_dtt_seed43/posed: epoch500 True, cases 200, ids=control None, snapshot True, dataset True, extra True, SCOREPROOF md5 True origin True classes 4

| field | arm s42 / s43 | control s42 / s43 | arm / control | D | 2 SE | N | flag | seed-matched | arm / ISLA | control / ISLA |
|---|---|---|---|---|---|---|---|---|---|---|
| pressure_l2 | 0.0128019 / 0.0134642 | 0.012428 / 0.0123887 | 1.0584 | +0.0568 | 0.0505 | **no change** | up | [1.0301, 1.0868] | 1.052 | 0.994 |
| velocity_l2 | 0.00736782 / 0.00720454 | 0.00737341 / 0.00697252 | 1.0158 | +0.0157 | 0.0602 | **no change** | none | [0.9992, 1.0333] | 1.146 | 1.129 |
| nut_l2 | 0.0228764 / 0.0213511 | 0.0207112 / 0.0204842 | 1.0736 | +0.0710 | 0.0699 | **no change** | up | [1.1045, 1.0423] | 1.440 | 1.342 |

- pressure_l2: posed / native arm [1.0015, 1.0022], control [1.0017, 1.0022]; posed N no change (ratio 1.0583); case median arm 0.0090802 control 0.009588; arm better in 50.5% of cases
- velocity_l2: posed / native arm [1.0011, 1.0013], control [1.0010, 1.0012]; posed N no change (ratio 1.0159); case median arm 0.00542 control 0.0056095; arm better in 60.5% of cases
- nut_l2: posed / native arm [1.0001, 1.0002], control [1.0001, 1.0001]; posed N no change (ratio 1.0736); case median arm 0.01437 control 0.014422; arm better in 46.5% of cases

Q: pass: dttposex_af_dtt_seed42/native 200/200 bitwise, pairs/pt 21.03; dttposex_af_dtt_seed42/posed 200/200 bitwise, pairs/pt 21.03; dttposex_af_dtt_seed43/native 200/200 bitwise, pairs/pt 21.03; dttposex_af_dtt_seed43/posed 200/200 bitwise, pairs/pt 21.03

