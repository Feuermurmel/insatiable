# insatiable

Nothing finished yet, see `example.py` for a simple usage of the library. Run `python example.py` to run the example:

```
$ py example.py 
nand(nand(nand(vars.x1), nand(nand(nand(vars.x2), vars.x3, vars.x4, vars.x5), nand(vars.x2, nand(nand(nand(vars.x3), vars.x4, vars.x5), nand(vars.x3, nand(nand(nand(vars.x4), vars.x5), nand(vars.x4, nand(vars.x5)))))))), nand(vars.x1, nand(nand(nand(vars.x2), nand(nand(nand(vars.x3), vars.x4, vars.x5), nand(vars.x3, nand(nand(nand(vars.x4), vars.x5), nand(vars.x4, nand(vars.x5)))))), nand(vars.x2, nand(nand(nand(vars.x3), nand(nand(nand(vars.x4), vars.x5), nand(vars.x4, nand(vars.x5)))), nand(vars.x3, nand(vars.x4), nand(vars.x5)))))))
p cnf 23 59
-5 6 -7 -8 -9 0
5 -6 0
5 7 0
5 8 0
5 9 0
7 -8 -9 -12 0
[...]
============================[ Problem Statistics ]=============================
|                                                                             |
|  Number of variables:            23                                         |
|  Number of clauses:              58                                         |
|  Parse time:                   0.00 s                                       |
|  Eliminated clauses:           0.00 Mb                                      |
|  Simplification time:          0.00 s                                       |
|                                                                             |
============================[ Search Statistics ]==============================
| Conflicts |          ORIGINAL         |          LEARNT          | Progress |
|           |    Vars  Clauses Literals |    Limit  Clauses Lit/Cl |          |
===============================================================================
===============================================================================
restarts              : 1
conflicts             : 0              (0 /sec)
decisions             : 1              (0.00 % random) (483 /sec)
propagations          : 1              (483 /sec)
conflict literals     : 0              ( nan % deleted)
Memory used           : 0.11 MB
CPU time              : 0.002071 s

SATISFIABLE
SAT
1 2 3 -4 5 6 7 -8 -9 10 -11 12 13 -14 15 16 -17 18 19 -20 21 22 -23 0
x1: True
x2: True
x3: True
x4: False
x5: False
```


## Development Setup

```
python3 -m venv venv
. venv/bin/activate
pip install -e '.[dev]'
```
