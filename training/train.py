"""Training entry point (see training/trainer.py for the loop).

    python -m training.train --run s1 --stage 1 --fixed-stage --total-steps 20000000
    python -m training.train --run cur --stage 1 --resume checkpoints/s1/best.pt   # full curriculum
    python -m training.train --run smoke --smoke                                   # 2-minute smoke test

Why a separate tiny launcher: on Windows, multiprocessing "spawn" children re-import the __main__
module. If __main__ imported torch, each of the 24 env workers would load torch's CUDA DLLs
(~2-3 GB commit each) and exhaust the page file. This file imports nothing heavy.
"""

import os

# One BLAS/OpenMP thread per process (inherited by every spawned worker): OpenBLAS otherwise sizes
# per-thread buffers for all 28 cores in each of the 24 worker processes.
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

if __name__ == "__main__":
    from training.trainer import main

    main()
