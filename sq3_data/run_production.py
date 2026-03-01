"""
SQ3 Production Runner — n=50 multistep + n=100 option_d + plots.
Designed to run detached from SSH via Start-Process.
All output logged to sq3_data/production_run.log
"""
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "production_run.log")

class Tee:
    """Write to both file and stdout."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.file.write(data)
        self.file.flush()
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = Tee(LOG_PATH)
sys.stderr = sys.stdout

t0 = time.time()

try:
    print("=" * 60)
    print("  SQ3 PRODUCTION RUN")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Multistep (n=50)
    print("\n\n>>> PHASE 1: run_multistep.py (n=50) <<<\n")
    from run_multistep import run_multistep
    run_multistep()

    # 2. Option D (n=100, already fast)
    print("\n\n>>> PHASE 2: run_option_d.py (n=100) <<<\n")
    from run_option_d import run_option_d
    run_option_d()

    # 3. Plots
    print("\n\n>>> PHASE 3: plot_sq3.py <<<\n")
    from plot_sq3 import main as plot_main
    plot_main()

    elapsed = time.time() - t0
    print(f"\n\nALL DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")

except Exception:
    traceback.print_exc()
    print(f"\nFAILED after {time.time()-t0:.0f}s")
    sys.exit(1)
