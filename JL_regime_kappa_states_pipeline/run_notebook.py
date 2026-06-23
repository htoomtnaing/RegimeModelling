"""
run_notebook.py — execute regime_kappa_states.ipynb in place, streaming each cell's
stdout to the console live so you can watch per-stage progress (instead of nbconvert's
silent capture). Saves the executed notebook (with outputs) when done.

Usage:  python run_notebook.py
"""
import sys, asyncio, warnings

warnings.filterwarnings("ignore")

# quiet the Windows asyncio/zmq Proactor warning (cosmetic only)
if sys.platform.startswith("win"):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

import nbformat
from nbclient import NotebookClient

NB = "regime_kappa_states.ipynb"


class LiveClient(NotebookClient):
    """NotebookClient that echoes kernel stream (stdout) output to the real console live."""
    def output(self, outs, msg, *args, **kwargs):
        result = super().output(outs, msg, *args, **kwargs)
        try:
            if isinstance(msg, dict) and msg.get("msg_type") == "stream":
                sys.stdout.write(msg.get("content", {}).get("text", ""))
                sys.stdout.flush()
        except Exception:
            pass
        return result


def main():
    nb = nbformat.read(NB, as_version=4)
    client = LiveClient(nb, timeout=1800, kernel_name="python3")
    try:
        client.execute()
        print("\n[run_notebook] EXECUTED OK")
    finally:
        nbformat.write(nb, NB)  # persist outputs even on failure


if __name__ == "__main__":
    main()
