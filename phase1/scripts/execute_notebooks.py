"""Executes notebooks cell by cell (fresh namespace per notebook), embeds text + figure outputs, and fails on any error.
Use JupyterLab/`jupyter nbconvert --execute` if available; this runner exists for environments without Jupyter."""
import ast, base64, contextlib, io, json, pathlib, sys, traceback
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
NB = pathlib.Path(__file__).resolve().parents[1] / "05_Starter" / "notebooks"

def run(path):
    nb = json.loads(path.read_text()); ns = {"__name__": "__main__"}; n = 0; errors = []
    for c in nb["cells"]:
        if c["cell_type"] != "code": continue
        n += 1; src = "".join(c["source"]); outs = []; buf = io.StringIO(); last = None
        try:
            tree = ast.parse(src); body = tree.body
            if body and isinstance(body[-1], ast.Expr): last = ast.Expression(body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(ast.Module(body, []), "<cell>", "exec"), ns)
                val = eval(compile(last, "<cell>", "eval"), ns) if last else None
            if buf.getvalue(): outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
            if val is not None and not isinstance(val, plt.Figure) and not (hasattr(val, "__class__") and val.__class__.__module__.startswith("matplotlib")):
                outs.append({"output_type": "execute_result", "execution_count": n, "metadata": {}, "data": {"text/plain": repr(val).splitlines(True)}})
        except Exception:
            tb = traceback.format_exc(); errors.append(tb); outs.append({"output_type": "error", "ename": "Error", "evalue": tb.splitlines()[-1], "traceback": tb.splitlines()})
        for num in plt.get_fignums():
            b = io.BytesIO(); plt.figure(num).savefig(b, format="png", dpi=110, bbox_inches="tight")
            outs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": base64.b64encode(b.getvalue()).decode()}})
        plt.close("all"); c["outputs"] = outs; c["execution_count"] = n
    path.write_text(json.dumps(nb, indent=1)); return n, errors

if __name__ == "__main__":
    bad = 0
    for p in sorted(NB.glob("*.ipynb")):
        n, errs = run(p); print(f"{p.name}: {n} code cells, {len(errs)} errors"); bad += len(errs)
        for e in errs: print(e)
    sys.exit(1 if bad else 0)
