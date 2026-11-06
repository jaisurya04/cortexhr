"""Charts for the native SHAP run on all 430 employees. Run AFTER scripts/run_shap_full430.py:  python scripts/plot_shap_full430.py
Reads reports/task14/shap_full430_group_values.csv and shap_full430_comparison.csv; writes reports/figures/task14_native_shap_*.png (needs only matplotlib)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; T = ROOT / "reports" / "task14"; F = ROOT / "reports" / "figures"; F.mkdir(exist_ok=True)
g = pd.read_csv(T / "shap_full430_group_values.csv").set_index("employee_id"); cmp = pd.read_csv(T / "shap_full430_comparison.csv")
assert len(g) == 430, f"expected 430 rows, found {len(g)}"
imp = g.abs().mean().sort_values().tail(12)
fig, ax = plt.subplots(figsize=(7, 4.5)); ax.barh(imp.index, imp.values, color="#1f4e79"); ax.set_xlabel("mean |native SHAP value| (probability points)")
ax.set_title("Native SHAP, all 430 employees: global importance of HR groups"); fig.tight_layout(); fig.savefig(F / "task14_native_shap_global_importance.png", dpi=150); plt.close(fig)
order = g.abs().mean().sort_values().tail(12).index; fig, ax = plt.subplots(figsize=(7, 4.5)); rng = np.random.default_rng(0)
for i, c in enumerate(order): ax.scatter(g[c], i + rng.uniform(-0.25, 0.25, len(g)), s=6, alpha=0.5, color="#c55a11")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("native SHAP value (effect on risk)")
ax.set_title("Native SHAP summary, all 430 employees (one dot = one employee)"); fig.tight_layout(); fig.savefig(F / "task14_native_shap_summary.png", dpi=150); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 3.5)); cmp.top3_overlap.value_counts().sort_index().plot.bar(ax=ax, color="#2e75b6")
ax.set_xlabel("shared reasons out of 3 (Monte Carlo vs native SHAP)"); ax.set_ylabel("employees"); ax.set_title("Agreement of top-3 reasons"); fig.tight_layout(); fig.savefig(F / "task14_native_shap_agreement.png", dpi=150); plt.close(fig)
ff = T / "shap_full430_feature_values.csv"
if ff.exists():
    fv = pd.read_csv(ff).set_index("employee_id").drop(columns=["base_value", "prediction"]); assert len(fv) == 430
    top = fv.abs().mean().sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(7, 5.2)); ax.barh(top.index, top.values, color="#1f4e79"); ax.set_xlabel("mean |native SHAP value| (probability points)")
    ax.set_title("Native SHAP, all 430 employees: top 15 individual features"); fig.tight_layout(); fig.savefig(F / "task14_native_shap_feature_importance.png", dpi=150); plt.close(fig)
    order = top.index; fig, ax = plt.subplots(figsize=(7, 5.2)); rng = np.random.default_rng(0)
    for i, c in enumerate(order): ax.scatter(fv[c], i + rng.uniform(-0.25, 0.25, len(fv)), s=6, alpha=0.5, color="#c55a11")
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.axvline(0, color="grey", lw=0.8); ax.set_xlabel("native SHAP value (effect on risk)")
    ax.set_title("Native SHAP feature-level summary (one dot = one employee)"); fig.tight_layout(); fig.savefig(F / "task14_native_shap_feature_summary.png", dpi=150); plt.close(fig)
    print("wrote 2 feature-level figures")
else: print("feature-level CSV not found; rerun scripts/run_shap_full430.py first")
print("wrote group-level figures to", F)
