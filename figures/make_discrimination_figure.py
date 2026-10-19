import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows = [("CV death, 5 yr",0.901,0.006,0.910,0.009,"+0.010"),("CV death, 10 yr",0.899,0.003,0.907,0.003,"+0.008"),
        ("Cancer death, 5 yr",0.855,0.010,0.847,0.013,"−0.008"),("Cancer death, 10 yr",0.855,0.007,0.848,0.010,"−0.007")]
FG, DH, INK, MUTED = "#2E7FB8", "#D93A2B", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size":11,"axes.edgecolor":"#c9c8c1","axes.labelcolor":INK,
                     "xtick.color":MUTED,"ytick.color":INK})
fig, ax = plt.subplots(figsize=(8.2,3.8), dpi=200); fig.patch.set_facecolor("white")
for i,(lab,fm,fs,dm,ds,d) in enumerate(rows):
    y = len(rows)-1-i
    ax.errorbar(fm,y+0.12,xerr=fs,fmt="o",color=FG,ms=8,lw=2,capsize=0,zorder=3,mec="white",mew=2)
    ax.errorbar(dm,y-0.12,xerr=ds,fmt="s",color=DH,ms=8,lw=2,capsize=0,zorder=3,mec="white",mew=2)
    ax.text(0.927,y,"Δ "+d,va="center",ha="left",color=MUTED,fontsize=10)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1])
ax.set_xlim(0.83,0.94); ax.set_ylim(-0.6,len(rows)-0.4)
ax.set_xlabel("Time-dependent C-index (mean ± SD over 5 CV folds)")
ax.grid(axis="x",color="#ecebe6",lw=1); ax.set_axisbelow(True)
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y",length=0)
ax.plot([],[],"o",color=FG,ms=8,label="Fine-Gray"); ax.plot([],[],"s",color=DH,ms=8,label="DeepHit")
ax.legend(loc="upper left",bbox_to_anchor=(0,1.16),ncol=2,frameon=False)
ax.set_title("Discrimination: Fine-Gray vs DeepHit",loc="left",pad=26,fontsize=12.5,color=INK,fontweight="bold")
fig.tight_layout(); fig.savefig("figures/discrimination_comparison.png",facecolor="white")
