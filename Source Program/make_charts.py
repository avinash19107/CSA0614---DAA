import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (9, 5.5), 'figure.dpi': 150, 'font.size': 11,
    'axes.titlesize': 13, 'axes.titleweight': 'bold', 'axes.labelsize': 11,
    'legend.fontsize': 10,
})
COL_PLAIN = '#C44E52'
COL_PRUNE = '#4C72B0'

df = pd.read_csv('results.csv')
# average over seed trials
agg = df.groupby(['density_class', 'n_courses', 'algorithm'], as_index=False).agg(
    time_sec=('time_sec', 'mean'),
    steps=('steps', 'mean'),
    conflicts=('conflicts', 'mean'),
    peak_memory_kb=('peak_memory_kb', 'mean'),
    solved_rate=('solved', 'mean'),
)

density_order = ['sparse', 'medium', 'dense']

# ---------- Chart 1: Execution time vs n_courses, one panel per density ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, dens in zip(axes, density_order):
    sub = agg[agg.density_class == dens]
    for algo, color, marker in [('plain', COL_PLAIN, 'o'), ('pruning', COL_PRUNE, 's')]:
        d = sub[sub.algorithm == algo].sort_values('n_courses')
        ax.plot(d.n_courses, d.time_sec, marker=marker, color=color,
                label='Plain Backtracking' if algo == 'plain' else 'Pruning-Enhanced (MCV+FC)',
                linewidth=2)
    ax.set_yscale('log')
    ax.set_title(f'{dens.capitalize()} graphs')
    ax.set_xlabel('Number of courses (n)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel('Execution time (seconds, log scale)')
axes[0].legend(loc='upper left')
fig.suptitle('Execution Time vs Problem Size (mean of 3 seeds)', fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('chart1_time_vs_n.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------- Chart 2: Backtracking steps vs n_courses ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, dens in zip(axes, density_order):
    sub = agg[agg.density_class == dens]
    for algo, color, marker in [('plain', COL_PLAIN, 'o'), ('pruning', COL_PRUNE, 's')]:
        d = sub[sub.algorithm == algo].sort_values('n_courses')
        ax.plot(d.n_courses, d.steps, marker=marker, color=color,
                label='Plain Backtracking' if algo == 'plain' else 'Pruning-Enhanced (MCV+FC)',
                linewidth=2)
    ax.set_yscale('log')
    ax.set_title(f'{dens.capitalize()} graphs')
    ax.set_xlabel('Number of courses (n)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
axes[0].set_ylabel('Backtracking steps / assignment attempts (log scale)')
axes[0].legend(loc='upper left')
fig.suptitle('Search-Space Exploration (Backtracking Steps) vs Problem Size', fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('chart2_steps_vs_n.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------- Chart 3: Peak memory vs n_courses (dense only, worst case) ----------
fig, ax = plt.subplots(figsize=(8, 5.5))
sub = agg[agg.density_class == 'dense']
for algo, color, marker in [('plain', COL_PLAIN, 'o'), ('pruning', COL_PRUNE, 's')]:
    d = sub[sub.algorithm == algo].sort_values('n_courses')
    ax.plot(d.n_courses, d.peak_memory_kb, marker=marker, color=color,
            label='Plain Backtracking' if algo == 'plain' else 'Pruning-Enhanced (MCV+FC)',
            linewidth=2)
ax.set_xlabel('Number of courses (n)')
ax.set_ylabel('Peak traced memory (KB)')
ax.set_title('Peak Memory Utilisation vs Problem Size (Dense Graphs)')
ax.legend(loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('chart3_memory_vs_n.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------- Chart 4: Scalability / solved rate within time limit (dense) ----------
import numpy as np
fig, ax = plt.subplots(figsize=(9, 5.5))
sub = agg[agg.density_class == 'dense']
width = 0.35
ns = sorted(sub.n_courses.unique())
plain_rate = [sub[(sub.algorithm == 'plain') & (sub.n_courses == n)].solved_rate.values[0] * 100 for n in ns]
prune_rate = [sub[(sub.algorithm == 'pruning') & (sub.n_courses == n)].solved_rate.values[0] * 100 for n in ns]
x = np.arange(len(ns))
ax.bar(x - width/2, plain_rate, width=width, color=COL_PLAIN, label='Plain Backtracking')
ax.bar(x + width/2, prune_rate, width=width, color=COL_PRUNE, label='Pruning-Enhanced (MCV+FC)')
ax.set_xticks(x)
ax.set_xticklabels(ns)
ax.set_ylim(0, 115)
ax.set_xlabel('Number of courses (n)')
ax.set_ylabel('Runs solved within 8s time limit (%)')
ax.set_title('Scalability: Success Rate Within Time Limit (Dense Graphs)')
ax.legend(loc='lower left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig('chart4_scalability.png', dpi=150, bbox_inches='tight')
plt.close()

# ---------- Summary table CSV for the report ----------
summary = agg.copy()
summary['time_sec'] = summary['time_sec'].round(5)
summary['steps'] = summary['steps'].round(1)
summary['peak_memory_kb'] = summary['peak_memory_kb'].round(2)
summary['solved_rate'] = (summary['solved_rate'] * 100).round(1)
summary.to_csv('summary_table.csv', index=False)

print("Charts and summary table generated.")
print(summary.to_string(index=False))
