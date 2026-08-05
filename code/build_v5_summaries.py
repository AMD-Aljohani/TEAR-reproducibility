#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
FIG = ROOT / 'figures'

raw = pd.read_csv(RES / 'correct_spec_raw.csv')
raw['q_face'] = raw['q_hat'] <= 0.0200001
raw['gamma_face'] = raw['gamma_hat'] <= 1e-8
raw['any_face'] = raw['q_face'] | raw['gamma_face']
rows=[]
for (scenario,n,wall),g in raw.groupby(['scenario','n','wall'], sort=True):
    interior = ~g['any_face']
    def mcse(x):
        p=float(np.mean(x)); return math.sqrt(max(p*(1-p),0.0)/len(x))
    rows.append({
        'scenario':scenario,'n':int(n),'wall':wall,'reps':len(g),
        'q_face_rate':g.q_face.mean(),'gamma_face_rate':g.gamma_face.mean(),
        'any_face_rate':g.any_face.mean(),
        'beta1_coverage':g.cover_beta1.mean(),'beta1_coverage_mcse':mcse(g.cover_beta1),
        'gamma_coverage':g.cover_gamma.mean(),'gamma_coverage_mcse':mcse(g.cover_gamma),
        'interior_reps':int(interior.sum()),
        'beta1_coverage_interior':g.loc[interior,'cover_beta1'].mean() if interior.any() else np.nan,
        'gamma_coverage_interior':g.loc[interior,'cover_gamma'].mean() if interior.any() else np.nan,
    })
boundary = pd.DataFrame(rows)
boundary.to_csv(RES/'correct_spec_boundary_summary.csv',index=False)

end = pd.concat([
    pd.read_csv(RES/'endpoint_summary.csv'),
    pd.read_csv(RES/'endpoint_aligned_baseline_summary.csv')
], ignore_index=True)
end['relative_c_bias']=(end['mean_c_feasible']-end['mean_c_oracle'])/end['mean_c_oracle']
end['oracle_coverage_mcse']=np.sqrt(end['oracle_95_coverage']*(1-end['oracle_95_coverage'])/end['reps'])
end['feasible_coverage_mcse']=np.sqrt(end['feasible_95_coverage']*(1-end['feasible_95_coverage'])/end['feasible_reps'])
end.to_csv(RES/'endpoint_finite_sample_audit.csv',index=False)

# Rename labels in the actual plug-in figure and regenerate from the locked CSV.
plug=pd.read_csv(RES/'plugin_invariance_summary.csv')
labels={'baseline':'baseline','low_persistence':'low persistence','near_active':'boundary-stress'}
plt.figure(figsize=(7.2,4.8))
for scenario,g in plug.groupby('scenario'):
    g=g.sort_values('n')
    plt.plot(g['n'],g['mean_abs_sqrt_n_delta_gamma'],marker='o',label=labels.get(scenario,scenario))
plt.xscale('log')
plt.xlabel('n')
plt.ylabel(r'Mean $|\sqrt{n}(\widehat\gamma(\widehat K)-\widehat\gamma(K))|$')
plt.title('Known-wall versus estimated-wall dynamic estimates')
plt.legend()
plt.tight_layout()
plt.savefig(FIG/'Fig3_plugin_invariance.pdf')
plt.savefig(FIG/'Fig3_plugin_invariance.png',dpi=180)
plt.close()

# Machine-readable application caveat record.
app = pd.DataFrame([
    {'item':'population_screen','status':'locked summary supplied','caveat':'Raw trace is external; code downloads or accepts the official fastStorage archive.'},
    {'item':'VM 718 inference','status':'descriptive only in V5','caveat':'The focal series was selected partly by lag correlation and observed maximum; no post-selection p-value is interpreted.'},
    {'item':'continuous likelihood','status':'known limitation','caveat':'Telemetry rounding and discreteness can contribute to PIT defects.'},
])
app.to_csv(RES/'application_reproducibility_caveats.csv',index=False)


from build_v5_endpoint_bounds import main as build_v5_endpoint_bounds
build_v5_endpoint_bounds()
print('Wrote v5 summaries, exact endpoint-bound tables, and plug-in figure.')
