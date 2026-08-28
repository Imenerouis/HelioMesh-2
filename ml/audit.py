"""Audit script — no modifications to model or dataset."""
import pandas as pd
import numpy as np

df = pd.read_csv('ml/sequences.csv')
label = df['label']
kp_t5  = df['kp_index_t5']
orb_t5 = df['orbit_deviation_t5']
pwr_t5 = df['power_output_t5']

def snap_label(kp, orb, pwr):
    if kp > 6.0 or orb > 1.5 or pwr < 10.0:
        return 1
    if kp > 4.0 or orb > 0.8 or pwr < 40.0:
        return 1
    return 0

t5_pred = pd.Series([
    snap_label(k, o, p)
    for k, o, p in zip(kp_t5, orb_t5, pwr_t5)
], index=df.index)

agree    = (t5_pred == label).mean()
disagree = 1.0 - agree

print("=" * 60)
print("  AUDIT — t5 snapshot rule vs future label (all 12,000)")
print("=" * 60)
print(f"  t5 rule agrees with future label : {agree:.4f} ({int(agree*len(df))}/{len(df)})")
print(f"  Disagreement (true temporal cases): {disagree:.4f} ({int(disagree*len(df))} sequences)")
print()

# Cases where temporal reasoning matters
diff_mask = (t5_pred != label)
diff_df = df[diff_mask].copy()
diff_df['t5_pred']   = t5_pred[diff_mask]
diff_df['true_label'] = label[diff_mask]

rising = diff_df[diff_df['t5_pred'] == 0]   # NOMINAL now, CRITICAL ahead
falling = diff_df[diff_df['t5_pred'] == 1]  # CRITICAL now, NOMINAL ahead

print(f"Cases: t5=NOMINAL but future=CRITICAL  (temporal predictor earns its value): {len(rising)}")
if len(rising) > 0:
    print(f"  avg kp_t0        = {rising['kp_index_t0'].mean():.3f}")
    print(f"  avg kp_t5        = {rising['kp_index_t5'].mean():.3f}")
    print(f"  avg delta_kp     = {rising['delta_kp_index'].mean():.3f}")
    print(f"  avg kp_t5 range  min={rising['kp_index_t5'].min():.2f} max={rising['kp_index_t5'].max():.2f}")

print()
print(f"Cases: t5=CRITICAL but future=NOMINAL  (storm calmed down): {len(falling)}")
if len(falling) > 0:
    print(f"  avg kp_t5        = {falling['kp_index_t5'].mean():.3f}")
    print(f"  avg delta_kp     = {falling['delta_kp_index'].mean():.3f}")

print()
print("=" * 60)
print("  kp_t5 threshold analysis (full dataset)")
print("=" * 60)
for thresh in [3.0, 3.5, 4.0, 4.5, 5.0]:
    pred = (kp_t5 > thresh).astype(int)
    acc  = (pred == label).mean()
    tp   = ((pred == 1) & (label == 1)).sum()
    fn   = ((pred == 0) & (label == 1)).sum()
    fp   = ((pred == 1) & (label == 0)).sum()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec   = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  kp_t5 > {thresh:.1f}  acc={acc:.3f}  prec={prec:.3f}  recall={recall:.3f}")

print()
print("=" * 60)
print("  CRITICAL rate by kp_t5 bin")
print("=" * 60)
bins  = [0, 2, 3, 4, 5, 6, 9]
blabs = ['0-2', '2-3', '3-4', '4-5', '5-6', '6-9']
df['kp_bin'] = pd.cut(kp_t5, bins=bins, labels=blabs)
grp = df.groupby('kp_bin')['label'].agg(['mean', 'count'])
grp.columns = ['critical_rate', 'count']
print(grp.to_string())

print()
print("=" * 60)
print("  power_output across timesteps (sail is FIXED per sequence)")
print("=" * 60)
for i in range(6):
    col = f'power_output_t{i}'
    diff = abs(df[col] - df['power_output_t0']).max()
    print(f"  {col} vs t0: max_diff = {diff:.8f}")

print()
print("=" * 60)
print("  orbit_deviation — is it deterministic from kp+wind+sail?")
print("=" * 60)
# orbit_dev = (sail/90)*(1+kp*0.1)*(wind/400)*0.5
# This is fully deterministic given kp, sail, wind
recomputed = ((df['kp_index_t5'] * 0 + df['kp_index_t5'].apply(lambda x: x)) / 9)
# Just check variance of orbit_dev_t5 vs kp_t5
corr = df['orbit_deviation_t5'].corr(df['kp_index_t5'])
print(f"  corr(orbit_deviation_t5, kp_index_t5) = {corr:.4f}")
corr2 = df['orbit_deviation_t5'].corr(df['solar_wind_speed_t5'])
print(f"  corr(orbit_deviation_t5, wind_t5)     = {corr2:.4f}")

print()
print("=" * 60)
print("  SHORTCUT LEARNING CHECK")
print("  Can CRITICAL_AHEAD be derived directly from t5 features?")
print("=" * 60)
# How many sequences where label=1 already have kp_t5 > 4?
crit_kp_already = ((label == 1) & (kp_t5 > 4.0)).sum()
crit_total       = (label == 1).sum()
print(f"  Of {crit_total} CRITICAL_AHEAD sequences:")
print(f"    {crit_kp_already} ({crit_kp_already/crit_total:.1%}) already have kp_t5 > 4.0")
print(f"    {crit_total-crit_kp_already} ({(crit_total-crit_kp_already)/crit_total:.1%}) need temporal reasoning (kp_t5 <= 4 but label=1)")
print()
nom_kp_safe = ((label == 0) & (kp_t5 <= 4.0)).sum()
nom_total    = (label == 0).sum()
print(f"  Of {nom_total} NOMINAL_AHEAD sequences:")
print(f"    {nom_kp_safe} ({nom_kp_safe/nom_total:.1%}) have kp_t5 <= 4.0 (consistent with nominal future)")
print(f"    {nom_total-nom_kp_safe} ({(nom_total-nom_kp_safe)/nom_total:.1%}) have kp_t5 > 4 but label=0 (temporal: storm passed or didn't escalate)")
