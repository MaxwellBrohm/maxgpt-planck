"""A11. Leave-one-seed-out robustness of the three ruled readings (descriptive; the rule reads all 5 seeds):
Q-chat condition (a)'s ratios, the Q-H5 label, and the knowledge label."""
from common import *
from a4_qchat import arm_arrays
from a3_qh5 import label
import a5_know_cost_device as K

Gt, NT = arm_arrays("G", "tf"); Ct, _ = arm_arrays("C", "tf")
G2, _ = arm_arrays("G", "tf2"); C2, _ = arm_arrays("C", "tf2")
M = {(a, kd): matrix(a, "big", "plain", kd, ["H5"])["H5"] for a in ("C", "P", "e004w", "e005w") for kd in ("LIK", "GEN")}
kG, kC = K.kmat("G"), K.kmat("C")
keep = np.array([i not in set(K.EXP["strict"]) for i in range(kG.shape[1])])
for d in range(5):
    r = [s for s in range(5) if s != d]
    a_ratio = Gt[r].sum() <= 0.5 * Ct[r].sum() and G2[r].sum() <= 0.5 * C2[r].sum()
    ci = boot_conv(Gt[r], Ct[r], NT[r])
    dd = {kd: boot_pair(M[("P", kd)][r] - M[("C", kd)][r]) for kd in ("LIK", "GEN")}
    pl = {kd: boot_pair(M[("P", kd)][r] - M[("e004w", kd)][r]) for kd in ("LIK", "GEN")}
    dev = M[("C", "LIK")][r].mean() - M[("e005w", "LIK")][r].mean()
    k = boot_pair(kG[r][:, keep] - kC[r][:, keep])
    print(f"without s{d + 1}: chat (a) ratio {a_ratio} (TF G {Gt[r].sum()} vs half C {0.5 * Ct[r].sum()}; CI high"
          f" {ci[2]:+.3f}) | Q-H5 {label(dd['LIK'], dd['GEN'], pl['LIK'], pl['GEN'], dev)} (LIK {fmt(dd['LIK'])})"
          f" | K {fmt(k)}")
