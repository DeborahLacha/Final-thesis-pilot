"""Pilot study: removal-operator ranking reversals in a linear-Gaussian setting.

Model  : f(x) = beta^T x, X ~ N(0, Sigma_rho) (equicorrelated), d features.
Methods: interventional SHAP (exact, phi_j = beta_j x_j), conditional SHAP
         (exact, Gaussian conditioning over all 2^d subsets), conditional
         leave-one-out occlusion, random ordering.
Operators for deletion: marginal (interventional) and conditional removal.
Score  : oriented deletion AUC (mean over k of s*v(S_k)), lower is better.
"""
import itertools, json, sys
import numpy as np
from scipy import stats

def cond_matrices(Sigma):
    d = Sigma.shape[0]
    subsets = list(range(2 ** d))
    C, D = {}, {}
    for m in subsets:
        S = [j for j in range(d) if m >> j & 1]
        R = [j for j in range(d) if not m >> j & 1]
        Cm = np.zeros((d, d)); Dm = np.zeros((d, d))
        for j in S:
            Cm[j, j] = 1; Dm[j, j] = 1
        if S and R:
            Cm[np.ix_(R, S)] = Sigma[np.ix_(R, S)] @ np.linalg.inv(Sigma[np.ix_(S, S)])
        C[m], D[m] = Cm, Dm
    return C, D

def shapley_matrix(W, d):
    """W[m] = weight vector w_S s.t. v(x,S) = x @ w_S. Returns M (d x d), phi = x @ M."""
    from math import factorial
    M = np.zeros((d, d))
    for j in range(d):
        for m in range(2 ** d):
            if m >> j & 1:
                continue
            s = bin(m).count("1")
            wt = factorial(s) * factorial(d - s - 1) / factorial(d)
            M[:, j] += wt * (W[m | (1 << j)] - W[m])
    return M

def run(d=6, rhos=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95), trials=200, n=500, seed=0):
    rng = np.random.default_rng(seed)
    full = 2 ** d - 1
    out = []
    for rho in rhos:
        Sigma = (1 - rho) * np.eye(d) + rho * np.ones((d, d))
        Cm, Dm = cond_matrices(Sigma)
        L = np.linalg.cholesky(Sigma)
        rev, home_m, home_c, taus, prac, gM, gC = 0, 0, 0, [], 0, [], []
        for _ in range(trials):
            beta = rng.standard_normal(d)
            WC = {m: Cm[m].T @ beta for m in Cm}   # v_C(x,S) = x @ WC[m]
            WM = {m: Dm[m].T @ beta for m in Dm}   # v_M(x,S) = x @ WM[m]
            X = rng.standard_normal((n, d)) @ L.T
            s = np.sign(X @ beta)
            phis = {
                "IntSHAP": X @ np.diag(beta),
                "CondSHAP": X @ shapley_matrix(WC, d),
                "CondLOO": np.stack([X @ (WC[full] - WC[full ^ (1 << j)]) for j in range(d)], 1),
                "Random": rng.standard_normal((n, d)),
            }
            scores = {}
            for name, phi in phis.items():
                order = np.argsort(-(s[:, None] * phi), axis=1)
                masks = np.full(n, full)
                accM = s * (X @ WM[full]); accC = s * (X @ WC[full])
                for k in range(d):
                    masks = masks ^ (1 << order[:, k])
                    vm = np.array([X[i] @ WM[masks[i]] for i in range(n)])
                    vc = np.array([X[i] @ WC[masks[i]] for i in range(n)])
                    accM = accM + s * vm; accC = accC + s * vc
                scores[name] = (accM / (d + 1), accC / (d + 1))
            dM = scores["IntSHAP"][0] - scores["CondSHAP"][0]
            dC = scores["IntSHAP"][1] - scores["CondSHAP"][1]
            pM = stats.ttest_1samp(dM, 0).pvalue if np.std(dM) > 1e-12 else 1.0
            pC = stats.ttest_1samp(dC, 0).pvalue if np.std(dC) > 1e-12 else 1.0
            if pM < .05 and pC < .05 and np.sign(dM.mean()) != np.sign(dC.mean()):
                rev += 1
            if pM < .05 and dM.mean() < 0: home_m += 1      # IntSHAP wins under marginal
            if pC < .05 and dC.mean() > 0: home_c += 1      # CondSHAP wins under conditional
            # practical size: difference relative to the gap between Random and best method
            def norm(op, diff):
                means = [scores[k][op].mean() for k in scores]
                return diff / (scores["Random"][op].mean() - min(means))
            nM, nC = norm(0, -dM.mean()), norm(1, dC.mean())   # >0 = home method wins
            gM.append(nM); gC.append(nC)
            if nM > .05 and nC > .05: prac += 1
            names = list(scores)
            rM = [scores[k][0].mean() for k in names]; rC = [scores[k][1].mean() for k in names]
            taus.append(stats.kendalltau(rM, rC).statistic)
        row = dict(rho=rho, reversal=rev / trials, int_wins_marg=home_m / trials,
                   cond_wins_cond=home_c / trials, mean_tau=float(np.nanmean(taus)),
                   practical_reversal=prac / trials, margin_marg=float(np.median(gM)),
                   margin_cond=float(np.median(gC)))
        print(row, flush=True)
        out.append(row)
    return out

if __name__ == "__main__":
    res = run(trials=int(sys.argv[1]) if len(sys.argv) > 1 else 200)
    json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else "pilot_results.json", "w"), indent=1)
