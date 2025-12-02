import numpy as np

# =====================================================
# 1. 2-community SBM 生成
# =====================================================
def generate_sbm_2comm(N, c, eps, rng=None):
    if rng is None: rng = np.random.default_rng()
    tau = rng.choice([-1, 1], size=N)
    c_in, c_out = c*(1+eps), c*(1-eps)
    p_in, p_out = c_in/N, c_out/N
    A = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(i+1, N):
            p = p_in if tau[i]==tau[j] else p_out
            if rng.random() < p: A[i,j]=A[j,i]=1
    neighbors = [np.where(A[i]>0)[0] for i in range(N)]
    return A, neighbors, tau

# =====================================================
# 2. Nishimori MC
# =====================================================
def mc_gibbs_sbm_decelle(A, c, eps, n_sweeps_therm=200, n_sweeps_meas=200, rng=None):
    if rng is None: rng = np.random.default_rng()
    N = A.shape[0]
    sigma = rng.choice([-1,1], size=N)
    neighbors = [np.where(A[i]>0)[0] for i in range(N)]
    c_in, c_out = c*(1+eps), c*(1-eps)
    p_same, p_diff = c_in/N, c_out/N
    eps_p = 1e-12
    p_same, p_diff = np.clip(p_same, eps_p, 1-eps_p), np.clip(p_diff, eps_p, 1-eps_p)
    J_E = 0.5*np.log(p_same/p_diff)
    J_0 = 0.5*np.log((1-p_same)/(1-p_diff))
    M = sigma.sum()
    def local_field(i):
        S_i = sigma[neighbors[i]].sum()
        return J_E*S_i + J_0*(M - sigma[i] - S_i)
    for _ in range(n_sweeps_therm*N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2*sigma[i]*h
        if dE <= 0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2*sigma[i]
    for _ in range(n_sweeps_meas*N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2*sigma[i]*h
        if dE <= 0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2*sigma[i]
    return sigma

# =====================================================
# 3. NBスペクトル
# =====================================================
def nb_spectral_2comm(A, neighbors, n_iter=200, rng=None):
    if rng is None: rng = np.random.default_rng()
    N = A.shape[0]
    edge_index, idx = {}, 0
    for i in range(N):
        for j in neighbors[i]:
            edge_index[(i,j)] = idx
            idx += 1
    n_dir_edges = idx
    x, y = rng.normal(size=n_dir_edges), rng.normal(size=n_dir_edges)
    def B_mul(v):
        out = np.zeros_like(v)
        for (i,j), e_idx in edge_index.items():
            out[e_idx] = sum(v[edge_index[(k,i)]] for k in neighbors[i] if k!=j)
        return out
    # 2次元パワー法
    for _ in range(n_iter):
        x_new, y_new = B_mul(x), B_mul(y)
        nx = np.linalg.norm(x_new)
        if nx>0: x_new/=nx
        proj = np.dot(x_new,y_new)
        y_new -= proj*x_new
        ny = np.linalg.norm(y_new)
        if ny>0: y_new/=ny
        x,y = x_new, y_new
    node_score = np.zeros(N)
    for (i,j), e_idx in edge_index.items():
        node_score[j] += y[e_idx]
    # NBギャップ
    sorted_vals = np.sort(y)[::-1]
    gap = sorted_vals[0]-sorted_vals[1] if len(sorted_vals)>1 else 0.0
    sigma_hat = np.where(node_score>=0,1,-1)
    return sigma_hat, node_score, gap

# =====================================================
# 4. BP (メッセージ収束)
# =====================================================
def bp_sbm_decelle_messages(A, neighbors, c, eps, max_iter=200, tol=1e-6):
    N = A.shape[0]
    c_in, c_out = c*(1+eps), c*(1-eps)
    p_in, p_out = c_in/N, c_out/N
    edge_list = [(i,j) for i in range(N) for j in neighbors[i]]
    E = len(edge_list)
    m = np.full(E,0.5)
    m_new = np.zeros(E)
    edge_index = {e:idx for idx,e in enumerate(edge_list)}
    def update_one(i,j):
        idx = edge_index[(i,j)]
        prod = 0.0
        for k in neighbors[i]:
            if k==j: continue
            idx2 = edge_index[(k,i)]
            x = m[idx2]
            llr = np.log((p_in*x+p_out*(1-x))/(p_in*(1-x)+p_out*x))
            prod += llr
        v = np.exp(prod)
        return v/(1+v)
    norms = []
    for it in range(max_iter):
        for (i,j) in edge_list:
            m_new[edge_index[(i,j)]] = update_one(i,j)
        diff = np.max(np.abs(m_new-m))
        norms.append(diff)
        m[:] = m_new
        if diff<tol: break
    return norms

# =====================================================
# 5. overlap 計算
# =====================================================
def overlap(tau, sigma_hat):
    return abs(np.mean(tau*sigma_hat))

# =====================================================
# 6. sweep
# =====================================================
def sweep_eps_all(N=1000, c=16, eps_list=None, n_realizations=2, rng=None):
    if rng is None: rng=np.random.default_rng()
    if eps_list is None: eps_list=np.linspace(0,2/np.sqrt(c),10)
    result = {
        'eps':[], 'lam':[], 'r':[],
        'm_nb':[], 'm_mc':[], 't_bp':[],
        'gap':[], 'bp_norms':[], 'nb_scores':[], 'mc_mags':[]
    }
    for eps in eps_list:
        lam = c*eps**2
        r = (1-eps)/(1+eps)
        m_nb_runs, m_mc_runs, t_bp_runs, gap_runs, bp_norms_runs = [],[],[],[],[]
        nb_scores_all, mc_mags_all = [],[]
        for _ in range(n_realizations):
            A, neighbors, tau = generate_sbm_2comm(N,c,eps,rng)
            sigma_nb, node_score, gap = nb_spectral_2comm(A,neighbors,rng=rng)
            sigma_mc = mc_gibbs_sbm_decelle(A,c,eps,rng=rng)
            norms = bp_sbm_decelle_messages(A,neighbors,c,eps)
            t_bp_runs.append(len(norms))
            bp_norms_runs.append(norms)
            m_nb_runs.append(overlap(tau,sigma_nb))
            m_mc_runs.append(overlap(tau,sigma_mc))
            gap_runs.append(gap)
            nb_scores_all.extend(node_score.tolist())
            mc_mags_all.append(np.mean(sigma_mc*tau))
        result['eps'].append(eps)
        result['lam'].append(lam)
        result['r'].append(r)
        result['m_nb'].append(np.mean(m_nb_runs))
        result['m_mc'].append(np.mean(m_mc_runs))
        result['t_bp'].append(np.mean(t_bp_runs))
        result['gap'].append(np.mean(gap_runs))
        result['bp_norms'].append(bp_norms_runs)
        result['nb_scores'].append(nb_scores_all)
        result['mc_mags'].append(mc_mags_all)
    return result

# =====================================================
# 7. データ保存 (gnuplot 用)
# =====================================================
def save_gnuplot_data(result):
    # overlap vs lam
    np.savetxt("overlap_vs_lambda.dat",
               np.column_stack([result['lam'], result['m_nb'], result['m_mc']]),
               header="lambda m_nb m_mc")
    # overlap vs r
    np.savetxt("overlap_vs_r.dat",
               np.column_stack([result['r'], result['m_nb'], result['m_mc']]),
               header="r m_nb m_mc")
    # BP convergence iterations
    np.savetxt("bp_iterations.dat",
               np.column_stack([result['lam'], result['t_bp']]),
               header="lambda bp_iterations")
    # NB spectral gap
    np.savetxt("nb_gap.dat",
               np.column_stack([result['lam'], result['gap']]),
               header="lambda nb_gap")
    # BP norms vs iteration (平均を列ごとに)
    max_len = max(len(norms[0]) for norms in result['bp_norms'])
    bp_norms_matrix = np.zeros((max_len,len(result['lam'])))
    for j, norms_runs in enumerate(result['bp_norms']):
        norms = np.array(norms_runs[0])
        bp_norms_matrix[:len(norms),j] = norms
    np.savetxt("bp_norms.dat", bp_norms_matrix, header="iterations per eps column")
    # NB node_score distribution
    with open("nb_scores.dat","w") as f:
        for eps, scores in zip(result['eps'], result['nb_scores']):
            for s in scores:
                f.write(f"{eps} {s}\n")
    # MC magnetization distribution
    with open("mc_mags.dat","w") as f:
        for eps, mags in zip(result['eps'], result['mc_mags']):
            for m in mags:
                f.write(f"{eps} {m}\n")

# =====================================================
# 8. main
# =====================================================
if __name__=="__main__":
    rng = np.random.default_rng(0)
    N, c = 1000, 16
    eps_list = np.linspace(0, 2/np.sqrt(c), 10)
    result = sweep_eps_all(N=N, c=c, eps_list=eps_list, n_realizations=2, rng=rng)
    save_gnuplot_data(result)
    print("All gnuplot data saved.")
