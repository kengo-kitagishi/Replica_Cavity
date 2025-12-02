import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. 2-community SBM の生成
# =====================================================

def generate_sbm_2comm(N, c, eps, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    tau = rng.choice([-1, 1], size=N)

    c_in = c * (1.0 + eps)
    c_out = c * (1.0 - eps)
    p_in = c_in / N
    p_out = c_out / N

    A = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(i + 1, N):
            if tau[i] == tau[j]:
                p_ij = p_in
            else:
                p_ij = p_out
            if rng.random() < p_ij:
                A[i, j] = 1
                A[j, i] = 1

    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]
    return A, neighbors, tau


# =====================================================
# 2. Nishimori Ising (Decelle) + MC
# =====================================================

def mc_gibbs_sbm_decelle(
    A,
    c,
    eps,
    n_sweeps_therm=200,
    n_sweeps_meas=200,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    N = A.shape[0]
    sigma = rng.choice([-1, 1], size=N)
    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]

    c_in = c * (1.0 + eps)
    c_out = c * (1.0 - eps)
    p_same = c_in / N
    p_diff = c_out / N

    eps_p = 1e-12
    p_same = np.clip(p_same, eps_p, 1.0 - eps_p)
    p_diff = np.clip(p_diff, eps_p, 1.0 - eps_p)

    J_E = 0.5 * np.log(p_same / p_diff)
    J_0 = 0.5 * np.log((1.0 - p_same) / (1.0 - p_diff))

    M = sigma.sum()

    def local_field(i):
        S_i = sigma[neighbors[i]].sum()
        return J_E * S_i + J_0 * (M - sigma[i] - S_i)

    for _ in range(n_sweeps_therm * N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2.0 * sigma[i] * h
        if dE <= 0.0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2 * sigma[i]

    for _ in range(n_sweeps_meas * N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2.0 * sigma[i] * h
        if dE <= 0.0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2 * sigma[i]

    return sigma


# =====================================================
# 3. Non-backtracking spectral (2次元パワー法で第2固有ベクトル)
# =====================================================

def nb_spectral_2comm(A, neighbors, n_iter=200, rng=None):
    """
    非バックトラック行列 B の「上位2固有ベクトル」を
    2次元パワー法で求め、そのうち第2固有ベクトルを使って
    ノードのラベルを推定する。

    実際の B は 2m×2m の行列だが、
    明示的に作らず「有向辺メッセージ」の形で作用させる。
    """
    if rng is None:
        rng = np.random.default_rng()
    N = A.shape[0]

    # 有向辺に index を振る
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in neighbors[i]:
            edge_index[(i, j)] = idx
            idx += 1
    n_dir_edges = idx

    # 2本の初期ベクトル（ランダム）
    x = rng.normal(scale=1.0, size=n_dir_edges)
    y = rng.normal(scale=1.0, size=n_dir_edges)

    def B_mul(v):
        """ B v を v 上の有向辺表現で計算 """
        out = np.zeros_like(v)
        for (i, j), e_idx in edge_index.items():
            s = 0.0
            for k in neighbors[i]:
                if k == j:
                    continue
                e_in = edge_index[(k, i)]
                s += v[e_in]
            out[e_idx] = s
        return out

    # 2次元パワー法
    for it in range(n_iter):
        x_new = B_mul(x)
        y_new = B_mul(y)

        # Gram-Schmidt 正規直交化
        # まず x を正規化（最大固有ベクトル方向）
        nx = np.linalg.norm(x_new)
        if nx > 0:
            x_new /= nx
        # y から x 成分を引いて直交化
        proj = np.dot(x_new, y_new)
        y_new = y_new - proj * x_new
        ny = np.linalg.norm(y_new)
        if ny > 0:
            y_new /= ny

        x, y = x_new, y_new

    # y が「第2固有ベクトル」に対応するので、それをノードに落とす
    node_score = np.zeros(N)
    for (i, j), e_idx in edge_index.items():
        node_score[j] += y[e_idx]

    sigma_hat = np.where(node_score >= 0, 1, -1)
    return sigma_hat


# =====================================================
# 4. オーバーラップと eps スイープ
# =====================================================

def overlap(tau, sigma_hat):
    m = np.mean(tau * sigma_hat)
    return abs(m)


def sweep_eps_nb_mc(
    N=5000,
    c=16.0,
    eps_list=None,
    n_realizations=5,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    if eps_list is None:
        eps_c = 1.0 / np.sqrt(c)
        eps_list = np.linspace(0.0, 2.0 * eps_c, 15)

    m_nb_list = []
    m_mc_list = []

    for eps in eps_list:
        ms_nb = []
        ms_mc = []

        for r in range(n_realizations):
            A, neighbors, tau = generate_sbm_2comm(N, c, eps, rng=rng)

            sigma_nb = nb_spectral_2comm(A, neighbors, n_iter=200, rng=rng)
            ms_nb.append(overlap(tau, sigma_nb))

            sigma_mc = mc_gibbs_sbm_decelle(
                A, c, eps,
                n_sweeps_therm=200,
                n_sweeps_meas=200,
                rng=rng,
            )
            ms_mc.append(overlap(tau, sigma_mc))

        m_nb_mean = np.mean(ms_nb)
        m_mc_mean = np.mean(ms_mc)

        m_nb_list.append(m_nb_mean)
        m_mc_list.append(m_mc_mean)

        lam = c * eps**2
        print(f"eps={eps:.4f}, lambda={lam:.4f} : "
              f"<m_NB>={m_nb_mean:.3f}, "
              f"<m_MC>={m_mc_mean:.3f}")

    return np.array(eps_list), np.array(m_nb_list), np.array(m_mc_list)


# =====================================================
# 5. メイン：λ と c_out/c_in で図を出す
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 5000
    c = 16.0
    eps_c = 1.0 / np.sqrt(c)
    eps_list = np.linspace(0.0, 2.0 * eps_c, 15)

    eps_arr, m_nb, m_mc = sweep_eps_nb_mc(
        N=N,
        c=c,
        eps_list=eps_list,
        n_realizations=3,
        rng=rng,
    )

    lam_arr = c * eps_arr**2
    r_arr = (1.0 - eps_arr) / (1.0 + eps_arr)
    r_c = (np.sqrt(c) - 1.0) / (np.sqrt(c) + 1.0)

    # --- 図1：overlap vs λ ---
    plt.figure(figsize=(6, 4))
    plt.plot(lam_arr, m_nb, "o-", label="Non-backtracking (2nd eig)")
    plt.plot(lam_arr, m_mc, "s--", label="MC (Nishimori)")
    plt.axvline(1.0, color="k", linestyle="--", label=r"$\lambda_c = 1$")
    plt.xlabel(r"$\lambda = c \varepsilon^2$")
    plt.ylabel("overlap m")
    plt.title(f"2-community SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()

    # --- 図2：overlap vs c_out/c_in ---
    plt.figure(figsize=(6, 4))
    plt.plot(r_arr, m_nb, "o-", label="Non-backtracking (2nd eig)")
    plt.plot(r_arr, m_mc, "s--", label="MC (Nishimori)")
    plt.axvline(r_c, color="k", linestyle="--", label=r"$r_c$ (KS)")
    plt.xlabel(r"$c_{\rm out}/c_{\rm in}$")
    plt.ylabel("overlap m")
    plt.title(f"2-community SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(6, 4))
    plt.plot(lam_arr, t_bp, "o-", color="purple")
    plt.axvline(1.0, color="k", linestyle="--", label=r"$\lambda_c$")
    plt.xlabel(r"$\lambda = c \varepsilon^2$")
    plt.ylabel("BP convergence iterations")
    plt.title("BP convergence time (Decelle BP, seeded)")
    plt.legend()
    plt.tight_layout()

    plt.show()