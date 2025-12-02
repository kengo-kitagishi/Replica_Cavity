import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. 2-community SBM の生成 (Decelle+ と同じ sparse スケーリング)
# =====================================================

def generate_sbm_2comm(N, c, eps, rng=None):
    """
    2-community SBM:
        tau_i in {+1,-1}, P(tau_i=+1)=P(tau_i=-1)=1/2
        P(A_ij=1 | tau_i tau_j = +1) = c_in / N
        P(A_ij=1 | tau_i tau_j = -1) = c_out / N

    ここで
        c_in  = c (1 + eps)
        c_out = c (1 - eps)
    となるようにパラメータ化。

    戻り値:
        A        : (N,N) adjacency matrix (0/1, symmetric)
        neighbors: list of neighbor lists
        tau      : planted labels in {+1,-1}
    """
    if rng is None:
        rng = np.random.default_rng()

    # planted labels
    tau = rng.choice([-1, 1], size=N)

    # connection probabilities
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
# 2. BP for SBM (Decelle+2011, 2-community 対称ケース, τでシード)
# =====================================================

def bp_sbm_decelle_2comm(
    neighbors,
    c,
    eps,
    max_iter=200,
    tol=1e-6,
    damping=0.5,
    rng=None,
    tau=None,   # ★ 追加：planted ラベルを渡せるようにする
):
    """
    Decelle+2011 の sparse SBM の BP を，
    2-community, equal-size, symmetric case に簡約した実装。

    メッセージ: m_{i->j} = P_i(+1) - P_i(-1) (cavity)

    更新則:
        h_{i->j} = 1/2 ∑_{k∈∂i\{j}}
                     [ log(1+eps m_{k->i}) - log(1-eps m_{k->i}) ]
        m_{i->j} = tanh(h_{i->j})

    今回は Decelle の density evolution と同じように，
    初期メッセージを真のラベル tau に沿って小さく傾ける：
        m_{i->j}^{(0)} = δ tau_i  (δ≪1)
    とすることで，λ>1 でこのバイアスが増幅されるかを見る。
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(neighbors)

    # 有向辺 i->j に index を振る
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in neighbors[i]:
            edge_index[(i, j)] = idx
            idx += 1
    n_dir_edges = idx

    # -----------------------------
    # 初期メッセージの設定
    # -----------------------------
    if tau is None:
        # 旧来どおり：完全ランダム初期化
        m = 0.01 * rng.normal(size=n_dir_edges)
    else:
        # Decelle の density evolution と同様に
        # planted ラベル tau に沿った小さなバイアスを入れる
        delta = 0.01
        m = np.zeros(n_dir_edges, dtype=float)
        for (i, j), e_idx in edge_index.items():
            m[e_idx] = delta * tau[i]   # i のラベル方向にバイアス

    # 数値安定用のクリップ
    def clip_x(x):
        # eps * m が [-1+eta, 1-eta] に入るようにクリップ
        eta = 1e-6
        return np.clip(x, -1 + eta, 1 - eta)

    converged = False
    for it in range(max_iter):
        m_new = np.zeros_like(m)

        for (i, j), e_idx in edge_index.items():
            s = 0.0
            for k in neighbors[i]:
                if k == j:
                    continue
                e_in = edge_index[(k, i)]
                mk = m[e_in]
                x = clip_x(eps * mk)  # x = eps * m_{k->i}
                s += 0.5 * (np.log(1.0 + x) - np.log(1.0 - x))
            m_new[e_idx] = np.tanh(s)

        # ダンピング
        m_next = damping * m + (1.0 - damping) * m_new

        diff = np.max(np.abs(m_next - m))
        m = m_next

        if diff < tol:
            converged = True
            break

    n_iter = it + 1

    # ノード磁化 m_i（全ての隣接を使う）
    m_node = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in neighbors[i]:
            e_in = edge_index[(k, i)]
            mk = m[e_in]
            x = clip_x(eps * mk)
            s += 0.5 * (np.log(1.0 + x) - np.log(1.0 - x))
        m_node[i] = np.tanh(s)

    return m_node, n_iter, converged


# =====================================================
# 3. Nishimori 線に忠実な Ising への写像 (Decelle+2011)
# =====================================================

def mc_gibbs_sbm_decelle(
    A,
    c,
    eps,
    n_sweeps_therm=200,
    n_sweeps_meas=200,
    rng=None,
):
    """
    Decelle+2011 での SBM のベイズ事後分布を
    Ising Hamiltonian に写像したものを Metropolis で回す（Nishimori 線）。
    """
    if rng is None:
        rng = np.random.default_rng()

    N = A.shape[0]
    sigma = rng.choice([-1, 1], size=N)

    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]

    # SBM パラメータ
    c_in = c * (1.0 + eps)
    c_out = c * (1.0 - eps)
    p_same = c_in / N
    p_diff = c_out / N

    # 数値安定のためクリップ
    eps_p = 1e-12
    p_same = np.clip(p_same, eps_p, 1.0 - eps_p)
    p_diff = np.clip(p_diff, eps_p, 1.0 - eps_p)

    # couplings
    J_E = 0.5 * np.log(p_same / p_diff)
    J_0 = 0.5 * np.log((1.0 - p_same) / (1.0 - p_diff))

    # 全スピン和
    M = sigma.sum()

    def local_field(i):
        S_i = sigma[neighbors[i]].sum()
        return J_E * S_i + J_0 * (M - sigma[i] - S_i)

    # --- 熱平衡化 ---
    for _ in range(n_sweeps_therm * N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2.0 * sigma[i] * h
        if dE <= 0.0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2 * sigma[i]

    # --- 測定フェーズ ---
    for _ in range(n_sweeps_meas * N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2.0 * sigma[i] * h
        if dE <= 0.0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2 * sigma[i]

    return sigma


# =====================================================
# 4. オーバーラップと eps スイープ
# =====================================================

def overlap(tau, sigma_hat):
    m = np.mean(tau * sigma_hat)
    return abs(m)


def sweep_eps_bp_mc_decelle(
    N=5000,
    c=16.0,
    eps_list=None,
    n_realizations=5,
    rng=None,
):
    """
    eps をスイープして、
        - Decelle BP (seeded) の overlap
        - Decelle Nishimori Ising (MC) の overlap
        - BP の収束反復回数
    を測定。
    """
    if rng is None:
        rng = np.random.default_rng()

    if eps_list is None:
        eps_c = 1.0 / np.sqrt(c)
        eps_list = np.linspace(0.0, 2.0 * eps_c, 15)

    m_bp_list = []
    m_mc_list = []
    t_bp_list = []

    for eps in eps_list:
        ms_bp = []
        ms_mc = []
        ts_bp = []

        for r in range(n_realizations):
            # グラフ生成
            A, neighbors, tau = generate_sbm_2comm(N, c, eps, rng=rng)

            # --- BP (Decelle, seeded by tau) ---
            m_node, n_iter, conv = bp_sbm_decelle_2comm(
                neighbors, c, eps,
                max_iter=500,
                tol=1e-5,
                damping=0.5,
                rng=rng,
                tau=tau,   # ★ ここで tau を渡す
            )
            sigma_bp = np.where(m_node >= 0, 1, -1)
            ms_bp.append(overlap(tau, sigma_bp))
            ts_bp.append(n_iter)

            # --- MC (Decelle Nishimori 版) ---
            sigma_mc = mc_gibbs_sbm_decelle(
                A, c, eps,
                n_sweeps_therm=200,
                n_sweeps_meas=200,
                rng=rng,
            )
            ms_mc.append(overlap(tau, sigma_mc))

        m_bp_mean = np.mean(ms_bp)
        m_mc_mean = np.mean(ms_mc)
        t_bp_mean = np.mean(ts_bp)

        m_bp_list.append(m_bp_mean)
        m_mc_list.append(m_mc_mean)
        t_bp_list.append(t_bp_mean)

        lam = c * eps**2
        print(f"eps={eps:.4f}, lambda={lam:.4f} : "
              f"<m_BP>={m_bp_mean:.3f}, "
              f"<m_MC>={m_mc_mean:.3f}, "
              f"<T_BP>={t_bp_mean:.1f}")

    return np.array(eps_list), np.array(m_bp_list), np.array(m_mc_list), np.array(t_bp_list)


# =====================================================
# 5. メイン：λ と c_out/c_in で図を出す
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 5000
    c = 16.0
    eps_c = 1.0 / np.sqrt(c)
    eps_list = np.linspace(0.0, 2.0 * eps_c, 15)

    eps_arr, m_bp, m_mc, t_bp = sweep_eps_bp_mc_decelle(
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
    plt.plot(lam_arr, m_bp, "o-", label="BP (Decelle, seeded)")
    plt.plot(lam_arr, m_mc, "s--", label="MC (Nishimori)")
    plt.axvline(1.0, color="k", linestyle="--", label=r"$\lambda_c = 1$")
    plt.xlabel(r"$\lambda = c \varepsilon^2$")
    plt.ylabel("overlap m")
    plt.title(f"2-community SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()

    # --- 図2：BP 収束時間 vs λ ---
    plt.figure(figsize=(6, 4))
    plt.plot(lam_arr, t_bp, "o-", color="purple")
    plt.axvline(1.0, color="k", linestyle="--", label=r"$\lambda_c$")
    plt.xlabel(r"$\lambda = c \varepsilon^2$")
    plt.ylabel("BP convergence iterations")
    plt.title("BP convergence time (Decelle BP, seeded)")
    plt.legend()
    plt.tight_layout()

    # --- 図3：overlap vs c_out/c_in ---
    plt.figure(figsize=(6, 4))
    plt.plot(r_arr, m_bp, "o-", label="BP (Decelle, seeded)")
    plt.plot(r_arr, m_mc, "s--", label="MC (Nishimori)")
    plt.axvline(r_c, color="k", linestyle="--", label=r"$r_c$ (KS)")
    plt.xlabel(r"$c_{\rm out}/c_{\rm in}$")
    plt.ylabel("overlap m")
    plt.title(f"2-community SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()

    plt.show()
