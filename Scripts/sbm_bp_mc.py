import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. 2-community SBM の生成
# =====================================================

def generate_sbm_2comm(N, c, eps, rng=None):
    """
    2-community SBM:
        tau_i in {+1,-1}, P(tau_i=+1)=P(tau_i=-1)=1/2
        P(A_ij=1 | tau_i tau_j = s) = c/N * (1 + eps * s)

    戻り値:
        A        : (N,N) adjacency matrix (0/1, symmetric)
        neighbors: list of neighbor lists
        tau      : planted labels in {+1,-1}
    """
    if rng is None:
        rng = np.random.default_rng()

    tau = rng.choice([-1, 1], size=N)
    p_base = c / N

    A = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        s_i = tau[i]
        r = rng.random(N - i - 1)
        for offset, u in enumerate(r):
            j = i + 1 + offset
            s_ij = s_i * tau[j]  # +1 (同グループ) or -1 (異なるグループ)
            p_ij = p_base * (1.0 + eps * s_ij)
            # 念のためクリップ
            if p_ij < 0.0:
                p_ij = 0.0
            if p_ij > 1.0:
                p_ij = 1.0
            if u < p_ij:
                A[i, j] = 1
                A[j, i] = 1

    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]
    return A, neighbors, tau


# =====================================================
# 2. BP (cavity) for 2-community SBM （簡略 Ising 版）
# =====================================================

def bp_sbm_2comm(
    neighbors,
    eps,
    max_iter=200,
    tol=1e-6,
    damping=0.5,
    rng=None,
):
    """
    簡略版 BP:
        有向辺 i->j ごとに cavity メッセージ m_{i->j} を持つ。
        更新則:
            u(x) = atanh(eps * tanh x)
            m_{i->j} = tanh( sum_{k in ∂i\j} u(m_{k->i}) )

    実際の Decelle-BP から non-edge の寄与を捨てた
    「エッジ上だけの Ising-BP」に相当する近似。

    入力:
        neighbors : 隣接リスト
        eps       : SBM パラメータ ε
    戻り値:
        m_node    : 各ノードの磁化 (approx posterior mean of tau_i)
        n_iter    : 収束までの反復回数
        converged : True/False
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

    # cavity メッセージ m_{i->j}（小さな乱数で初期化）
    m = 0.01 * rng.normal(size=n_dir_edges)

    def u(x):
        # u(x) = atanh(eps * tanh x)
        return np.arctanh(eps * np.tanh(x))

    converged = False
    for it in range(max_iter):
        m_new = np.zeros_like(m)
        # すべての有向辺について更新
        for (i, j), e_idx in edge_index.items():
            s = 0.0
            for k in neighbors[i]:
                if k == j:
                    continue
                e_in = edge_index[(k, i)]
                s += u(m[e_in])
            m_new[e_idx] = np.tanh(s)

        # ダンピング
        m_next = damping * m + (1.0 - damping) * m_new

        diff = np.max(np.abs(m_next - m))
        m = m_next

        if diff < tol:
            converged = True
            break

    n_iter = it + 1

    # ノード磁化 m_i
    m_node = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in neighbors[i]:
            e_in = edge_index[(k, i)]
            s += u(m[e_in])
        m_node[i] = np.tanh(s)

    return m_node, n_iter, converged


# =====================================================
# 3. Monte Carlo (Gibbs) on approximate Nishimori Ising
# =====================================================

def mc_gibbs_sbm_nishimori(
    A,
    eps,
    c,
    n_sweeps_therm=200,
    n_sweeps_meas=200,
    rng=None,
):
    """
    SBM の Nishimori 事後分布を
        H(σ) ≈ - J1 * sum_{(i,j)∈E} σ_i σ_j
              + (J0 / (2N)) * (sum_i σ_i)^2
    で近似した Ising を Metropolis で回す。

    ここで
        p_in  = c(1+eps)/N
        p_out = c(1-eps)/N
        J1    = 0.5 * log( p_in(1-p_out) / [ p_out(1-p_in) ] )
              ≈ 0.5*log[(1+eps)/(1-eps)]
        J0    ≈ c * eps
    としている（p≪1 での 1/N 展開）。

    局所場は
        h_i = J1 * sum_{j∈∂i} σ_j - (J0/N) * M
    となるように (sum σ)^2 の項から平均場を入れている。
    （符号は SBM の非エッジ項の近似から来る）
    """
    if rng is None:
        rng = np.random.default_rng()
    N = A.shape[0]
    sigma = rng.choice([-1, 1], size=N)

    # 隣接リスト
    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]

    # SBM パラメータから couplings を計算
    p_in = c * (1.0 + eps) / N
    p_out = c * (1.0 - eps) / N

    # 数値安定のためクリップ
    p_in = min(max(p_in, 1e-12), 1 - 1e-12)
    p_out = min(max(p_out, 1e-12), 1 - 1e-12)

    J1 = 0.5 * np.log((p_in * (1 - p_out)) / (p_out * (1 - p_in)))
    # non-edge から来る "mean-field" の強さ（符号に注意）
    J0 = c * eps  # オーダーの見積もり

    # 全スピン和
    M = sigma.sum()

    def local_field(i):
        # edge 部分
        s_edge = sigma[neighbors[i]].sum()
        # mean-field 部分（(sum σ)^2 の微分）
        # d/dσ_i (J0/(2N) * M^2) = (J0/N) * M
        # Hamiltonian に -J0/(2N)*M^2 を入れるときは符号が逆になるので注意。
        # ここでは H = -J1 Σ_E σ_iσ_j + (J0/(2N)) M^2 としているので
        # h_i = J1 Σ_j∈∂i σ_j - (J0/N)*M になる。
        return J1 * s_edge - (J0 / N) * M

    # --- 熱平衡化 ---
    for _ in range(n_sweeps_therm * N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2 * sigma[i] * h  # flip のエネルギー差
        if dE <= 0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2 * sigma[i]  # flip 後の sigma[i] を足す

    # --- 測定フェーズ（ここでは最後の構成だけ返す） ---
    for _ in range(n_sweeps_meas * N):
        i = rng.integers(0, N)
        h = local_field(i)
        dE = 2 * sigma[i] * h
        if dE <= 0 or rng.random() < np.exp(-dE):
            sigma[i] *= -1
            M += 2 * sigma[i]

    return sigma


# =====================================================
# 4. オーバーラップと eps スイープ
# =====================================================

def overlap(tau, sigma_hat):
    """
    真のラベル tau と推定 sigma_hat の overlap
    符号反転の不定性があるので absolute を取る
    """
    m = np.mean(tau * sigma_hat)
    return abs(m)


def sweep_eps_bp_mc(
    N=5000,
    c=16.0,
    eps_list=None,
    n_realizations=5,
    rng=None,
):
    """
    eps をスイープして、
        - BP の overlap
        - MC (Nishimori 近似) の overlap
        - BP の収束反復回数（critical slowing down）
    を測定。
    """
    if rng is None:
        rng = np.random.default_rng()

    if eps_list is None:
        # KS 周り eps_c ~ 1/sqrt(c) なので、その前後を見る
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

            # BP
            m_node, n_iter, conv = bp_sbm_2comm(
                neighbors, eps,
                max_iter=500,
                tol=1e-5,
                damping=0.5,
                rng=rng,
            )
            sigma_bp = np.where(m_node >= 0, 1, -1)
            ms_bp.append(overlap(tau, sigma_bp))
            ts_bp.append(n_iter)

            # Monte Carlo (Nishimori 近似)
            sigma_mc = mc_gibbs_sbm_nishimori(
                A, eps, c,
                n_sweeps_therm=200,
                n_sweeps_meas=200,
                rng=rng
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
# 5. メイン：Fig.10 風の 2 枚図
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # --- パラメータ設定 ---
    N = 5000       # 計算資源に合わせて調整（10000 でもOKだが時間はかかる）
    c = 16.0       # 論文と同じ平均次数
    eps_c = 1.0 / np.sqrt(c)
    eps_list = np.linspace(0.0, 2.0 * eps_c, 15)

    # --- スイープ実行 ---
    eps_arr, m_bp, m_mc, t_bp = sweep_eps_bp_mc(
        N=N,
        c=c,
        eps_list=eps_list,
        n_realizations=3,  # 時間を見ながら増やしてOK
        rng=rng,
    )

    lam_arr = c * eps_arr**2

    # --- 図1：overlap vs λ ---
    plt.figure(figsize=(6, 4))
    plt.plot(lam_arr, m_bp, "o-", label="BP")
    plt.plot(lam_arr, m_mc, "s--", label="MC (Nishimori approx)")
    plt.axvline(1.0, color="k", linestyle="--", label=r"$\lambda_c \simeq 1$")
    plt.xlabel(r"$\lambda = c \varepsilon^2$")
    plt.ylabel("overlap m")
    plt.title(f"2-community SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()

    

    # --- 図2：BP 収束時間 vs λ （critical slowing down） ---
    plt.figure(figsize=(6, 4))
    plt.plot(lam_arr, t_bp, "o-")
    plt.axvline(1.0, color="k", linestyle="--", label=r"$\lambda_c$")
    plt.xlabel(r"$\lambda = c \varepsilon^2$")
    plt.ylabel("BP convergence iterations")
    plt.title("BP convergence time vs noise")
    plt.legend()
    plt.tight_layout()

    plt.show()
# ==========================================
#  プロット：横軸を c_out / c_in に変更
# ==========================================

# c_in = c (1 + eps)
# c_out = c (1 - eps)
# ratio r = c_out / c_in = (1 - eps) / (1 + eps)


    r_arr = (1.0 - eps_arr) / (1.0 + eps_arr)

# --- 図1：overlap vs c_out/c_in ---
    plt.figure(figsize=(6,4))
    plt.plot(r_arr, m_bp, "o-", label="BP")
    plt.plot(r_arr, m_mc, "s--", label="MC (Nishimori approx)")

# KS の臨界点 r_c = (sqrt(c)-1)/(sqrt(c)+1)
    r_c = (np.sqrt(c) - 1) / (np.sqrt(c) + 1)
    plt.axvline(r_c, color="k", linestyle="--", label=r"$r_c$ (KS)")

# BP の線形臨界 r_BP ≃ (1 - 1/(c-1)) / (1 + 1/(c-1))
    r_BP = (1 - 1/(c-1)) / (1 + 1/(c-1))
    plt.axvline(r_BP, color="r", linestyle="--", label=r"$r_{\rm BP}$ (linear)")

    plt.xlabel(r"$c_{\rm out}/c_{\rm in}$")
    plt.ylabel("overlap m")
    plt.title(f"SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()


# --- 図2：BP の収束時間 vs c_out/c_in ---
    plt.figure(figsize=(6,4))
    plt.plot(r_arr, t_bp, "o-", color="purple")
    plt.axvline(r_c, color="k", linestyle="--", label=r"$r_c$ (KS)")
    plt.axvline(r_BP, color="r", linestyle="--", label=r"$r_{\rm BP}$ (linear)")
    plt.xlabel(r"$c_{\rm out}/c_{\rm in}$")
    plt.ylabel("BP convergence iterations")
    plt.title("BP convergence time vs c_out/c_in")
    plt.legend()
    plt.tight_layout()

    plt.show()

    # ==========================================
#  プロット：横軸を c_out / c_in に変更
# ==========================================

# c_in = c (1 + eps)
# c_out = c (1 - eps)
# ratio r = c_out / c_in = (1 - eps) / (1 + eps)

    r_arr = (1.0 - eps_arr) / (1.0 + eps_arr)

# --- 図1：overlap vs c_out/c_in ---
    plt.figure(figsize=(6,4))
    plt.plot(r_arr, m_bp, "o-", label="BP")
    plt.plot(r_arr, m_mc, "s--", label="MC (Nishimori approx)")

# KS の臨界点 r_c = (sqrt(c)-1)/(sqrt(c)+1)
    r_c = (np.sqrt(c) - 1) / (np.sqrt(c) + 1)
    plt.axvline(r_c, color="k", linestyle="--", label=r"$r_c$ (KS)")

# BP の線形臨界 r_BP ≃ (1 - 1/(c-1)) / (1 + 1/(c-1))
    r_BP = (1 - 1/(c-1)) / (1 + 1/(c-1))
    plt.axvline(r_BP, color="r", linestyle="--", label=r"$r_{\rm BP}$ (linear)")

    plt.xlabel(r"$c_{\rm out}/c_{\rm in}$")
    plt.ylabel("overlap m")
    plt.title(f"SBM (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()


# --- 図2：BP の収束時間 vs c_out/c_in ---
    plt.figure(figsize=(6,4))
    plt.plot(r_arr, t_bp, "o-", color="purple")
    plt.axvline(r_c, color="k", linestyle="--", label=r"$r_c$ (KS)")
    plt.axvline(r_BP, color="r", linestyle="--", label=r"$r_{\rm BP}$ (linear)")
    plt.xlabel(r"$c_{\rm out}/c_{\rm in}$")
    plt.ylabel("BP convergence iterations")
    plt.title("BP convergence time vs c_out/c_in")
    plt.legend()
    plt.tight_layout()

    plt.show()

