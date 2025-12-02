import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. 2-community SBM の生成
# =====================================================

def generate_sbm_2community(N, c, eps, rng=None):
    """
    2-community SBM:
        P(tau_i = ±1) = 1/2
        P(A_ij=1 | tau_i tau_j = s) = c/N * (1 + eps * s)

    戻り値:
        A: (N,N) adjacency matrix (0/1, symmetric)
        tau: true labels in {+1,-1}
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
            sign_ij = s_i * tau[j]          # tau_i tau_j = ±1
            p_ij = p_base * (1.0 + eps * sign_ij)
            # 念のためクリップ
            if p_ij < 0.0:
                p_ij = 0.0
            if p_ij > 1.0:
                p_ij = 1.0
            if u < p_ij:
                A[i, j] = 1
                A[j, i] = 1

    return A, tau

# =====================================================
# 2. spectral 2-way partition（超シンプル版）
# =====================================================

def spectral_2way_partition(A, use_laplacian=True):
    """
    A: adjacency matrix (N,N)
    戻り値:
        sigma_hat: estimated labels in {+1,-1}
    """
    N = A.shape[0]
    d = A.sum(axis=1)

    if use_laplacian:
        # 正規化ラプラシアン L = D^{-1/2} A D^{-1/2}
        d_sqrt_inv = np.zeros_like(d, dtype=float)
        mask = d > 0
        d_sqrt_inv[mask] = 1.0 / np.sqrt(d[mask])
        D_half_inv = np.diag(d_sqrt_inv)
        M = D_half_inv @ A @ D_half_inv
    else:
        M = A.astype(float)

    # 最も大きい固有値の固有ベクトルで2分割する（簡略版）
    vals, vecs = np.linalg.eigh(M)
    v = vecs[:, -2]  # 最大固有値の固有ベクトル
    sigma_hat = np.where(v >= 0, 1, -1)
    return sigma_hat

# =====================================================
# 3. Belief Propagation (BP) for 2-community SBM
# =====================================================

def bp_sbm_2community(A, eps, max_iter=50, damping=0.5, tol=1e-6, rng=None):
    """
    2-community SBM 用の簡易 BP (tree-like approximation)
    A: adjacency matrix (N,N)
    eps: edge channel correlation parameter
         P(same label | edge) ~ (1+eps)/2
    戻り値:
        sigma_hat: estimated labels in {+1,-1}
    """
    if rng is None:
        rng = np.random.default_rng()

    N = A.shape[0]
    neighbors = [np.where(A[i] > 0)[0] for i in range(N)]

    # 有向辺 i->j に index を振る
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in neighbors[i]:
            edge_index[(i, j)] = idx
            idx += 1
    n_dir_edges = idx

    # メッセージ h_{i->j} の初期値（小さなランダム）
    h = 0.01 * rng.normal(size=n_dir_edges)

    def u(x):
        # u(x) = artanh(eps * tanh x)
        return np.arctanh(eps * np.tanh(x))

    for it in range(max_iter):
        h_new = np.zeros_like(h)
        for (i, j), e_idx in edge_index.items():
            s = 0.0
            for k in neighbors[i]:
                if k == j:
                    continue
                e_in = edge_index[(k, i)]
                s += u(h[e_in])
            h_new[e_idx] = s

        # ダンピング付き更新
        h_next = damping * h + (1.0 - damping) * h_new

        diff = np.max(np.abs(h_next - h))
        h = h_next
        if diff < tol:
            # print(f"BP converged at iter {it}, diff={diff:.2e}")
            break

    # ノードごとの場 H_i
    H = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in neighbors[i]:
            e_in = edge_index[(k, i)]
            s += u(h[e_in])
        H[i] = s

    sigma_hat = np.where(H >= 0, 1, -1)
    return sigma_hat

# =====================================================
# 4. overlap m の定義 & λスキャン
# =====================================================

def overlap_m(tau, sigma_hat):
    """
    真のラベル tau と推定 sigma_hat の overlap m を計算
    符号反転の不定性があるので、普通は絶対値をとる
    """
    m = np.mean(tau * sigma_hat)
    return abs(m)

def sweep_lambda_sbm(
    N=2000,
    c=3.0,
    eps_list=None,
    n_realizations=5,
    rng=None,
):
    """
    λ = c eps^2 をスキャンして
    spectral / BP の overlap m を測定
    """
    if rng is None:
        rng = np.random.default_rng()
    if eps_list is None:
        # λ ~ 0 〜 2 をざっくりカバー
        eps_list = np.linspace(0.0, np.sqrt(2.0 / c), 10)

    lambdas = c * eps_list ** 2
    m_spec_list = []
    m_bp_list = []

    for eps, lam in zip(eps_list, lambdas):
        ms_spec = []
        ms_bp = []
        for r in range(n_realizations):
            A, tau = generate_sbm_2community(N, c, eps, rng=rng)
            # spectral クラスタリング
            sigma_spec = spectral_2way_partition(A, use_laplacian=True)
            ms_spec.append(overlap_m(tau, sigma_spec))
            # BP
            sigma_bp = bp_sbm_2community(A, eps, max_iter=50, damping=0.5, tol=1e-4, rng=rng)
            ms_bp.append(overlap_m(tau, sigma_bp))

        m_spec = np.mean(ms_spec)
        m_bp = np.mean(ms_bp)

        m_spec_list.append(m_spec)
        m_bp_list.append(m_bp)

        print(f"eps={eps:.3f}, lambda={lam:.3f} : "
              f"<m_spec>={m_spec:.3f}, <m_bp>={m_bp:.3f}")

    return lambdas, np.array(m_spec_list), np.array(m_bp_list)

# =====================================================
# 5. メイン: λ vs m のプロット
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 2000
    c = 3.0

    # λ が 0〜2 くらいになるように eps を選ぶ
    eps_list = np.linspace(0.0, np.sqrt(2.0/c), 12)
    lambdas, m_spec, m_bp = sweep_lambda_sbm(
        N=N,
        c=c,
        eps_list=eps_list,
        n_realizations=5,
        rng=rng,
    )

    plt.figure()
    plt.plot(lambdas, m_spec, "o-", label="spectral")
    plt.plot(lambdas, m_bp, "s-", label="BP")
    plt.axvline(1.0, color="k", linestyle="--", label=r"KS: $\lambda=1$")
    plt.xlabel(r"$\lambda = c\varepsilon^2$")
    plt.ylabel("overlap m")
    plt.title("2-community SBM: overlap vs lambda")
    plt.legend()
    plt.tight_layout()
    plt.show()
