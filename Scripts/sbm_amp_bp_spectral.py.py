import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import time


# =====================================================
# 1. 2-community SBM の生成
# =====================================================

def generate_sbm_2comm(N, c, eps, rng=None):
    """
    2-community SBM:
        tau_i in {+1,-1}, P(tau_i=+1) = P(tau_i=-1) = 1/2
        P(A_ij=1 | tau_i tau_j = s) = c/N * (1 + eps * s)

    戻り値:
        A        : (N,N) CSR adjacency (0/1, 対称)
        tau      : 真のラベル {+1,-1}
    """
    if rng is None:
        rng = np.random.default_rng()

    tau = rng.choice([-1, 1], size=N)
    p_base = c / N

    rows = []
    cols = []

    for i in range(N):
        s_i = tau[i]
        # 上三角だけを走る
        r = rng.random(N - i - 1)
        for offset, u in enumerate(r):
            j = i + 1 + offset
            s_ij = s_i * tau[j]  # ±1
            p_ij = p_base * (1.0 + eps * s_ij)
            if p_ij < 0.0:
                p_ij = 0.0
            if p_ij > 1.0:
                p_ij = 1.0
            if u < p_ij:
                rows.extend([i, j])
                cols.extend([j, i])

    data = np.ones(len(rows), dtype=np.int8)
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    return A, tau


def overlap(tau, sigma_hat):
    """真のラベル tau と推定 sigma_hat の overlap（符号不定性は abs を取る）"""
    m = np.mean(tau * sigma_hat)
    return abs(m)


# =====================================================
# 2. AMP（GAMP のスピン版）実装
# =====================================================

def amp_sbm_2comm(
    A,
    c,
    eps,
    n_iter=50,
    rng=None,
    tau=None,
):
    """
    SBM 用 AMP（線形化＋Onsager correction）の非線形版：

        h_i^t = eps * (A m^t)_i - eps^2 (c-1) * m_i^{t-1}
        m_i^{t+1} = tanh(h_i^t)

    A : (N,N) CSR adjacency
    c : 平均次数
    eps : コミュニティ相関パラメータ
    tau : 真のラベル（あれば overlap をトラッキング）

    戻り値:
        m        : 最終ステップの磁化ベクトル (N,)
        hist_m   : 各ステップの平均 |m_i|
        hist_ovl : 各ステップの overlap（tau が与えられた場合）
    """
    if rng is None:
        rng = np.random.default_rng()
    N = A.shape[0]

    # 初期条件：小さなランダム磁化
    m_prev = np.zeros(N)
    m = 0.01 * rng.normal(size=N)

    hist_m = []
    hist_ovl = []

    for t in range(n_iter):
        # 有効場 h^t
        # A @ m は CSR × dense ベクトルの sparse mat-vec
        field = eps * (A @ m) - eps**2 * (c - 1.0) * m_prev
        m_next = np.tanh(field)

        # ログ用
        hist_m.append(np.mean(np.abs(m_next)))
        if tau is not None:
            sigma_hat = np.sign(m_next)
            sigma_hat[sigma_hat == 0] = 1
            hist_ovl.append(overlap(tau, sigma_hat))

        m_prev, m = m, m_next

    return m, np.array(hist_m), (np.array(hist_ovl) if tau is not None else None)


# =====================================================
# 3. BP 実装（メッセージ版）
# =====================================================

def bp_sbm_2comm(
    A,
    eps,
    max_iter=100,
    tol=1e-6,
    damping=0.5,
    rng=None,
    tau=None,
):
    """
    2-community SBM の BP：

        m_{i->j}^{t+1} = tanh( sum_{k in ∂i\j} atanh(eps * m_{k->i}^t) )

    A : (N,N) CSR adjacency
    eps : コミュニティ相関パラメータ
    """
    if rng is None:
        rng = np.random.default_rng()

    N = A.shape[0]
    neighbors = [A[i].indices for i in range(N)]

    # 有向辺 i->j に index をつける
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in neighbors[i]:
            edge_index[(i, j)] = idx
            idx += 1
    n_dir_edges = idx

    # メッセージ m_{i->j}
    m = 0.01 * rng.normal(size=n_dir_edges)

    def u(x):
        # atanh(eps * tanh x)
        return np.arctanh(eps * np.tanh(x))

    hist_ovl = []

    for it in range(max_iter):
        m_new = np.zeros_like(m)
        # 全有向辺を更新
        for (i, j), e_idx in edge_index.items():
            s = 0.0
            for k in neighbors[i]:
                if k == j:
                    continue
                e_in = edge_index[(k, i)]
                s += u(m[e_in])
            m_new[e_idx] = np.tanh(s)

        m_next = damping * m + (1.0 - damping) * m_new
        diff = np.max(np.abs(m_next - m))
        m = m_next
        # 収束判定
        if diff < tol:
            break

        # tau があれば overlap を計算
        if tau is not None:
            # ノード磁化 m_i = sum_{k in ∂i} u(m_{k->i})
            M = np.zeros(N)
            for i in range(N):
                s = 0.0
                for k in neighbors[i]:
                    e_in = edge_index[(k, i)]
                    s += u(m[e_in])
                M[i] = np.tanh(s)
            sigma_hat = np.sign(M)
            sigma_hat[sigma_hat == 0] = 1
            hist_ovl.append(overlap(tau, sigma_hat))

    # 最終のノード磁化
    M = np.zeros(N)
    for i in range(N):
        s = 0.0
        for k in neighbors[i]:
            e_in = edge_index[(k, i)]
            s += u(m[e_in])
        M[i] = np.tanh(s)
    sigma_hat = np.sign(M)
    sigma_hat[sigma_hat == 0] = 1

    return sigma_hat, np.array(hist_ovl)


# =====================================================
# 4. Bethe Hessian spectral（オプション）
# =====================================================

def bethe_hessian_labels(A, c=None, r=None):
    """
    Bethe Hessian:
        H(r) = (r^2 - 1) I - r A + D
    の最小固有ベクトルの符号で 2 クラスタに分割
    """
    N = A.shape[0]
    d = np.array(A.sum(axis=1)).ravel()

    if c is None:
        c = d.mean()
    if r is None:
        r = np.sqrt(c)

    I = sp.eye(N, format="csr")
    D = sp.diags(d, 0, format="csr")
    H = (r**2 - 1.0) * I - r * A + D

    vals, vecs = spla.eigsh(H, k=1, which="SA")
    v = vecs[:, 0]
    sigma_hat = np.where(v >= 0, 1, -1)
    return sigma_hat


# =====================================================
# 5. state evolution（1次元近似）
# =====================================================

def state_evolution_linear(c, eps, n_iter=50, m0=1e-2):
    """
    線形化した AMP の state evolution（平均場近似）：

        m^{t+1} = eps * c * m^t - eps^2 (c-1) m^{t-1}

    を 1 次元で回す（実際には tanh を付けた非線形版を使いたくなるが、
    しきい値の議論では線形で十分）。
    """
    m_prev = 0.0
    m = m0
    hist = [abs(m)]

    for t in range(n_iter - 1):
        m_next = eps * c * m - eps**2 * (c - 1.0) * m_prev
        m_prev, m = m, m_next
        hist.append(abs(m))
    return np.array(hist)


# =====================================================
# 6. 実験ドライバ
# =====================================================

def experiment_single(N=100000, c=3.0, eps=0.4, n_amp_iter=30, seed=0,
                      run_bp=False, run_bh=False):
    """
    1つの SBM インスタンスで AMP / BP / BH を比較する実験用関数
    """
    rng = np.random.default_rng(seed)
    print(f"[GENERATE] N={N}, c={c}, eps={eps}")
    t0 = time.time()
    A, tau = generate_sbm_2comm(N, c, eps, rng=rng)
    t1 = time.time()
    print(f"  graph generated in {t1 - t0:.2f} sec, average degree ~{A.sum(axis=1).mean():.2f}")

    # --- AMP ---
    print("[AMP] running...")
    t0 = time.time()
    m_amp, hist_m_amp, hist_ovl_amp = amp_sbm_2comm(A, c, eps, n_iter=n_amp_iter, rng=rng, tau=tau)
    t1 = time.time()
    sigma_amp = np.sign(m_amp)
    sigma_amp[sigma_amp == 0] = 1
    ovl_amp = overlap(tau, sigma_amp)
    print(f"  AMP overlap={ovl_amp:.3f}, time={t1 - t0:.2f} sec")

    # --- BP（オプション: N が大きいと重いので注意） ---
    ovl_bp = None
    hist_ovl_bp = None
    if run_bp:
        print("[BP] running...")
        t0 = time.time()
        sigma_bp, hist_ovl_bp = bp_sbm_2comm(A, eps, max_iter=100, tol=1e-4, damping=0.5, rng=rng, tau=tau)
        t1 = time.time()
        ovl_bp = overlap(tau, sigma_bp)
        print(f"  BP overlap={ovl_bp:.3f}, time={t1 - t0:.2f} sec")

    # --- Bethe Hessian（オプション: N が大きいとこれも重い） ---
    ovl_bh = None
    if run_bh:
        print("[BH] running...")
        t0 = time.time()
        sigma_bh = bethe_hessian_labels(A, c=c, r=None)
        t1 = time.time()
        ovl_bh = overlap(tau, sigma_bh)
        print(f"  BH overlap={ovl_bh:.3f}, time={t1 - t0:.2f} sec")

    # --- state evolution ---
    se_hist = state_evolution_linear(c, eps, n_iter=n_amp_iter, m0=1e-2)

    # --- 簡単なプロット ---
    iters = np.arange(len(hist_m_amp))
    plt.figure(figsize=(6,4))
    plt.plot(iters, hist_m_amp, "o-", label="|m_i| (AMP, avg)")
    if hist_ovl_amp is not None:
        plt.plot(iters, hist_ovl_amp, "s-", label="overlap (AMP)")
    if hist_ovl_bp is not None:
        plt.plot(np.arange(len(hist_ovl_bp)), hist_ovl_bp, "x-", label="overlap (BP)")
    plt.plot(np.arange(len(se_hist)), se_hist, "--", label="|m| (state evolution, linear)")
    plt.xlabel("iteration")
    plt.ylabel("m or overlap")
    plt.title(f"AMP/BP vs state evolution (N={N}, c={c}, eps={eps})")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "ovl_amp": ovl_amp,
        "ovl_bp": ovl_bp,
        "ovl_bh": ovl_bh,
    }


if __name__ == "__main__":
    # ここでパラメータをいじる
    N = 100000   # AMP なら 1e5 まで現実的
    c = 3.0
    # KS しきい値 eps_c ~ 1/sqrt(c)
    eps_c = 1.0 / np.sqrt(c)

    # しきい値のちょい上で試す
    eps = 1.2 * eps_c

    # 実験：BP / BH は N が大きいと重いので最初は False 推奨
    results = experiment_single(
        N=N,
        c=c,
        eps=eps,
        n_amp_iter=30,
        seed=0,
        run_bp=False,   # 最初は False にして様子見
        run_bh=False,   # これも最初は False
    )

    print("Final overlaps:", results)
