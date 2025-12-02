import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# =====================================================
# 1. 2-community SBM の生成
# =====================================================

def generate_sbm_2comm(N, c, eps, rng=None):
    """
    2-community SBM:
        tau_i in {+1,-1}, P(tau_i=+1)=P(tau_i=-1)=1/2
        P(A_ij=1 | tau_i tau_j = s) = c/N * (1 + eps * s)

    戻り値:
        A        : (N,N) adjacency matrix (0/1, symmetric, CSR)
        tau      : true labels in {+1,-1}
        neighbors: list of neighbors
    """
    if rng is None:
        rng = np.random.default_rng()

    tau = rng.choice([-1, 1], size=N)
    p_base = c / N

    rows = []
    cols = []

    for i in range(N):
        s_i = tau[i]
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

    neighbors = [A[i].indices for i in range(N)]
    return A, neighbors, tau


# =====================================================
# 2. Bethe Hessian によるクラスタリング
# =====================================================

def bethe_hessian_labels(A, c=None, r=None, k=1):
    """
    Bethe Hessian H(r) を構成し、その最小固有ベクトルで2クラスタに分割。
    H(r) = (r^2 - 1) I - r A + D

    引数:
        A : (N,N) sparse adjacency (CSR)
        c : 平均次数 (なければ A から推定)
        r : H のパラメータ。指定しなければ r = sqrt(c_eff)
        k : 取り出す固有ベクトルの本数（デフォルト1）

    戻り値:
        sigma_hat : {+1,-1} のラベル推定
    """
    N = A.shape[0]
    d = np.array(A.sum(axis=1)).ravel()

    if c is None:
        c = d.mean()

    if r is None:
        # 論文中の "r = sqrt(c)" or excess-degree-based estimate
        # ここでは簡単に平均次数から
        r = np.sqrt(c)

    # H = (r^2-1)I - rA + D
    I = sp.eye(N, format="csr")
    D = sp.diags(d, 0, format="csr")
    H = (r**2 - 1.0) * I - r * A + D

    # 最小固有値側の固有ベクトルを取る
    # k=1 で十分（2コミュニティ）
    vals, vecs = spla.eigsh(H, k=k, which="SA")  # "SA": smallest algebraic
    v = vecs[:, 0]
    sigma_hat = np.where(v >= 0, 1, -1)
    return sigma_hat


# =====================================================
# 3. Non-backtracking 行列によるクラスタリング
# =====================================================

def nonbacktracking_matrix(A):
    """
    Non-backtracking 行列 B を sparse で構成するための補助。
    B は 2M x 2M 行列だが、実装では辺を有向辺に展開して
      edges_dir = [(i,j), ...]
    とし、
      B_{(i->j),(k->l)} = 1  if j == k and l != i
    と定義する。

    戻り値:
        B       : (2M,2M) sparse matrix (CSR)
        dir_edges: list of directed edges (i,j)
    """
    A = A.tocsr()
    N = A.shape[0]
    rows = []
    cols = []

    dir_edges = []
    edge_index = {}
    idx = 0

    # 全ての有向辺 i->j を列挙
    for i in range(N):
        nbrs = A[i].indices
        for j in nbrs:
            dir_edges.append((i, j))
            edge_index[(i, j)] = idx
            idx += 1

    # B の非ゼロを構成
    for (i, j), idx1 in edge_index.items():
        # j から出る有向辺 j->l を見て、l != i のとき B_{(i->j),(j->l)} = 1
        for l in A[j].indices:
            if l == i:
                continue
            idx2 = edge_index[(j, l)]
            rows.append(idx1)
            cols.append(idx2)

    data = np.ones(len(rows), dtype=np.float32)
    B = sp.csr_matrix((data, (rows, cols)), shape=(idx, idx))

    return B, dir_edges


def nb_labels(A, k=1):
    """
    Non-backtracking 行列 B の最大固有ベクトルからノードラベルを復元する
    （Hashimoto matrix を使った標準的なやり方）

    手順:
        1. B の上位固有ベクトル v を計算
        2. ノード i のスコア x_i = sum_{edges e=(i->j)} v_e で集約
        3. x_i の符号で2クラスタに分ける

    戻り値:
        sigma_hat : {+1,-1} の推定ラベル
    """
    B, dir_edges = nonbacktracking_matrix(A)
    # 最大固有値側を取りたいので which="LR" (largest real)
    vals, vecs = spla.eigs(B, k=k, which="LR")
    v = np.real(vecs[:, 0])

    N = A.shape[0]
    x = np.zeros(N)
    for val, (i, j) in zip(v, dir_edges):
        x[i] += val
        # あるいは x[j] += val など集約方法はいくつかあるが、
        # この程度の違いは実験的には大差ない。

    sigma_hat = np.where(x >= 0, 1, -1)
    return sigma_hat


# =====================================================
# 4. overlap と λ スイープ
# =====================================================

def overlap(tau, sigma_hat):
    """
    planted label tau と推定 sigma_hat の overlap
    符号反転不定性があるので abs をとる
    """
    m = np.mean(tau * sigma_hat)
    return abs(m)


def sweep_lambda_spectral(
    N=50000,
    c=3.0,
    eps_list=None,
    n_realizations=3,
    use_nb=True,
    rng=None,
):
    """
    λ=c eps^2 をスイープしながら、
    Bethe Hessian と non-backtracking の overlap m を測る。
    """
    if rng is None:
        rng = np.random.default_rng()

    if eps_list is None:
        # KS しきい値 eps_c ~ 1/sqrt(c) の前後をスキャン
        eps_c = 1.0 / np.sqrt(c)
        eps_list = np.linspace(0.0, 2.0 * eps_c, 12)

    lambdas = c * eps_list**2
    m_bh_list = []
    m_nb_list = []

    for eps, lam in zip(eps_list, lambdas):
        ms_bh = []
        ms_nb = []

        for r in range(n_realizations):
            A, neighbors, tau = generate_sbm_2comm(N, c, eps, rng=rng)

            # Bethe Hessian
            sigma_bh = bethe_hessian_labels(A, c=c, r=None)
            ms_bh.append(overlap(tau, sigma_bh))

            # Non-backtracking
            if use_nb:
                sigma_nb = nb_labels(A)
                ms_nb.append(overlap(tau, sigma_nb))

        m_bh = np.mean(ms_bh)
        m_bh_list.append(m_bh)

        if use_nb:
            m_nb = np.mean(ms_nb)
            m_nb_list.append(m_nb)
            print(f"eps={eps:.3f}, lambda={lam:.3f} : "
                  f"<m_BH>={m_bh:.3f}, <m_NB>={m_nb:.3f}")
        else:
            m_nb_list.append(np.nan)
            print(f"eps={eps:.3f}, lambda={lam:.3f} : "
                  f"<m_BH>={m_bh:.3f}")

    return lambdas, np.array(m_bh_list), np.array(m_nb_list)


# =====================================================
# 5. メイン：spectral redemption 図
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    N = 50000       # N=5万 くらいまでOK（NBが少し重いときは N を減らす）
    c = 3.0
    eps_c = 1.0 / np.sqrt(c)
    eps_list = np.linspace(0.0, 2.0 * eps_c, 10)

    lambdas, m_bh, m_nb = sweep_lambda_spectral(
        N=N,
        c=c,
        eps_list=eps_list,
        n_realizations=3,
        use_nb=True,
        rng=rng,
    )

    plt.figure(figsize=(6,4))
    plt.plot(lambdas, m_bh, "o-", label="Bethe Hessian")
    plt.plot(lambdas, m_nb, "s--", label="Non-backtracking")
    plt.axvline(1.0, color="k", linestyle="--", label=r"KS: $c\varepsilon^2=1$")
    plt.xlabel(r"$\lambda = c\varepsilon^2$")
    plt.ylabel("overlap m")
    plt.title(f"2-community SBM spectral methods (N={N}, c={c})")
    plt.legend()
    plt.tight_layout()
    plt.show()
