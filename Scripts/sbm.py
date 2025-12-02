import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

# ============================================================
# 0. ユーティリティ：ラベルの正規化 overlap
# ============================================================

def normalized_overlap(true_labels, est_labels, q):
    """
    true_labels, est_labels: shape (N,), 値は {0,...,q-1}
    SBM ではラベルの並び替えが不定なので、全ての順列を試して
    一番一致数が多くなる perm を選び、そのときの normalized overlap を返す。

    normalized overlap:
        m = (acc - 1/q) / (1 - 1/q)
    で [0,1] に正規化（ランダム ≈ 0, 完全一致 = 1）
    """
    N = len(true_labels)
    best_acc = 0.0
    for perm in permutations(range(q)):
        mapping = {a: perm[a] for a in range(q)}
        mapped = np.array([mapping[z] for z in est_labels])
        acc = np.mean(mapped == true_labels)
        if acc > best_acc:
            best_acc = acc
    # 正規化（random baseline を 0 に）
    baseline = 1.0 / q
    if best_acc <= baseline:
        return 0.0
    return (best_acc - baseline) / (1.0 - baseline)

# ============================================================
# 1. SBM の生成（q 等分, 平均次数 c, eps = cout/cin）
# ============================================================

def compute_cin_cout_from_eps(q, c, eps):
    """
    q: グループ数
    c: 平均次数
    eps = cout / cin
    条件:
      (cin + (q-1) cout)/q = c
      cout = eps * cin
    から cin, cout を求める。
    """
    # cin * (1 + (q-1)*eps) / q = c
    cin = q * c / (1.0 + (q - 1) * eps)
    cout = eps * cin
    return cin, cout

def generate_sbm_equal_groups(N, q, c, eps, rng=None):
    """
    N: 頂点数
    q: グループ数
    c: 平均次数
    eps: eps = cout/cin

    1. グループ割り当てをほぼ等分に作る
    2. p_in = cin / N, p_out = cout / N でエッジを生やす
       （O(N^2) 実装なので N は ~1e4 程度までが現実的）

    戻り値:
        adj_list : list of lists, adj_list[i] は頂点 i の隣接リスト
        labels   : shape (N,), 真のラベル in {0,...,q-1}
        cin, cout
    """
    if rng is None:
        rng = np.random.default_rng()

    cin, cout = compute_cin_cout_from_eps(q, c, eps)
    p_in = cin / N
    p_out = cout / N

    # ラベルをほぼ等分に振る
    base = N // q
    rem = N % q
    labels = np.empty(N, dtype=int)
    idx = 0
    for a in range(q):
        sz = base + (1 if a < rem else 0)
        labels[idx:idx+sz] = a
        idx += sz
    rng.shuffle(labels)

    # 隣接リストの初期化
    adj_list = [[] for _ in range(N)]

    # 全ペアを回す O(N^2) 実装
    # N を大きくしたいときはここをもっと賢く書き換える必要あり
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] == labels[j]:
                p_ij = p_in
            else:
                p_ij = p_out
            if rng.random() < p_ij:
                adj_list[i].append(j)
                adj_list[j].append(i)

    return adj_list, labels, cin, cout

# ============================================================
# 2. BP on SBM（q-state Potts, 辺だけの寄与で近似）
# ============================================================

def bp_sbm_q(adj_list, q, cin, cout,
             max_iter=200, tol=1e-6, damping=0.5, rng=None):
    """
    q-state SBM 用の BP（辺のみの寄与を使う sparse 近似版）。
    Zdeborová レビューでの SBM BP の「枝だけ見る」フォームに対応するイメージ。

    メッセージ:
        m_{i->j}(a) : 頂点 i から j へ、「i のラベルが a」の確率（未規格化OK）
    アップデート（エッジ (i,k) の情報だけを使う簡略版）:
        m_{i->j}(a) ∝ π_a * ∏_{k∈∂i\{j}} ∑_b [ m_{k->i}(b) * c_{ab} ]
        ここで c_{ab} = cin (a=b), cout (a≠b), π_a = 1/q

    戻り値:
        node_beliefs : shape (N,q)
        est_labels   : shape (N,)
        n_iter       : BP が収束（or 打ち切り）した反復数
    """
    if rng is None:
        rng = np.random.default_rng()

    N = len(adj_list)
    # グループ prior
    pi = np.ones(q) / q

    # c_ab 行列
    c_mat = np.full((q, q), cout)
    np.fill_diagonal(c_mat, cin)

    # 有向辺に index を振る
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in adj_list[i]:
            edge_index[(i, j)] = idx
            idx += 1
    n_dir_edges = idx

    # メッセージ m[e_idx, a]
    # 初期値: ほぼ一様 + 小さなノイズ

    m = np.ones((n_dir_edges, q)) / q
    m += 1e-3 * rng.normal(size=(n_dir_edges, q))
    m = np.clip(m, 1e-12, None)
    m /= m.sum(axis=1, keepdims=True)

    def logsumexp(x, axis=None):
        mmax = np.max(x, axis=axis, keepdims=True)
        return mmax + np.log(np.sum(np.exp(x - mmax), axis=axis, keepdims=True))

    for it in range(max_iter):
        m_new = np.zeros_like(m)

        # 各有向辺 i->j について更新
        for (i, j), e_idx in edge_index.items():
            neighbors = adj_list[i]
            # i の他の隣接 k != j から来るメッセージをまとめる
            # log 空間で計算
            log_msg = np.log(pi + 1e-16)  # shape (q,)
            for k in neighbors:
                if k == j:
                    continue
                e_in = edge_index[(k, i)]
                # m[k->i] と c_mat から "effective factor" を作る:
                #   f_a = Σ_b m_{k->i}(b) c_ab
                f = m[e_in].dot(c_mat.T)  # shape (q,)
                f = np.clip(f, 1e-16, None)
                log_msg += np.log(f)

            # log_msg を正規化して確率に
            # log-normalize: m_new ∝ exp(log_msg)
            lse = logsumexp(log_msg, axis=0)
            log_msg_norm = log_msg - lse
            m_new[e_idx] = np.exp(log_msg_norm)

        # ダンピング
        m_next = damping * m + (1.0 - damping) * m_new

        diff = np.max(np.abs(m_next - m))
        m = m_next
        if diff < tol:
            # 収束
            break

    n_iter = it + 1

    # ノード周辺分布（belief）を計算
    node_beliefs = np.zeros((N, q))
    for i in range(N):
        log_bel = np.log(pi + 1e-16)
        for k in adj_list[i]:
            e_in = edge_index[(k, i)]
            f = m[e_in].dot(c_mat.T)
            f = np.clip(f, 1e-16, None)
            log_bel += np.log(f)
        # 正規化
        lse = logsumexp(log_bel, axis=0)
        log_bel -= lse
        node_beliefs[i] = np.exp(log_bel)

    est_labels = node_beliefs.argmax(axis=1)
    return node_beliefs, est_labels, n_iter

# ============================================================
# 3. ε を振って Fig.10 型のプロットを作る
# ============================================================

def experiment_fig10_style(
    N=5000,
    q=4,
    c=16.0,
    eps_list=None,
    n_realizations=3,
    max_iter_bp=200,
    rng=None
):
    """
    Fig.10 左/右 パネル風のデータを作る:
      - 左: normalized overlap vs ε
      - 右: BP 収束反復数 vs ε（critical slowing down）

    実験ステップ:
      for eps in eps_list:
        cin, cout 決定
        乱数シードを変えつつ n_realizations 回:
          SBM 生成
          BP 実行
          overlap, n_iter を記録
    """
    if rng is None:
        rng = np.random.default_rng()

    if eps_list is None:
        eps_list = np.linspace(0.0, 1.0, 11)

    overlaps = []
    conv_iters = []

    for eps in eps_list:
        print(f"=== eps={eps:.3f} ===")
        ovs = []
        its = []
        for r in range(n_realizations):
            adj_list, labels_true, cin, cout = generate_sbm_equal_groups(
                N=N, q=q, c=c, eps=eps, rng=rng
            )
            node_beliefs, labels_est, n_iter = bp_sbm_q(
                adj_list, q, cin, cout,
                max_iter=max_iter_bp,
                tol=1e-4,
                damping=0.5,
                rng=rng
            )
            m_norm = normalized_overlap(labels_true, labels_est, q)
            ovs.append(m_norm)
            its.append(n_iter)
            print(f"  trial {r+1}/{n_realizations}: "
                  f"overlap={m_norm:.3f}, iters={n_iter}")

        overlaps.append(np.mean(ovs))
        conv_iters.append(np.mean(its))
        print(f"  -> mean overlap={np.mean(ovs):.3f}, "
              f"mean iters={np.mean(its):.1f}")

    return np.array(eps_list), np.array(overlaps), np.array(conv_iters)

# ============================================================
# 4. メイン：実行 & プロット
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # N を本家 Fig.10 の 1/10〜1/20 くらいにしておく（O(N^2) 生成なので）
    N = 5000
    q = 4
    c = 16.0

    eps_list = np.linspace(0.0, 1.0, 11)

    eps_arr, m_arr, iters_arr = experiment_fig10_style(
        N=N,
        q=q,
        c=c,
        eps_list=eps_list,
        n_realizations=5,
        max_iter_bp=200,
        rng=rng
    )

    # --- Fig.10 左パネル風: overlap vs eps ---
    plt.figure(figsize=(6, 4))
    plt.plot(eps_arr, m_arr, "o-", label="BP (normalized overlap)")
    plt.xlabel(r"$\varepsilon = c_{\mathrm{out}}/c_{\mathrm{in}}$")
    plt.ylabel("normalized overlap")
    plt.title(f"q={q}, c={c}, N={N} (BP only)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Fig.10 右パネル風: 収束反復数 vs eps ---
    plt.figure(figsize=(6, 4))
    plt.plot(eps_arr, iters_arr, "s-", label="BP convergence iterations")
    plt.xlabel(r"$\varepsilon = c_{\mathrm{out}}/c_{\mathrm{in}}$")
    plt.ylabel("BP iterations to converge")
    plt.title(f"q={q}, c={c}, N={N}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
