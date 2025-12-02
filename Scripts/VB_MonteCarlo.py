import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. Viana–Bray のグラフ生成
# =====================================================

def generate_vb_instance(N, c, J_scale=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    p = c / N
    neighbors = [[] for _ in range(N)]
    J_dict = {}
    for i in range(N):
        probs = rng.random(N - i - 1)
        js = np.where(probs < p)[0] + i + 1
        for j in js:
            neighbors[i].append(j)
            neighbors[j].append(i)
            J = J_scale * (1 if rng.random() < 0.5 else -1)
            J_dict[(i, j)] = J
    return neighbors, J_dict

# =====================================================
# 2. Glauber dynamics（1レプリカ用の補助）
# =====================================================

def local_field(i, spins, neighbors, J_dict):
    h = 0.0
    for j in neighbors[i]:
        if i < j:
            J = J_dict[(i, j)]
        else:
            J = J_dict[(j, i)]
        h += J * spins[j]
    return h

def mc_step_glauber(spins, beta, neighbors, J_dict, rng):
    N = len(spins)
    for _ in range(N):
        i = rng.integers(0, N)
        h = local_field(i, spins, neighbors, J_dict)
        dE = 2.0 * spins[i] * h
        if rng.random() < 1.0 / (1.0 + np.exp(beta * dE)):
            spins[i] *= -1

# =====================================================
# 3. 2レプリカで q を測るメイン関数
# =====================================================

def vb_two_replica_mc(
    N=500,
    c=3.0,
    T=1.5,
    n_therm=500,
    n_meas=10000,
    meas_interval=10,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    beta = 1.0 / T

    neighbors, J_dict = generate_vb_instance(N, c, rng=rng)

    # 2レプリカ初期化
    spins1 = rng.choice([-1, 1], size=N)
    spins2 = rng.choice([-1, 1], size=N)

    # 熱平衡化
    for _ in range(n_therm):
        mc_step_glauber(spins1, beta, neighbors, J_dict, rng)
        mc_step_glauber(spins2, beta, neighbors, J_dict, rng)

    # 測定
    q_samples = []
    q2_samples = []
    n_steps = 0
    while n_steps < n_meas:
        mc_step_glauber(spins1, beta, neighbors, J_dict, rng)
        mc_step_glauber(spins2, beta, neighbors, J_dict, rng)
        n_steps += 1
        if n_steps % meas_interval == 0:
            q = np.mean(spins1 * spins2)
            q_samples.append(q)
            q2_samples.append(q**2)

    q_samples = np.array(q_samples)
    q2_samples = np.array(q2_samples)

    return q_samples.mean(), q2_samples.mean()

# =====================================================
# 4. 温度スイープ
# =====================================================

def sweep_temperature_vb(
    N=500,
    c=3.0,
    Ts=np.linspace(0.2, 2.0, 15),
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
    q_means = []
    q2_means = []
    for T in Ts:
        q, q2 = vb_two_replica_mc_T_dependent(N=N, c=c, T=T, rng=rng)
        q_means.append(q)
        q2_means.append(q2)
        print(f"T={T:.3f} : <q>={q:.3f}, <q^2>={q2:.3f}")
    return np.array(Ts), np.array(q_means), np.array(q2_means)

# =====================================================
# 5. 実行 & プロット
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # 温度のリスト（適当に調整してOK）
    Ts = np.linspace(0.1, 2.5, 50)

    Ts, q_means, q2_means = sweep_temperature_vb(
        N=500,
        c=3.0,
        Ts=Ts,
        rng=rng,
    )

    plt.figure()
    plt.plot(Ts, q2_means, marker="o")
    plt.xlabel("Temperature T")
    plt.ylabel(r"$\langle q^2 \rangle$")
    plt.title("Viana-Bray model: spin-glass order parameter vs T")
    plt.gca().invert_xaxis()  # 右に行くほど低温に見せたければ
    plt.tight_layout()
    plt.show()

