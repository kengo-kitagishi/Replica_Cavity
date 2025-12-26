# %%
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import poisson

class IntegratedSBMAnalysis:
    """
    SBM解析クラス: BP, TAP, State Evolution を統合
    Based on Viana-Bray (Diluted Spin Glass) formulation
    """
    def __init__(self, N, c_in, c_out):
        self.N = N
        self.c_in = c_in
        self.c_out = c_out
        self.c_avg = (c_in + c_out) / 2.0
        
        # --- 物理パラメータ設定 (Nishimori Line) ---
        # 相互作用強度 J (beta=1 とする)
        # 文書: beta * J = (1/2) * ln(c_in / c_out)
        if c_out > 1e-9:
            self.beta_J = 0.5 * np.log(c_in / c_out)
        else:
            self.beta_J = 5.0 # 発散回避用
            
        self.tanh_beta_J = np.tanh(self.beta_J)

    def generate_instance(self):
        """SBMグラフと真の配置を生成"""
        sigma_true = np.ones(self.N)
        sigma_true[self.N//2:] = -1.0
        
        probs = [[self.c_in/self.N, self.c_out/self.N], 
                 [self.c_out/self.N, self.c_in/self.N]]
        G = nx.stochastic_block_model([self.N//2, self.N - self.N//2], probs)
        
        # 隣接リスト形式と隣接行列形式の両方を準備
        adj_list = [list(G.neighbors(i)) for i in range(self.N)]
        A_matrix = nx.to_numpy_array(G)
        
        return sigma_true, adj_list, A_matrix

    # =========================================================
    # Method 1: Belief Propagation (BP)
    # =========================================================
    def run_bp(self, adj_list, sigma_true, max_iter=20, damping=0.5):
        # メッセージの初期化 (微小なバイアス)
        # 辞書キー: (source, target) -> message value
        messages = {}
        edges = []
        for i in range(self.N):
            for j in adj_list[i]:
                edges.append((i, j))
                # 初期値: Planted initialization
                messages[(i, j)] = np.random.normal(0, 0.1) + 0.1 * sigma_true[i]
        
        for t in range(max_iter):
            new_messages = {}
            diff = 0.0
            
            # メッセージ更新
            # h_{i->j} = sum_{k in N(i)\j} atanh( tanh(J) * tanh(h_{k->i}) )
            for i, j in edges:
                sum_h = 0.0
                for k in adj_list[i]:
                    if k == j: continue
                    
                    h_ki = messages.get((k, i), 0.0)
                    # 非線形変換
                    val = self.tanh_beta_J * np.tanh(h_ki)
                    val = np.clip(val, -0.9999, 0.9999)
                    sum_h += np.arctanh(val)
                
                # Damping
                new_msg = (1-damping)*sum_h + damping*messages[(i, j)]
                new_messages[(i, j)] = new_msg
                diff += np.abs(new_msg - messages[(i, j)])
            
            messages = new_messages
            if diff / len(edges) < 1e-4: break

        # 周辺化 (Marginalization)
        m_est = np.zeros(self.N)
        for i in range(self.N):
            sum_h = 0.0
            for k in adj_list[i]:
                h_ki = messages.get((k, i), 0.0)
                val = self.tanh_beta_J * np.tanh(h_ki)
                val = np.clip(val, -0.9999, 0.9999)
                sum_h += np.arctanh(val)
            m_est[i] = np.tanh(sum_h)
            
        overlap = np.abs(np.mean(sigma_true * m_est))
        return overlap

    # =========================================================
    # Method 2: TAP Equations (AMP for Ising)
    # =========================================================
    def run_tap(self, A_matrix, sigma_true, max_iter=20, damping=0.5):
        J_mat = self.beta_J * A_matrix
        J2_mat = J_mat**2
        
        # 初期化
        m_t = np.random.normal(0, 0.1, self.N) + 0.1 * sigma_true
        m_prev = np.zeros(self.N)
        
        for t in range(max_iter):
            # 1. Mean Field Term: sum J_{ij} m_j
            mf_term = J_mat @ m_t
            
            # 2. Onsager Reaction Term: - m_i^{t-1} * sum J_{ij}^2 (1 - m_j^2)
            variance = 1 - m_t**2
            reaction_coeff = J2_mat @ variance
            onsager_term = m_prev * reaction_coeff
            
            # TAP Update
            effective_field = mf_term - onsager_term
            m_new = np.tanh(effective_field)
            
            # Damping & Update
            m_new = (1-damping)*m_new + damping*m_t
            
            diff = np.mean(np.abs(m_new - m_t))
            if diff < 1e-4: break
            
            m_prev = m_t.copy()
            m_t = m_new.copy()
            
        overlap = np.abs(np.mean(sigma_true * m_t))
        return overlap

    # =========================================================
    # Method 3: State Evolution (Population Dynamics)
    # =========================================================
    def run_state_evolution(self, pop_size=5000, max_iter=20):
        # 初期ポピュレーション: 情報を持つ分布 (Planted Gauge)
        population = np.random.normal(1.0, 0.5, pop_size)
        
        # Plantedモデルでは、結合定数の符号は J * s_i * s_j となる。
        # Nishimori線上では、これを "有効的に強磁性 + 確率的な符号反転" として扱える。
        # 近似: 平均次数 c の Poisson分布に従う近傍からの入力を合計する
        
        for t in range(max_iter):
            new_pop = np.zeros(pop_size)
            
            # 次数分布 (Poisson)
            degrees = np.random.poisson(self.c_avg, pop_size)
            
            
            total_neighbors = np.sum(degrees)
            neighbor_indices = np.random.randint(0, pop_size, total_neighbors)
            neighbor_fields = population[neighbor_indices]
            
            # Planted Gauge変換:
            # SBMにおいて、エッジ上でのスピンの積 s_i*s_j は確率的に+1か-1になる。
            # エッジが存在する条件付きでの s_i*s_j=1 の確率は c_in / (c_in + c_out)
            p_ferro = self.c_in / (self.c_in + self.c_out)
            signs = np.random.choice([1, -1], size=total_neighbors, p=[p_ferro, 1-p_ferro])
            
            # 非線形変換 u = atanh( tanh(beta J) * sign * tanh(h) )
            vals = self.tanh_beta_J * signs * np.tanh(neighbor_fields)
            vals = np.clip(vals, -0.9999, 0.9999)
            u_vals = np.arctanh(vals)
            
            current_idx = 0
            for i in range(pop_size):
                k = degrees[i]
                if k > 0:
                    new_pop[i] = np.sum(u_vals[current_idx : current_idx+k])
                    current_idx += k
                else:
                    new_pop[i] = 0.0
            
            population = new_pop

        # 秩序変数 (Overlap) m = int P(h) tanh(h) dh
        m = np.mean(np.tanh(population))
        return m

# --- Main Execution Loop ---

N = 50000  
c_avg = 3.0
delta_c_vals = np.linspace(0.5, 6.0, 50) # Delta c = c_in - c_out

results = {'dc': [], 'BP': [], 'TAP': [], 'SE': []}

# 理論的閾値 (Kesten-Stigum)
# (c_in - c_out)^2 > 2(c_in + c_out) for q=2 SBM ?
# 正確には (c_in - c_out)^2 > q * c_avg ですが、q=2のとき
# lambda = (c_in - c_out)/2, threshold is lambda^2 > c_avg
# => (c_in - c_out)^2 / 4 > c_avg => (c_in - c_out) > 2 * sqrt(c_avg)
threshold = 2 * np.sqrt(c_avg)

print(f"Running Analysis for N={N}, c_avg={c_avg}")
print(f"Theoretical Threshold (Delta c): {threshold:.3f}")
print(f"{'Delta c':<10} | {'BP':<8} | {'TAP':<8} | {'SE (inf)':<8}")
print("-" * 45)

for dc in delta_c_vals:
    c_in = c_avg + dc / 2.0
    c_out = c_avg - dc / 2.0
    
    if c_out <= 0: continue
    
    analyzer = IntegratedSBMAnalysis(N, c_in, c_out)
    
    # Generate Graph
    sigma, adj, A = analyzer.generate_instance()
    
    # 1. Run BP
    m_bp = analyzer.run_bp(adj, sigma)
    
    # 2. Run TAP
    m_tap = analyzer.run_tap(A, sigma)
    
    # 3. Run State Evolution
    m_se = analyzer.run_state_evolution()
    
    results['dc'].append(dc)
    results['BP'].append(m_bp)
    results['TAP'].append(m_tap)
    results['SE'].append(m_se)
    
    print(f"{dc:<10.2f} | {m_bp:<8.3f} | {m_tap:<8.3f} | {m_se:<8.3f}")

# --- Plotting ---
plt.figure(figsize=(10, 6))
plt.plot(results['dc'], results['BP'], 'o-', label=f'BP (Exact, N={N})', linewidth=2)
plt.plot(results['dc'], results['TAP'], 's--', label=f'TAP (Approx, N={N})')
plt.plot(results['dc'], results['SE'], 'k-.', label='State Evolution (N→∞)', linewidth=2)

plt.axvline(x=threshold, color='red', linestyle=':', label='KS Threshold')
plt.title(f'SBM Phase Transition Analysis (c_avg={c_avg})')
plt.xlabel(r'Signal Strength $\Delta c = c_{in} - c_{out}$')
plt.ylabel('Overlap (Magnetization)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
# %%
