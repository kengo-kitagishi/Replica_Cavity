import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf
from typing import Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

class VianaBraySparseSpinGlass:
    """
    Viana-Bray型希薄スピンガラスのレプリカ対称解析
    文書の式を直接実装
    """
    
    def __init__(self, c: float, beta: float, J0: float = 1.0):
        """
        Parameters:
        -----------
        c : float
            平均次数（connectivity）
        beta : float
            逆温度 β = 1/T
        J0 : float
            相互作用の強さ（±J0 分布の場合）
        """
        self.c = c
        self.beta = beta
        self.J0 = J0
        
        # 結合分布 ρ(J) = (1/2)δ(J-J0) + (1/2)δ(J+J0)
        self.rho_J = lambda J: 0.5 * (np.abs(J - J0) < 1e-10) + 0.5 * (np.abs(J + J0) < 1e-10)
        
    # ==================== 1. 2スピン局所分配関数 ====================
    
    def Z1(self, h: float) -> float:
        """
        1スピン分配関数: Z_1(h) = 2cosh(βh)
        式 (eq:VB-Z1-def)
        """
        return 2 * np.cosh(self.beta * h)
    
    def Z2(self, h: float, h_prime: float, J: float) -> float:
        """
        2スピン分配関数: Z_2(h,h',J)
        式 (eq:VB-Z2-def)
        """
        return (2 * np.exp(self.beta * h) * np.cosh(self.beta * (h_prime + J)) +
                2 * np.exp(-self.beta * h) * np.cosh(self.beta * (h_prime - J)))
    
    def K_kernel(self, J: float, h: float, h_prime: float) -> float:
        """
        RS ansatz カーネル: K(J;h,h')
        式 (eq:VB-K-final-form)
        
        K(J;h,h') = cosh(βJ) + tanh(βh)tanh(βh')sinh(βJ)
        """
        return (np.cosh(self.beta * J) + 
                np.tanh(self.beta * h) * np.tanh(self.beta * h_prime) * np.sinh(self.beta * J))
    
    # ==================== 2. cavity field関数 ====================
    
    def u_function(self, J: float, h: float) -> float:
        """
        cavity field 関数: u(J,h)
        tanh(βu) = tanh(βJ)tanh(βh) から計算
        
        Parameters:
        -----------
        J : float
            結合強度
        h : float
            入力場
            
        Returns:
        --------
        u : float
            出力場
        """
        tanh_prod = np.tanh(self.beta * J) * np.tanh(self.beta * h)
        # tanh^{-1}(x) = atanh(x)
        return np.arctanh(np.clip(tanh_prod, -0.999, 0.999)) / self.beta
    
    # ==================== 3. レプリカ対称自由エネルギー汎関数 ====================
    
    def S_site(self, P_h: Callable[[float], float], h_range: np.ndarray) -> float:
        """
        Site項: -∫dh P(h)log P(h)
        式 (eq:VB-S-site-again)
        """
        dh = h_range[1] - h_range[0]
        P_vals = np.array([P_h(h) for h in h_range])
        P_vals = np.clip(P_vals, 1e-15, None)  # log(0)を避ける
        
        return -np.sum(P_vals * np.log(P_vals)) * dh
    
    def S_bond(self, P_h: Callable[[float], float], h_range: np.ndarray) -> float:
        """
        Bond項: (c/2)∫∫dh dh' P(h)P(h')∫dJ ρ(J)log K(J;h,h')
        式 (eq:VB-S-bond-again)
        """
        dh = h_range[1] - h_range[0]
        J_vals = np.array([self.J0, -self.J0])
        
        result = 0.0
        for h in h_range:
            for h_prime in h_range:
                # ∫dJ ρ(J)log K(J;h,h')
                integral_J = 0.0
                for J in J_vals:
                    K_val = self.K_kernel(J, h, h_prime)
                    if K_val > 1e-15:
                        integral_J += 0.5 * np.log(K_val)  # ρ(J) = 0.5 for each J
                
                result += P_h(h) * P_h(h_prime) * integral_J * dh * dh
        
        return (self.c / 2) * result
    
    def S_functional(self, P_h: Callable[[float], float], h_range: np.ndarray) -> float:
        """
        レプリカ対称自由エネルギー汎関数: S[P]
        式 (eq:VB-S-functional-RS)
        
        S[P] = S_site[P] + S_bond[P]
        """
        return self.S_site(P_h, h_range) + self.S_bond(P_h, h_range)
    
    # ==================== 4. Cavity方程式の数値解法 ====================
    
    def cavity_update(self, P_h_old: np.ndarray, h_range: np.ndarray, 
                     n_samples: int = 1000) -> np.ndarray:
        """
        Cavity方程式の1回の更新:
        式 (eq:VB-cavity-eq-final)
        
        P(h) = Σ_k e^{-c}c^k/k! ∫∏[dh_ℓ P(h_ℓ)dJ_ℓ ρ(J_ℓ)] δ(h - Σu(J_ℓ,h_ℓ))
        
        Monte Carloサンプリングで実装
        """
        dh = h_range[1] - h_range[0]
        P_h_new = np.zeros_like(h_range)
        
        # Poisson分布から次数をサンプル
        max_k = int(self.c + 5 * np.sqrt(self.c))  # 十分大きな上限
        
        for _ in range(n_samples):
            # 次数kをPoisson分布からサンプル
            k = np.random.poisson(self.c)
            
            if k == 0:
                h_sum = 0.0
            else:
                h_sum = 0.0
                for _ in range(k):
                    # h_ℓをP(h)からサンプル
                    h_ell = np.random.choice(h_range, p=P_h_old * dh / np.sum(P_h_old * dh))
                    # J_ℓをρ(J)からサンプル
                    J_ell = np.random.choice([self.J0, -self.J0])
                    # u(J_ℓ, h_ℓ)を計算
                    h_sum += self.u_function(J_ell, h_ell)
            
            # デルタ関数を離散化: 最も近いグリッド点に寄与
            idx = np.argmin(np.abs(h_range - h_sum))
            P_h_new[idx] += 1.0
        
        # 正規化
        P_h_new = P_h_new / (np.sum(P_h_new) * dh + 1e-15)
        
        return P_h_new
    
    def solve_cavity_equation(self, h_range: np.ndarray, 
                             max_iter: int = 100, tol: float = 1e-4) -> np.ndarray:
        """
        Cavity方程式を反復法で解く
        
        Returns:
        --------
        P_h : np.ndarray
            収束した有効場分布 P(h)
        """
        dh = h_range[1] - h_range[0]
        
        # 初期分布（Gaussian近似）
        sigma_init = 1.0 / np.sqrt(self.c * self.beta**2)
        P_h = np.exp(-h_range**2 / (2 * sigma_init**2))
        P_h = P_h / (np.sum(P_h) * dh)
        
        print("Solving cavity equation...")
        for iteration in range(max_iter):
            P_h_new = self.cavity_update(P_h, h_range)
            
            # 収束判定
            diff = np.max(np.abs(P_h_new - P_h))
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration}: diff = {diff:.6f}")
            
            if diff < tol:
                print(f"Converged at iteration {iteration}")
                break
            
            # ダンピング
            damping = 0.5
            P_h = damping * P_h_new + (1 - damping) * P_h
        
        return P_h
    
    # ==================== 5. 物理量の計算 ====================
    
    def compute_magnetization(self, P_h: np.ndarray, h_range: np.ndarray) -> float:
        """
        平均磁化: m = ∫dh P(h)tanh(βh)
        """
        dh = h_range[1] - h_range[0]
        m = np.sum(P_h * np.tanh(self.beta * h_range)) * dh
        return m
    
    def compute_overlap(self, P_h: np.ndarray, h_range: np.ndarray) -> float:
        """
        Edwards-Anderson オーバーラップ: q = ∫dh P(h)tanh²(βh)
        """
        dh = h_rßange[1] - h_range[0]
        q = np.sum(P_h * np.tanh(self.beta * h_range)**2) * dh
        return q
    
    def compute_free_energy(self, P_h: np.ndarray, h_range: np.ndarray) -> float:
        """
        自由エネルギー密度: f = -S[P]
        """
        # P(h)を関数として扱うためにinterpolate
        from scipy.interpolate import interp1d
        P_h_func = interp1d(h_range, P_h, kind='cubic', 
                            bounds_error=False, fill_value=0.0)
        
        S = self.S_functional(P_h_func, h_range)
        return -S
    
    # ==================== 6. 可視化 ====================
    
    def visualize_solution(self, P_h: np.ndarray, h_range: np.ndarray):
        """
        解の可視化
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 有効場分布 P(h)
        ax = axes[0, 0]
        ax.plot(h_range, P_h, linewidth=2, color='darkblue')
        ax.fill_between(h_range, P_h, alpha=0.3, color='lightblue')
        ax.set_title(f'Field Distribution P(h)\nc={self.c:.1f}, β={self.beta:.2f}', 
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('h (effective field)', fontsize=11)
        ax.set_ylabel('P(h)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # 2. 局所磁化分布
        ax = axes[0, 1]
        m_local = np.tanh(self.beta * h_range)
        ax.plot(h_range, m_local, linewidth=2, color='darkred', label='tanh(βh)')
        ax.plot(h_range, P_h * m_local / np.max(P_h * m_local), 
               linewidth=2, color='orange', linestyle='--', label='P(h)×tanh(βh) (scaled)')
        ax.set_title('Local Magnetization', fontsize=12, fontweight='bold')
        ax.set_xlabel('h', fontsize=11)
        ax.set_ylabel('m_local', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 自由エネルギー汎関数の各項
        ax = axes[1, 0]
        from scipy.interpolate import interp1d
        P_h_func = interp1d(h_range, P_h, kind='cubic', 
                           bounds_error=False, fill_value=0.0)
        
        S_site_val = self.S_site(P_h_func, h_range)
        S_bond_val = self.S_bond(P_h_func, h_range)
        S_total = S_site_val + S_bond_val
        
        components = ['Site\n(Entropy)', 'Bond\n(Interaction)', 'Total']
        values = [S_site_val, S_bond_val, S_total]
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        bars = ax.bar(components, values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_title('Free Energy Functional Components', fontsize=12, fontweight='bold')
        ax.set_ylabel('S[P]', fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        
        # 値を表示
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 4. 物理量のサマリー
        ax = axes[1, 1]
        ax.axis('off')
        
        m = self.compute_magnetization(P_h, h_range)
        q = self.compute_overlap(P_h, h_range)
        f = self.compute_free_energy(P_h, h_range)
        
        summary_text = f"""
        Physical Quantities
        ━━━━━━━━━━━━━━━━━━━━━
        
        Connectivity:      c = {self.c:.2f}
        Inverse Temp:      β = {self.beta:.3f}
        Temperature:       T = {1/self.beta:.3f}
        
        Magnetization:     m = {m:.4f}
        EA Overlap:        q = {q:.4f}
        Free Energy:       f = {f:.4f}
        
        Site Entropy:      S_site = {S_site_val:.4f}
        Bond Energy:       S_bond = {S_bond_val:.4f}
        Total:             S[P] = {S_total:.4f}
        """
        
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
               verticalalignment='center', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('viana_bray_replica_solution.png', dpi=300, bbox_inches='tight')
        plt.show()


# ==================== 実行例：相転移の観察 ====================

def phase_diagram_analysis():
    """
    温度を変えながら相転移を観察
    """
    c = 3.0  # 平均次数
    T_range = np.linspace(0.5, 3.0, 15)
    
    magnetizations = []
    overlaps = []
    free_energies = []
    
    h_range = np.linspace(-5, 5, 200)
    
    print("\n" + "="*60)
    print("Phase Diagram Analysis: Viana-Bray Sparse Spin Glass")
    print("="*60)
    
    for T in T_range:
        beta = 1.0 / T
        print(f"\nT = {T:.3f} (β = {beta:.3f})")
        
        vb = VianaBraySparseSpinGlass(c=c, beta=beta, J0=1.0)
        
        # Cavity方程式を解く
        P_h = vb.solve_cavity_equation(h_range, max_iter=50, tol=1e-3)
        
        # 物理量を計算
        m = vb.compute_magnetization(P_h, h_range)
        q = vb.compute_overlap(P_h, h_range)
        f = vb.compute_free_energy(P_h, h_range)
        
        magnetizations.append(m)
        overlaps.append(q)
        free_energies.append(f)
        
        print(f"  → m = {m:.4f}, q = {q:.4f}, f = {f:.4f}")
    
    # 結果の可視化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    ax = axes[0]
    ax.plot(T_range, magnetizations, 'o-', linewidth=2, markersize=6, color='darkblue')
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_ylabel('Magnetization m', fontsize=11)
    ax.set_title('Magnetization vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    ax = axes[1]
    ax.plot(T_range, overlaps, 'o-', linewidth=2, markersize=6, color='darkred')
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_ylabel('EA Overlap q', fontsize=11)
    ax.set_title('Edwards-Anderson Overlap vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[2]
    ax.plot(T_range, free_energies, 'o-', linewidth=2, markersize=6, color='darkgreen')
    ax.set_xlabel('Temperature T', fontsize=11)
    ax.set_ylabel('Free Energy f', fontsize=11)
    ax.set_title('Free Energy vs Temperature', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('viana_bray_phase_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== メイン実行 ====================

if __name__ == "__main__":
    # 1. 単一温度での詳細解析
    print("\n" + "="*60)
    print("Detailed Analysis at Single Temperature")
    print("="*60)
    
    c = 3.0
    beta = 1.0  # T = 1
    
    vb = VianaBraySparseSpinGlass(c=c, beta=beta, J0=1.0)
    h_range = np.linspace(-5, 5, 200)
    
    # Cavity方程式を解く
    P_h = vb.solve_cavity_equation(h_range, max_iter=100, tol=1e-4)
    
    # 結果の可視化
    vb.visualize_solution(P_h, h_range)
    
    # 2. 相図解析
    phase_diagram_analysis()
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)