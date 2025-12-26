import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve, brentq
from scipy.interpolate import interp1d
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class SpinGlassComparison:
    """
    Cavity法とReplica法の両方を実装して比較
    """
    
    def __init__(self, c: float, J0: float = 1.0):
        """
        Parameters:
        -----------
        c : float
            平均次数（connectivity）
        J0 : float
            相互作用の強さ（±J0 分布）
        """
        self.c = c
        self.J0 = J0
        
    # ==================== CAVITY METHOD ====================
    
    def u_cavity(self, J: float, h: float, beta: float) -> float:
        """
        Cavity field関数: u(J,h)
        tanh(βu) = tanh(βJ)tanh(βh)
        """
        tanh_prod = np.tanh(beta * J) * np.tanh(beta * h)
        return np.arctanh(np.clip(tanh_prod, -0.999, 0.999)) / beta
    
    def cavity_iteration(self, P_h_old: np.ndarray, h_range: np.ndarray, 
                         beta: float, n_samples: int = 2000) -> np.ndarray:
        """
        Cavity方程式の1反復:
        P(h) = Σ_k e^{-c}c^k/k! ∫∏[dh_ℓ P(h_ℓ)dJ_ℓ ρ(J_ℓ)] δ(h - Σu(J_ℓ,h_ℓ))
        """
        dh = h_range[1] - h_range[0]
        P_h_new = np.zeros_like(h_range)
        
        # 正規化されたP_h_oldを使う
        P_h_norm = P_h_old / (np.sum(P_h_old) * dh + 1e-15)
        
        for _ in range(n_samples):
            # 次数kをPoisson分布からサンプル
            k = np.random.poisson(self.c)
            
            if k == 0:
                h_sum = 0.0
            else:
                h_sum = 0.0
                for _ in range(k):
                    # h_ℓをP(h)からサンプル
                    h_ell = np.random.choice(h_range, p=P_h_norm * dh / np.sum(P_h_norm * dh))
                    # J_ℓを±J0から選ぶ
                    J_ell = np.random.choice([self.J0, -self.J0])
                    # u(J_ℓ, h_ℓ)を加算
                    h_sum += self.u_cavity(J_ell, h_ell, beta)
            
            # 最も近いグリッド点に寄与
            idx = np.argmin(np.abs(h_range - h_sum))
            if 0 <= idx < len(P_h_new):
                P_h_new[idx] += 1.0
        
        # 正規化
        P_h_new = P_h_new / (np.sum(P_h_new) * dh + 1e-15)
        
        return P_h_new
    
    def solve_cavity(self, beta: float, h_range: np.ndarray, 
                     max_iter: int = 100, tol: float = 1e-4, 
                     damping: float = 0.3) -> Dict:
        """
        Cavity方程式を反復法で解く
        """
        dh = h_range[1] - h_range[0]
        
        # 初期分布（Gaussian）
        sigma_init = 1.0 / np.sqrt(self.c * beta**2)
        P_h = np.exp(-h_range**2 / (2 * sigma_init**2))
        P_h = P_h / (np.sum(P_h) * dh)
        
        history = {'iteration': [], 'diff': [], 'magnetization': [], 'overlap': []}
        
        print("\n" + "="*60)
        print("CAVITY METHOD")
        print("="*60)
        
        for iteration in range(max_iter):
            P_h_new = self.cavity_iteration(P_h, h_range, beta)
            
            # 収束判定
            diff = np.sqrt(np.sum((P_h_new - P_h)**2) * dh)
            
            # 物理量
            m = np.sum(P_h * np.tanh(beta * h_range)) * dh
            q = np.sum(P_h * np.tanh(beta * h_range)**2) * dh
            
            history['iteration'].append(iteration)
            history['diff'].append(diff)
            history['magnetization'].append(m)
            history['overlap'].append(q)
            
            if iteration % 10 == 0:
                print(f"Iter {iteration:3d}: diff={diff:.6f}, m={m:.4f}, q={q:.4f}")
            
            if diff < tol and iteration > 20:
                print(f"\n✓ Converged at iteration {iteration}")
                break
            
            # ダンピング
            P_h = damping * P_h_new + (1 - damping) * P_h
        
        # 最終的な物理量
        m_final = np.sum(P_h * np.tanh(beta * h_range)) * dh
        q_final = np.sum(P_h * np.tanh(beta * h_range)**2) * dh
        
        # 自由エネルギー（近似）
        entropy = -np.sum(P_h * np.log(P_h + 1e-15)) * dh
        
        return {
            'P_h': P_h,
            'h_range': h_range,
            'magnetization': m_final,
            'overlap': q_final,
            'entropy': entropy,
            'history': history,
            'method': 'cavity'
        }
    
    # ==================== REPLICA METHOD ====================
    
    def replica_free_energy(self, q: float, beta: float) -> float:
        """
        Replica対称自由エネルギー（単位系あたり）:
        f_RS(q) = -β⁻¹[log(2) + (c/2)log(cosh²(βJ₀) - q·sinh²(βJ₀))]
        
        より正確には：
        f = -(1/β)[-(1/2)log(1-q²) - (c/2)∫dJ ρ(J)log K(J;q)]
        
        ここではViana-Bray型のシンプルな形を使用
        """
        if q >= 1.0 or q <= -1.0:
            return 1e10  # ペナルティ
        
        # Ising model on random graph with ±J₀
        # 各エッジの寄与
        cosh_term = np.cosh(beta * self.J0)**2
        sinh_term = np.sinh(beta * self.J0)**2
        
        if cosh_term - q * sinh_term <= 0:
            return 1e10
        
        # レプリカ対称自由エネルギー
        f_replica = -(1.0/beta) * (
            -0.5 * np.log(1 - q**2) +
            (self.c/2) * np.log(cosh_term - q * sinh_term)
        )
        
        return f_replica
    
    def replica_equation(self, q: float, beta: float) -> float:
        """
        Replica自己無撞着方程式:
        q = <tanh²(βh)>_P(h)
        
        P(h)はGaussian近似を使用:
        P(h) ~ N(0, σ²), σ² = c·tanh²(βJ₀)·q
        """
        sigma_sq = self.c * np.tanh(beta * self.J0)**2 * q
        
        if sigma_sq < 1e-10:
            return -q  # q=0が解
        
        sigma = np.sqrt(sigma_sq)
        
        # <tanh²(βh)>をGaussian積分で計算
        def integrand(h):
            weight = np.exp(-h**2 / (2 * sigma_sq)) / np.sqrt(2 * np.pi * sigma_sq)
            return weight * np.tanh(beta * h)**2
        
        q_new, _ = quad(integrand, -10*sigma, 10*sigma, limit=100)
        
        return q_new - q
    
    def solve_replica(self, beta: float, h_range: np.ndarray) -> Dict:
        """
        Replica法でオーダーパラメータqを解く
        """
        print("\n" + "="*60)
        print("REPLICA METHOD")
        print("="*60)
        
        # 自己無撞着方程式を解く
        try:
            # まず弱結合解を試す（q ≈ 0）
            q_solution = fsolve(lambda q: self.replica_equation(q, beta), 0.01)[0]
            
            # 負の解や1以上の解は物理的でない
            if q_solution < 0 or q_solution >= 1.0:
                q_solution = 0.0
                print("Warning: Non-physical solution, setting q=0")
        except:
            q_solution = 0.0
            print("Warning: Failed to solve, setting q=0")
        
        print(f"\nSolved: q = {q_solution:.6f}")
        
        # 対応するGaussian分布を構成
        sigma_sq = self.c * np.tanh(beta * self.J0)**2 * q_solution
        sigma = np.sqrt(sigma_sq) if sigma_sq > 0 else 1e-3
        
        dh = h_range[1] - h_range[0]
        P_h = np.exp(-h_range**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)
        P_h = P_h / (np.sum(P_h) * dh)
        
        # 物理量
        m = np.sum(P_h * np.tanh(beta * h_range)) * dh
        q_check = np.sum(P_h * np.tanh(beta * h_range)**2) * dh
        
        # 自由エネルギー
        f_replica = self.replica_free_energy(q_solution, beta)
        
        print(f"Magnetization: m = {m:.6f}")
        print(f"Overlap (check): q = {q_check:.6f}")
        print(f"Free energy: f = {f_replica:.6f}")
        
        return {
            'P_h': P_h,
            'h_range': h_range,
            'magnetization': m,
            'overlap': q_solution,
            'overlap_check': q_check,
            'free_energy': f_replica,
            'sigma': sigma,
            'method': 'replica'
        }
    
    # ==================== COMPARISON & VISUALIZATION ====================
    
    def compare_methods(self, beta: float, h_range: np.ndarray = None) -> Dict:
        """
        Cavity法とReplica法を両方実行して比較
        """
        if h_range is None:
            h_range = np.linspace(-5, 5, 300)
        
        T = 1.0 / beta
        print(f"\n{'='*60}")
        print(f"COMPARISON: c={self.c:.2f}, T={T:.3f} (β={beta:.3f})")
        print(f"{'='*60}")
        
        # 1. Cavity法
        cavity_result = self.solve_cavity(beta, h_range, max_iter=100, tol=1e-4)
        
        # 2. Replica法
        replica_result = self.solve_replica(beta, h_range)
        
        # 3. 比較
        self._print_comparison(cavity_result, replica_result)
        
        return {
            'cavity': cavity_result,
            'replica': replica_result,
            'beta': beta,
            'temperature': T
        }
    
    def _print_comparison(self, cavity: Dict, replica: Dict):
        """
        結果の比較を表示
        """
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print(f"\n{'Quantity':<20} {'Cavity':<15} {'Replica':<15} {'Difference':<15}")
        print("-"*65)
        
        m_diff = abs(cavity['magnetization'] - replica['magnetization'])
        q_diff = abs(cavity['overlap'] - replica['overlap'])
        
        print(f"{'Magnetization':<20} {cavity['magnetization']:>14.6f} {replica['magnetization']:>14.6f} {m_diff:>14.6f}")
        print(f"{'Overlap':<20} {cavity['overlap']:>14.6f} {replica['overlap']:>14.6f} {q_diff:>14.6f}")
        
        if q_diff < 0.01:
            print("\n✓ Excellent agreement between methods!")
        elif q_diff < 0.05:
            print("\n✓ Good agreement between methods")
        else:
            print("\n⚠ Methods show significant difference")
    
    def visualize_comparison(self, results: Dict):
        """
        Cavity法とReplica法の詳細な比較可視化
        """
        cavity = results['cavity']
        replica = results['replica']
        beta = results['beta']
        T = results['temperature']
        
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. P(h)の比較
        ax1 = fig.add_subplot(gs[0, :2])
        h_range = cavity['h_range']
        ax1.plot(h_range, cavity['P_h'], linewidth=2.5, label='Cavity Method', 
                color='darkblue', alpha=0.8)
        ax1.plot(h_range, replica['P_h'], linewidth=2.5, label='Replica Method (Gaussian)', 
                color='darkred', linestyle='--', alpha=0.8)
        ax1.fill_between(h_range, cavity['P_h'], alpha=0.2, color='blue')
        ax1.fill_between(h_range, replica['P_h'], alpha=0.2, color='red')
        ax1.set_xlabel('h (effective field)', fontsize=12)
        ax1.set_ylabel('P(h)', fontsize=12)
        ax1.set_title(f'Field Distribution Comparison (c={self.c:.1f}, T={T:.3f})', 
                     fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # 2. 累積分布関数
        ax2 = fig.add_subplot(gs[0, 2])
        dh = h_range[1] - h_range[0]
        cdf_cavity = np.cumsum(cavity['P_h']) * dh
        cdf_replica = np.cumsum(replica['P_h']) * dh
        ax2.plot(h_range, cdf_cavity, linewidth=2, label='Cavity', color='darkblue')
        ax2.plot(h_range, cdf_replica, linewidth=2, label='Replica', 
                color='darkred', linestyle='--')
        ax2.set_xlabel('h', fontsize=11)
        ax2.set_ylabel('CDF', fontsize=11)
        ax2.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 局所磁化分布
        ax3 = fig.add_subplot(gs[1, 0])
        m_local = np.tanh(beta * h_range)
        ax3.plot(h_range, m_local * cavity['P_h'], linewidth=2, 
                label='Cavity', color='darkblue')
        ax3.plot(h_range, m_local * replica['P_h'], linewidth=2, 
                label='Replica', color='darkred', linestyle='--')
        ax3.set_xlabel('h', fontsize=11)
        ax3.set_ylabel('m(h) × P(h)', fontsize=11)
        ax3.set_title('Local Magnetization Distribution', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Cavity収束履歴
        ax4 = fig.add_subplot(gs[1, 1])
        history = cavity['history']
        ax4.semilogy(history['iteration'], history['diff'], 'o-', 
                    linewidth=2, markersize=4, color='darkblue')
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('||ΔP||', fontsize=11)
        ax4.set_title('Cavity Method Convergence', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. 物理量の収束
        ax5 = fig.add_subplot(gs[1, 2])
        ax5_twin = ax5.twinx()
        line1 = ax5.plot(history['iteration'], history['magnetization'], 'o-', 
                        linewidth=2, markersize=3, color='blue', label='m (Cavity)')
        line2 = ax5_twin.plot(history['iteration'], history['overlap'], 's-', 
                             linewidth=2, markersize=3, color='red', label='q (Cavity)')
        ax5.axhline(replica['magnetization'], color='blue', linestyle='--', 
                   alpha=0.5, label='m (Replica)')
        ax5_twin.axhline(replica['overlap'], color='red', linestyle='--', 
                        alpha=0.5, label='q (Replica)')
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Magnetization m', fontsize=11, color='blue')
        ax5_twin.set_ylabel('Overlap q', fontsize=11, color='red')
        ax5.set_title('Physical Quantities Evolution', fontsize=12, fontweight='bold')
        ax5.tick_params(axis='y', labelcolor='blue')
        ax5_twin.tick_params(axis='y', labelcolor='red')
        ax5.grid(True, alpha=0.3)
        
        # 凡例を統合
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, fontsize=9, loc='upper right')
        
        # 6. 数値比較表
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        comparison_text = f"""
        ╔═══════════════════════════════════════════════════════════════════════════════╗
        ║                          DETAILED COMPARISON                                   ║
        ╠═══════════════════════════════════════════════════════════════════════════════╣
        ║                                                                                ║
        ║  System Parameters:                                                            ║
        ║    Connectivity:           c = {self.c:.3f}                                      ║
        ║    Coupling strength:      J₀ = {self.J0:.3f}                                    ║
        ║    Temperature:            T = {T:.4f}                                         ║
        ║    Inverse temperature:    β = {beta:.4f}                                      ║
        ║                                                                                ║
        ║  ─────────────────────────────────────────────────────────────────────────  ║
        ║                                                                                ║
        ║  Physical Quantities:                                                          ║
        ║                                                                                ║
        ║                              Cavity          Replica        |Difference|       ║
        ║    Magnetization (m):     {cavity['magnetization']:>10.6f}    {replica['magnetization']:>10.6f}    {abs(cavity['magnetization']-replica['magnetization']):>10.6f}       ║
        ║    Overlap (q):           {cavity['overlap']:>10.6f}    {replica['overlap']:>10.6f}    {abs(cavity['overlap']-replica['overlap']):>10.6f}       ║
        ║                                                                                ║
        ║  ─────────────────────────────────────────────────────────────────────────  ║
        ║                                                                                ║
        ║  Distribution Properties:                                                      ║
        ║    Width (std):  Cavity = {np.sqrt(np.sum(cavity['P_h'] * h_range**2 * dh)):.4f},  Replica = {replica['sigma']:.4f}                    ║
        ║    Entropy:      Cavity = {cavity['entropy']:.4f}                                     ║
        ║                                                                                ║
        ║  ─────────────────────────────────────────────────────────────────────────  ║
        ║                                                                                ║
        ║  Agreement: {'✓ EXCELLENT' if abs(cavity['overlap']-replica['overlap'])<0.01 else '⚠ MODERATE' if abs(cavity['overlap']-replica['overlap'])<0.05 else '✗ POOR'}                                                           ║
        ║                                                                                ║
        ╚═══════════════════════════════════════════════════════════════════════════════╝
        """
        
        ax6.text(0.5, 0.5, comparison_text, fontsize=10, family='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.savefig(f'cavity_vs_replica_comparison_T{T:.2f}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


# ==================== 温度スキャン ====================

def temperature_scan_comparison(c: float = 3.0):
    """
    温度を変えながらCavity法とReplica法を比較
    """
    T_range = np.linspace(0.3, 3.0, 15)
    
    cavity_overlaps = []
    replica_overlaps = []
    cavity_mags = []
    replica_mags = []
    
    sg = SpinGlassComparison(c=c, J0=1.0)
    h_range = np.linspace(-6, 6, 250)
    
    print("\n" + "="*70)
    print("TEMPERATURE SCAN: Cavity vs Replica")
    print("="*70)
    
    for T in T_range:
        beta = 1.0 / T
        print(f"\n{'─'*70}")
        print(f"T = {T:.3f}")
        
        # Cavity
        cavity_result = sg.solve_cavity(beta, h_range, max_iter=80, tol=1e-3)
        
        # Replica
        replica_result = sg.solve_replica(beta, h_range)
        
        cavity_overlaps.append(cavity_result['overlap'])
        replica_overlaps.append(replica_result['overlap'])
        cavity_mags.append(abs(cavity_result['magnetization']))
        replica_mags.append(abs(replica_result['magnetization']))
        
        print(f"  Cavity:  q={cavity_result['overlap']:.4f}, |m|={abs(cavity_result['magnetization']):.4f}")
        print(f"  Replica: q={replica_result['overlap']:.4f}, |m|={abs(replica_result['magnetization']):.4f}")
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overlap
    ax = axes[0]
    ax.plot(T_range, cavity_overlaps, 'o-', linewidth=2.5, markersize=7, 
           color='darkblue', label='Cavity Method')
    ax.plot(T_range, replica_overlaps, 's--', linewidth=2.5, markersize=6, 
           color='darkred', label='Replica Method', alpha=0.8)
    ax.set_xlabel('Temperature T', fontsize=13)
    ax.set_ylabel('Edwards-Anderson Overlap q', fontsize=13)
    ax.set_title(f'Overlap vs Temperature (c={c:.1f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Magnetization
    ax = axes[1]
    ax.plot(T_range, cavity_mags, 'o-', linewidth=2.5, markersize=7, 
           color='darkblue', label='Cavity Method')
    ax.plot(T_range, replica_mags, 's--', linewidth=2.5, markersize=6, 
           color='darkred', label='Replica Method', alpha=0.8)
    ax.set_xlabel('Temperature T', fontsize=13)
    ax.set_ylabel('|Magnetization| |m|', fontsize=13)
    ax.set_title(f'Magnetization vs Temperature (c={c:.1f})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cavity_vs_replica_temperature_scan.png', dpi=300, bbox_inches='tight')
    plt.show()


# ==================== メイン実行 ====================

if __name__ == "__main__":
    # 1. 単一温度での詳細比較
    print("\n" + "╔"+"═"*68+"╗")
    print("║" + " "*15 + "CAVITY vs REPLICA COMPARISON" + " "*25 + "║")
    print("╚"+"═"*68+"╝")
    
    c = 3.0
    T = 1.0
    beta = 1.0 / T
    
    sg = SpinGlassComparison(c=c, J0=1.0)
    h_range = np.linspace(-6, 6, 300)
    
    results = sg.compare_methods(beta, h_range)
    sg.visualize_comparison(results)
    
    # 2. 温度スキャン
    temperature_scan_comparison(c=c)
    
    print("\n" + "="*70)
    print("All analyses complete!")
    print("="*70)