import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# =====================================================
# ユーティリティ：グレースケール読み込み（今回は使わなくてもOK）
# =====================================================

def load_grayscale(path, target_size=None):
    """
    画像ファイルをグレースケール float32 (0..1) として読み込む。
    target_size=(H,W) が指定されていればリサイズ。
    """
    img = Image.open(path).convert("L")
    if target_size is not None:
        img = img.resize((target_size[1], target_size[0]), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr

# =====================================================
# 物体面の phantom（振幅 + 位相）を作る
# =====================================================

def generate_object_phantom(H=256, W=256, rng=None):
    """
    物体面の複素振幅 x_true(r) を自作する。
      - 振幅: 中央に円 + 斜めのバー
      - 位相: ゆっくり変化する 2D ランプ + 少し乱れ
    戻り値:
        x_true      : 物体面の複素場 (H,W)
        amp_true    : |x_true|
        phase_true  : arg x_true
        support     : 物体が存在する領域 (0/1)
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    X, Y = np.meshgrid(x, y)

    # 振幅：中心円
    R = np.sqrt(X**2 + Y**2)
    amp = np.zeros((H, W), dtype=np.float32)
    amp[R < 0.5] = 1.0

    # 斜めバーを追加
    bar = np.abs(X + Y) < 0.1
    amp[bar] = 1.0

    # support: 振幅>0 のところを 1 に
    support = (amp > 0).astype(np.float32)

    # 位相：ゆるい 2D ランプ + 少しランダム
    phase_ramp = 2.0 * np.pi * (0.3 * X + 0.1 * Y)  # なだらかな傾斜
    phase_noise = 0.3 * rng.normal(size=(H, W)) * support  # support 内だけ乱れ
    phase = (phase_ramp + phase_noise) * support

    # 複素場
    x_true = amp * np.exp(1j * phase)

    return x_true, amp, phase, support

# =====================================================
# Gerchberg-Saxton 法
# =====================================================

def gerchberg_saxton(
    mag_k,
    support_mask,
    n_iter=200,
    real_nonnegative=False,
    verbose=True,
    random_seed=0,
):
    """
    Gerchberg-Saxton 法で位相回復を行う。

    入力:
        mag_k        : フーリエ空間の振幅 |X(k)| (2D float, >=0)
        support_mask : 物体が存在しうる領域 (1: inside, 0: outside)
        n_iter       : 反復回数
        real_nonnegative : 物体面で実数・非負制約を課すかどうか
                           （屈折体なら実数>0, 散乱体なら複素でもよいなどで選ぶ）
    戻り値:
        x_rec        : 復元された物体面の複素振幅 (2D complex)
        hist_err     : 各反復での誤差指標（フーリエ振幅 MSE）
    """
    rng = np.random.default_rng(random_seed)
    H, W = mag_k.shape

    # 初期位相をランダムに与える
    rand_phase = rng.uniform(0.0, 2*np.pi, size=(H, W))
    Xk = mag_k * np.exp(1j * rand_phase)

    support_mask_c = support_mask.astype(np.float32)
    hist_err = []

    for it in range(n_iter):
        # --- 1. カメラ面 -> 物体面 ---
        x = np.fft.ifft2(Xk)

        # --- 2. 物体面で制約 ---
        if real_nonnegative:
            x_real = np.real(x)
            x_real[x_real < 0] = 0.0
            x_constrained = x_real * support_mask_c
        else:
            # support 外は 0 にするのみ（複素を許す）
            x_constrained = x * support_mask_c

        # --- 3. 物体面 -> カメラ面 ---
        Xk_new = np.fft.fft2(x_constrained)

        # --- 4. 測定振幅に差し替え、位相のみ更新 ---
        amp = np.abs(Xk_new)
        amp[amp == 0] = 1e-12  # ゼロ割回避
        Xk = mag_k * (Xk_new / amp)

        # --- 誤差評価（フーリエ振幅の MSE） ---
        err = np.mean((np.abs(Xk) - mag_k)**2)
        hist_err.append(err)

        if verbose and (it % 20 == 0 or it == n_iter-1):
            print(f"[GS] iter={it+1}/{n_iter}, amp_error={err:.3e}")

    x_rec = np.fft.ifft2(Xk)
    return x_rec, np.array(hist_err)

# =====================================================
# メイン：phantom → フーリエ強度 → GS → 真値と比較
# =====================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # ---- 1. 物体面 phantom を作る ----
    H, W = 256, 256
    x_true, amp_true, phase_true, support = generate_object_phantom(H, W, rng=rng)
    print("phantom generated:", x_true.shape)

    # ---- 2. フーリエ空間での強度を測定したと仮定 ----
    Xk_true = np.fft.fft2(x_true)
    I_k = np.abs(Xk_true)**2          # 測定強度
    mag_k = np.abs(Xk_true)           # 測定振幅

    # ノイズを入れたければここで I_k に Poisson/Gauss ノイズを足してもよい

    # ---- 3. GS 法で位相回復 ----
    x_rec, hist_err = gerchberg_saxton(
        mag_k,
        support_mask=support,   # phantom と同じ support を使ってみる
        n_iter=200,
        real_nonnegative=False, # phantom は複素なので False
        verbose=True,
        random_seed=0,
    )

    amp_rec = np.abs(x_rec)
    phase_rec = np.angle(x_rec)

    # 位相は 2π 周期なので、真値との差分を見るときは unwrap したり、
    # global 位相シフトを最小にするように調整すると良い。
    # ここでは簡単に、support 内での平均位相差を引いて合わせる。
    mask = (support > 0)
    phase_diff = phase_rec - phase_true
    # [-pi, pi] に折りたたむ
    phase_diff = (phase_diff + np.pi) % (2*np.pi) - np.pi
    global_shift = np.mean(phase_diff[mask])
    phase_rec_aligned = phase_rec - global_shift
    phase_diff_aligned = phase_rec_aligned - phase_true
    phase_diff_aligned = (phase_diff_aligned + np.pi) % (2*np.pi) - np.pi

    # ---- 4. 可視化 ----
    plt.figure(figsize=(12,8))

    plt.subplot(2,3,1)
    plt.title("True amplitude |x_true|")
    plt.imshow(amp_true, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,2)
    plt.title("True phase arg x_true")
    plt.imshow(phase_true * support, cmap="hsv")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(2,3,3)
    plt.title("Measured intensity |X(k)|^2")
    plt.imshow(np.log1p(I_k), cmap="gray")  # dynamic range のため log
    plt.axis("off")

    plt.subplot(2,3,4)
    plt.title("Reconstructed amplitude |x_rec|")
    plt.imshow(amp_rec, cmap="gray")
    plt.axis("off")

    plt.subplot(2,3,5)
    plt.title("Reconstructed phase (aligned)")
    plt.imshow(phase_rec_aligned * support, cmap="hsv")
    plt.colorbar()
    plt.axis("off")

    plt.subplot(2,3,6)
    plt.title("Phase error (aligned)")
    plt.imshow(phase_diff_aligned * support, cmap="bwr", vmin=-np.pi, vmax=np.pi)
    plt.colorbar()
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ---- 5. 誤差の減衰 ----
    plt.figure()
    plt.semilogy(hist_err)
    plt.xlabel("iteration")
    plt.ylabel("MSE(|X_k|-|X_k^true|)")
    plt.title("GS convergence (Fourier amplitude error)")
    plt.tight_layout()
    plt.show()

    # support 内だけの振幅誤差・位相誤差をざっくり表示
    amp_err = np.mean((amp_rec[mask] - amp_true[mask])**2)
    phase_err = np.mean(phase_diff_aligned[mask]**2)
    print(f"Amplitude MSE (object, support region) = {amp_err:.3e}")
    print(f"Phase MSE (object, support region)     = {phase_err:.3e}")
