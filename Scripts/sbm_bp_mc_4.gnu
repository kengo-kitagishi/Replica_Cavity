# plot_all.gnu
set term pdfcairo enhanced font "Arial,12"
set output "all_figures.pdf"

# =====================================================
# 図1: overlap vs lambda
# =====================================================
set multiplot layout 3,2 title "SBM 解析結果" font ",14"
set xlabel "lambda = c * eps^2"
set ylabel "overlap m"
set key outside

plot "overlap_vs_lambda.dat" using 1:2 with linespoints lt 1 pt 7 lw 2 title "NB (2nd eig)", \
     "overlap_vs_lambda.dat" using 1:3 with linespoints lt 2 pt 5 lw 2 title "MC (Nishimori)"

# =====================================================
# 図2: overlap vs c_out/c_in
# =====================================================
set xlabel "c_out/c_in"
set ylabel "overlap m"
plot "overlap_vs_r.dat" using 1:2 with linespoints lt 1 pt 7 lw 2 title "NB (2nd eig)", \
     "overlap_vs_r.dat" using 1:3 with linespoints lt 2 pt 5 lw 2 title "MC (Nishimori)"

# =====================================================
# 図3: BP収束回数 vs lambda
# =====================================================
set xlabel "lambda = c * eps^2"
set ylabel "BP convergence iterations"
plot "bp_iterations.dat" using 1:2 with linespoints lt 3 pt 7 lw 2 title "BP iterations"

# =====================================================
# 図4: NBスペクトルギャップ vs lambda
# =====================================================
set xlabel "lambda = c * eps^2"
set ylabel "NB spectral gap"
plot "nb_gap.dat" using 1:2 with linespoints lt 4 pt 7 lw 2 title "NB 2nd eig gap"

# =====================================================
# 図5: BPメッセージ収束曲線
# =====================================================
set xlabel "iteration"
set ylabel "max |delta message|"
unset key
n_eps = system("head -1 bp_norms.dat | wc -w")  # 列数 = eps数
do for [i=1:n_eps] {
    plot "bp_norms.dat" using 0:i with lines lw 2 title sprintf("eps col %d", i)
}

# =====================================================
# 図6: NB 2nd eigenvector node_score 分布
# =====================================================
set xlabel "node_score"
set ylabel "count"
set style data histograms
set style fill solid 0.5
plot "nb_scores.dat" using 2:($2) index 0 with boxes notitle

# =====================================================
# 図7: MC magnetization 分布
# =====================================================
set xlabel "MC magnetization m"
set ylabel "count"
set style data histograms
set style fill solid 0.5
plot "mc_mags.dat" using 2:($2) index 0 with boxes notitle

unset multiplot
set output
