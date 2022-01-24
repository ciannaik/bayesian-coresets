#!/bin/bash

#python3 main.py --dataset synth_sparsereg plot coreset_size mu_err --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Mean Error"
#python3 main.py --dataset synth_sparsereg plot coreset_size Sig_err --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Cov Error"
#python3 main.py --dataset synth_sparsereg plot coreset_size imq_stein --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
#python3 main.py --dataset synth_sparsereg plot coreset_size gauss_stein --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
#python3 main.py --dataset synth_sparsereg plot coreset_size imq_mmd --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
#python3 main.py --dataset synth_sparsereg plot coreset_size gauss_mmd --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
#python3 main.py --dataset synth_sparsereg plot coreset_size fklw --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
#python3 main.py --dataset synth_sparsereg plot coreset_size rklw --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
#python3 main.py --dataset synth_sparsereg plot coreset_size t_build --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
#python3 main.py --dataset synth_sparsereg plot coreset_size t_per_sample --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"


#python3 main.py --dataset synth_sparsereg_large plot coreset_size mu_err --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Mean Error"
#python3 main.py --dataset synth_sparsereg_large plot coreset_size Sig_err --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Cov Error"
#python3 main.py --dataset synth_sparsereg_large plot coreset_size imq_stein --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
#python3 main.py --dataset synth_sparsereg_large plot coreset_size gauss_stein --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
#python3 main.py --dataset synth_sparsereg_large plot coreset_size imq_mmd --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
#python3 main.py --dataset synth_sparsereg_large plot coreset_size gauss_mmd --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
#python3 main.py --dataset synth_sparsereg_large plot coreset_size fklw --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
#python3 main.py --dataset synth_sparsereg_large plot coreset_size rklw --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
#python3 main.py --dataset synth_sparsereg_large plot coreset_size t_build --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
#python3 main.py --dataset synth_sparsereg_large plot coreset_size t_per_sample --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"

python3 main.py --dataset delays plot coreset_size mu_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Mean Error"
python3 main.py --dataset delays plot coreset_size Sig_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Cov Error"
python3 main.py --dataset delays plot coreset_size imq_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
python3 main.py --dataset delays plot coreset_size gauss_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
python3 main.py --dataset delays plot coreset_size imq_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
python3 main.py --dataset delays plot coreset_size gauss_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
python3 main.py --dataset delays plot coreset_size fklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
python3 main.py --dataset delays plot coreset_size rklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
python3 main.py --dataset delays plot coreset_size t_build --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
python3 main.py --dataset delays plot coreset_size t_per_sample --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"




