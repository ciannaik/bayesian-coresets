#!/bin/bash

python3 main.py --data_num 1000 --data_dim 10 plot coreset_size mu_err --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Mean Error"
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size Sig_err --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Cov Error"
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size imq_stein --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size gauss_stein --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size imq_mmd --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size gauss_mmd --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size fklw --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size rklw --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size t_build --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
python3 main.py --data_num 1000 --data_dim 10 plot coreset_size t_per_sample --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"


#python3 main.py plot Ms fklw --groupby Ms --summarize trial --plot_type line --plot_x_type linear --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Forward KL"
#python3 main.py plot csizes fklw --groupby Ms --summarize trial --plot_type line --plot_x_type linear --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
#python3 main.py plot cputs fklw --groupby Ms --summarize trial --plot_type line --plot_x_type linear --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Forward KL"
#
#python3 main.py plot Ms mu_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Relative Mean Error"
#python3 main.py plot csizes mu_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Mean Error"
#python3 main.py plot cputs mu_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Relative Mean Error"
#
#python3 main.py plot Ms Sig_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Relative Covariance Error"
#python3 main.py plot csizes Sig_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Relative Covariance Error"
#python3 main.py plot cputs Sig_errs --summarize trial --groupby Ms --plot_x_type linear --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Relative Covariance Error"
#






