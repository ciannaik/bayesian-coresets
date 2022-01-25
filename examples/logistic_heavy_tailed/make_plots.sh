#!/bin/bash

#for dnm in "synth_lr" "phishing" "ds1"
#do
#    python3 main.py --model lr --dataset $dnm plot Ms Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "F-Norm Error"
#    python3 main.py --model lr --dataset $dnm plot csizes Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "F-Norm Error"
#    python3 main.py --model lr --dataset $dnm plot cputs Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "F-Norm Error"
#done

#for dnm in "synth_lr" "phishing" "ds1"
for dnm in "criteo"
do
#    python3 main.py --model lr --dataset $dnm plot coreset_size mu_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Mean Error"
#    python3 main.py --model lr --dataset $dnm plot coreset_size Sig_err --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Cov Error"
    python3 main.py --model lr --dataset $dnm plot coreset_size mu_err_full --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Full Mean Error"
    python3 main.py --model lr --dataset $dnm plot coreset_size Sig_err_full --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Full Cov Error"
    python3 main.py --model lr --dataset $dnm plot coreset_size imq_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ Stein Discr."
#    python3 main.py --model lr --dataset $dnm plot coreset_size gauss_stein --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss Stein Discr."
#    python3 main.py --model lr --dataset $dnm plot coreset_size imq_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "IMQ MMD Discr."
#    python3 main.py --model lr --dataset $dnm plot coreset_size gauss_mmd --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Gauss MMD Discr."
    python3 main.py --model lr --dataset $dnm plot coreset_size fklw --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Forward KL"
    python3 main.py --model lr --dataset $dnm plot coreset_size rklw_full --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
	python3 main.py --model lr --dataset $dnm plot coreset_size t_build --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Build Time (s)"
	python3 main.py --model lr --dataset $dnm plot coreset_size t_per_sample --plot_toolbar --groupby coreset_size --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Per Sample Time (s)"
done

#for dnm in "synth_poiss" "biketrips" "airportdelays"
#do
#    python3 main.py --model poiss --dataset $dnm plot Ms Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "F-Norm Error"
#    python3 main.py --model poiss --dataset $dnm plot csizes Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "F-Norm Error"
#    python3 main.py --model poiss --dataset $dnm plot cputs Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "F-Norm Error"
#done


##for dnm in "synth_poiss" "biketrips" "airportdelays"
#for dnm in "synth_poiss_large"
#do
#    python3 main.py --model poiss --dataset $dnm plot Ms rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Reverse KL"
#    python3 main.py --model poiss --dataset $dnm plot csizes rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
#    python3 main.py --model poiss --dataset $dnm plot cputs rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Reverse KL"
#done
