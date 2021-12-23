#!/bin/bash

#for dnm in "synth_lr" "phishing" "ds1"
#do
#    python3 main.py --model lr --dataset $dnm plot Ms Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "F-Norm Error"
#    python3 main.py --model lr --dataset $dnm plot csizes Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "F-Norm Error"
#    python3 main.py --model lr --dataset $dnm plot cputs Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "F-Norm Error"
#done
#
#for dnm in "synth_lr" "phishing" "ds1"
##for dnm in "ds1"
#do
#    python3 main.py --model lr --dataset $dnm plot Ms rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Reverse KL"
#    python3 main.py --model lr --dataset $dnm plot csizes rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
#    python3 main.py --model lr --dataset $dnm plot cputs rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Reverse KL"
#done

#for dnm in "synth_poiss" "biketrips" "airportdelays"
#do
#    python3 main.py --model poiss --dataset $dnm plot Ms Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "F-Norm Error"
#    python3 main.py --model poiss --dataset $dnm plot csizes Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "F-Norm Error"
#    python3 main.py --model poiss --dataset $dnm plot cputs Fs --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "F-Norm Error"
#done


for dnm in "synth_poiss" "biketrips" "airportdelays"
#for dnm in "synth_poiss"
do
    python3 main.py --model poiss --dataset $dnm plot Ms rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label "Reverse KL"
    python3 main.py --model poiss --dataset $dnm plot csizes rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label "Reverse KL"
    python3 main.py --model poiss --dataset $dnm plot cputs rklw --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label "Reverse KL"
done
