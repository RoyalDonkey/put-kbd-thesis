#! /bin/sh
printf "Saving plots, please wait...\n"
python best_models_per_dataset.py saved_plots/best_models_per_train_test.jpg > /dev/null
python best_models_per_dataset.py saved_plots_pdf/best_models_per_train_test.pdf > /dev/null

printf "7\n1\n" | python best_acc.py > /dev/null
printf "7\n2\n" | python best_acc.py > /dev/null
printf "7\n1\n" | python datasets_performance.py > /dev/null
printf "7\n2\n" | python datasets_performance.py > /dev/null
printf "7\n1\n" | python model_comparison.py > /dev/null
printf "7\n2\n" | python model_comparison.py > /dev/null
printf "7\n1\n" | python model_summary.py > /dev/null
printf "7\n2\n" | python model_summary.py > /dev/null