Online Resource 1
Run-level results, clipping measurements, CPU timings and numerical checks for private federated personalization.

Clipping sensitivity and private prior correction in federated classifier personalization
Pattern Analysis and Applications
Hanchen Ren, Le Yu, Xuru Li, Xinyu Meng, Yu Xie and Lifei Wei
College of Information Engineering, Shanghai Maritime University, Shanghai 201306, China
Corresponding author: Lifei Wei (lfwei@shmtu.edu.cn)

Accuracy columns are fractions unless a difference is specified in percentage points. The main analysis treats one run, averaging 20 clients, as one observation. The 95% intervals use paired run-level differences and four degrees of freedom.

01_main_accuracy.csv (90 rows)
Tables 3 and 7. Five matched runs for each of 18 settings. B0 is the shared checkpoint, B1 output bias, B2 the full head, B3 all biases and B4 the pilot-selected prior. B4_05 and B4_1 use fixed prior strengths. Accuracies are fractions.

02_clipping_sweep.csv (200 rows)
Figure 2. Validation and test accuracy, cross-entropy and sampled-record clipping fraction by dataset, seed, method and threshold.

03_whole_budget.csv (20 rows)
Table 4. Matched run-level scores. B0_full spends the full budget on shared training. The diagnostic CIFAR-10 seeds differ from the main matrix.

04_cpu_timing.csv (180 rows)
Table 6. Local CPU timings in seconds, five timed repetitions per client and method. Rows preserve measurement order within each client and method.

05_score_noise.csv (360 rows)
Figure 3 and Section 4.6. Per-image Gaussian score perturbations and class-pair reversals. The variance columns compare empirical and analytic values.

06_update_moments.csv (6 rows)
Table 8. Empirical and theoretical moments of the private output-bias gradient estimate by client and clipping threshold.

07_finite_step.csv (18 rows)
Table 9. Trajectory-averaged training objective gaps, clipping terms, distance terms and bounds by client and threshold.

08_prior_extension.csv (25 rows)
Section 4.7 and Table 10. Accuracy, cross-entropy, privacy budget and training communication steps for the fresh private histogram (H-full), initialization (H-init), local and shared scaling, and the regularized bias residual (Bias-ref).

09_prior_extension_differences.csv (5 rows)
Section 4.7. Within-run accuracy differences in percentage points between the two methods named in each column.

Each CSV includes article_title, journal, authors, affiliation, corresponding_author and corresponding_email metadata columns. The original data fields retain their reported values.

10_paired_clipping.csv (20 rows)
Table 5. Matched bias-minus-head accuracy gaps at thresholds 1 and 4, their change and the run reversal indicator.
