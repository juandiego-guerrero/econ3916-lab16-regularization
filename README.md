# econ3916-lab16-regularization
Lab 16 Submission

# Regularization: Buying Bias to Sell Variance

## Objective
I tested how LASSO and Ridge trade a little bias for a large cut in variance when predictors outnumber training rows, and checked an AI-written cross-validation loop against my own.

## Methodology
- Simulated 150 rows and 200 standard-normal predictors, where only the first 5 had nonzero coefficients, then split 70/30, leaving 105 training rows.
- Fitted OLS, then LASSO and Ridge, each in a pipeline with a scaler and its penalty chosen by 5-fold cross-validation.
- Compared training MSE, test MSE and the number of nonzero coefficients across the three models.
- Traced the LASSO coefficient path over 60 penalties to see when the true predictors and the first noise variable entered.
- Chose the LASSO penalty with 5-fold shuffled cross-validation, refitting the scaler inside every fold so the held-out rows never influenced their own scaling.
- Scored LASSO's selected variables against the known truth.
- Built a second design where two predictors were near-duplicates and both mattered, and compared how LASSO and Ridge handled the pair.
- Wrote my own prompt for an AI to write the cross-validation loop as an explicit function, and checked its λ and CV MSE against my results on the full grid and on every second penalty.

## Key Findings
- **OLS overfits completely.** With 200 predictors and 105 rows, it fit the training data perfectly but had a test MSE of 12.9587.
- **Both penalties helped, LASSO far more.** Ridge cut the test MSE to 10.1128 (−22%), and LASSO cut it to 1.4219 (−89%). Because the true model is sparse, LASSO could set most noise coefficients exactly to zero, while Ridge only shrank them and kept all 200.
- **The path separated signal from noise cleanly.** All five true predictors were in from step 6, and the first noise variable did not enter until step 17. That clean gap comes from strong true effects and nearly uncorrelated columns, which real data rarely has.
- **Cross-validation chose λ = 0.1995**, with a CV MSE of 1.5502.
- **Selection was good but not exact.** LASSO found 5 of 5 true predictors, along with 11 false positives.
- **With near-duplicates, LASSO picked one and Ridge split the weight.** LASSO put 4.462 on one predictor and zero on the other, while Ridge gave them 1.618 and 1.609. LASSO's choice is fine if the two columns measure the same thing, but misleading if they are different things.
- **The AI's loop matched mine exactly.** I went 2 rounds. Both returned λ = 0.1995 and CV MSE = 1.5502 on the full grid, and λ = 0.1585 and CV MSE = 1.5810 on every second penalty, all +0% apart. I stopped at round 2 because the loop followed the same logic as GridSearchCV and hard-coded nothing about this dataset.
