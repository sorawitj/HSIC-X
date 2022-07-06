library(AER)
library(ivpack)

card <- read.csv("card.csv")
card$exp_bin <- as.factor(card$exp_bin)

ols_model <- lm(lwage ~ educ + C(smsa66) + C(exp_bin) + C(black) + C(south66),
                data = card)

summary(ols_model)

iv_model <- ivreg(lwage ~ educ + smsa66 + C(exp_bin) + black + south66 |
                       nearc4 + smsa66 + C(exp_bin) + black + south66,
                     data = card, x = TRUE)
summary(iv_model)
anderson.rubin.ci(iv_model)

