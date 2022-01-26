### Distribution generalization
library(dHSIC)
library(ggplot2)

# Data generating processs
causal_par <- c(1, 1)
generate_data <- function(n, int_par){
  indicator <- rbinom(n, 1, int_par/4)
  Z1 <- runif(n, 0, int_par)
  Z2 <- runif(n, int_par, 4)
  Z <- Z1
  Z[indicator==1] <- Z2[indicator==1]
  U1 <- rnorm(n)
  U2 <- rnorm(n)
  X1 <- Z + U1*(Z<=3.8) + rnorm(n)
  X2 <- U2 + rnorm(n)
  Y <- causal_par[1]*X1 + causal_par[2]*X2 + U1 + U2
  return(list(Y=Y,
              X=cbind(X1, X2),
              Z=Z,
              U=U))
}

# Generate a sample from the train data
data_train <- generate_data(1000, 0.5)
Ztrain <- data_train$Z
Xtrain <- data_train$X
Ytrain <- data_train$Y

# Fit OLS
fitlm <- lm(Ytrain ~ -1 + Xtrain)
summary(fitlm)

# Define loss for HSIC-IV
loss_fun <- function(beta, lambda=10^4, gamma=1,
                     Ytrain=Ytrain,
                     Xtrain=Xtrain,
                     Ztrain=Ztrain){
  R <- Ytrain - Xtrain %*% matrix(beta, nrow=2)
  hsic <- dhsic(R, Ztrain, kernel=c("gaussian", "gaussian"))$dHSIC
  mse <- mean(R^2)
  return(hsic*lambda + mse*gamma)
}

# Fit HSIC-IV using optim
opt <- optim(coefficients(fitlm),
             function(x) loss_fun(x, 10^4, 1,
                                  Ytrain, Xtrain, Ztrain))
print(opt)
print(coefficients(fitlm))
print(opt$par)

## Experiment
B <- 10
intvec <- c(1, 2, 3, 3.5, 3.9, 3.99)
loss_mat1 <- matrix(NA, length(intvec), B)
loss_mat2 <- matrix(NA, length(intvec), B)
loss_mat3 <- matrix(NA, length(intvec), B)
for(k in 1:B){
  print(k)
  # train
  data_train <- generate_data(1000, 0.5)
  Ztrain <- data_train$Z
  Xtrain <- data_train$X
  Ytrain <- data_train$Y
  # OLS
  fitlm <- lm(Ytrain ~ -1 + Xtrain)
  # HSIC-IV
  opt <- optim(coefficients(fitlm),
             function(x) loss_fun(x, 10^4, 1,
                                  Ytrain, Xtrain, Ztrain))
  # Evaluate on test distributions
  for(i in 1:length(intvec)){
    data_test <- generate_data(1000,intvec[i])
    Ztest <- data_test$Z
    Xtest <- data_test$X
    Ytest <- data_test$Y
    loss_mat1[i, k] <- mean((Ytest - Xtest %*% coefficients(fitlm))^2)
    loss_mat2[i, k] <- mean((Ytest - Xtest %*% opt$par)^2)
    loss_mat3[i, k] <- mean((Ytest - Xtest %*% causal_par)^2)
  }
}


# Plot results
df1 <- c(as.numeric(loss_mat1),
         as.numeric(loss_mat2),
         as.numeric(loss_mat3))
df2 <- rep(rep(1:length(intvec),B), 3)
df3 <- rep(c("OLS", "HSIC-IV", "Causal"), each=B*length(intvec))
df <- as.data.frame(cbind(df1, df2, df3))
df$df1 <- df1
colnames(df) <- c("mse", "intstrength", "method")

plt <- ggplot(df, aes(x=intstrength, y=mse, fill=method))
plt <- plt + geom_boxplot()
plt <- plt + xlab("intervention strength") + ylab("mean squared test error")
ggsave(plt, filename="distribution_generalization_example.pdf")
