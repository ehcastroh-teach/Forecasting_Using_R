####-------------------------------------------------------------------####
#   title:   Forecasting Automobile Sales (Elantra)                       #
#   purpose: Perform feature selection,                                   #
#            and perform a linear regression to forecast sales            #
#            of the Hyundai Elantra                                       #
#                                                                         #
#   notes:   For simplicity, data is loaded as needed with new names      #
####-------------------------------------------------------------------####

#### Load library dependencies ####
# Data manipulation
library(dplyr)
# Plotting
library(ggplot2)
library(GGally)
# Variable Inflation Factor (VIF)
library(car)

# Ensure images output directory exists
dir.create("images", showWarnings = FALSE)

# Load data
elantra <- read.csv("elantra_sales.csv")
head(elantra)


#### Linear filtering of time series, by chronological order (year) ####
elantra.train <- filter(elantra, Year < 2015)
elantra.test  <- filter(elantra, Year > 2014)
head(elantra.train)
head(elantra.test)


#### Train model using linear regression: lm(<output> ~ predictors, data = <dataset>) ####
ElantraSales <- lm(ElantraSales ~ Unemployment + ElantraQueries + CPI.Energy + CPI.All, data = elantra.train)
# Identify subset of features to consider, based on significance codes
summary(ElantraSales)


#### Testing quality of model estimators ####
# 1. Confidence interval plot
ggcoef(
  ElantraSales,
  vline_color = "red",
  vline_linetype =  "solid",
  errorbar_color = "blue",
  errorbar_height = .25,
  exclude_intercept = TRUE
)
ggsave("images/CI_plot_naive.png", width = 8, height = 6)

# 2. Variance Inflation Factor
vif(ElantraSales)


#### New model, excluding non-relevant predictors: CPI.Energy and Unemployment ####
sub_ElantraSales <- lm(ElantraSales ~ ElantraQueries + CPI.All, data = elantra.train)
summary(sub_ElantraSales)


#### Testing quality of narrowed model ####
# 1. Confidence interval plot
ggcoef(
  sub_ElantraSales,
  vline_color = "red",
  vline_linetype =  "solid",
  errorbar_color = "orange",
  errorbar_height = .25,
  exclude_intercept = TRUE
)
ggsave("images/narrowed_model.png", width = 8, height = 6)

# 2. Variance Inflation Factor
vif(sub_ElantraSales)


#### Evaluate narrowed model on test data ####
# Compute predicted values
predictions_ElantraSales <- predict(sub_ElantraSales, newdata = elantra.test)
summary(predictions_ElantraSales)

# Sum of Squared Errors from regression model (Residual Sum of Squares)
SSE_a <- sum((elantra.test$ElantraSales - predictions_ElantraSales)^2)
# Sum of Squared Errors from baseline model (Total Sum of Squares)
SST_a <- sum((elantra.test$ElantraSales - mean(elantra.train$ElantraSales))^2)
# Out-of-Sample R-squared
OSR2_a <- 1 - SSE_a / SST_a
OSR2_a


# Load dataset with month numeric index and MonthFactor
elantra_plot <- read.csv("elantra_sales_monthly.csv")
head(elantra_plot)

#### EDA: sales by month ####
ggplot(elantra_plot, aes(x = MonthNumeric, y = ElantraSales, color = MonthFactor)) +
  geom_point() + xlab("2010 through 2017") + ylab("Elantra Units Sold")
ggsave("images/variable_error_plot.png", width = 8, height = 6)

#### Seasonal model: add MonthFactor to capture yearly buying cycles ####
seasonal_ES <- lm(ElantraSales ~ MonthFactor + Unemployment +
                    ElantraQueries + CPI.Energy + CPI.All, data = elantra.train)

#### Identify subset of features to consider ####
# 1. Confidence interval plot
ggcoef(
  seasonal_ES,
  vline_color = "red",
  vline_linetype =  "solid",
  errorbar_color = "blue",
  errorbar_height = .25,
  exclude_intercept = TRUE)
ggsave("images/seasonal_model.png", width = 8, height = 6)

# 2. Variance Inflation Factor
vif(seasonal_ES)

## Second iteration: remove Unemployment ##
seasonal_ES2 <- lm(ElantraSales ~ MonthFactor + ElantraQueries + CPI.Energy + CPI.All, data = elantra.train)
vif(seasonal_ES2)

## Third iteration: remove CPI.Energy ##
seasonal_ES3 <- lm(ElantraSales ~ MonthFactor + ElantraQueries + CPI.All, data = elantra.train)
vif(seasonal_ES3)
ggcoef(seasonal_ES3, vline_color = "red", vline_linetype = "solid", errorbar_color = "blue", errorbar_height = .25, exclude_intercept = TRUE)
ggsave("images/seasonality_lm.png", width = 8, height = 6)
summary(seasonal_ES3)

## Evaluate seasonal model on test data ##
prediction_seasonal_ES <- predict(seasonal_ES3, newdata = elantra.test)
summary(prediction_seasonal_ES)


#### Quantitative evaluation of seasonal model ####
SSE_b <- sum((elantra.test$ElantraSales - prediction_seasonal_ES)^2)
SST_b <- sum((elantra.test$ElantraSales - mean(elantra.train$ElantraSales))^2)
OSR2_b <- 1 - SSE_b / SST_b
OSR2_b


#### Combined model: best subset from narrow and seasonal experiments ####
combination_ES <- lm(ElantraSales ~ MonthFactor + CPI.All, data = elantra.train)
vif(combination_ES)
ggcoef(combination_ES, vline_color = "red", vline_linetype = "solid", errorbar_color = "blue", errorbar_height = .25, exclude_intercept = TRUE)
ggsave("images/combined_model.png", width = 8, height = 6)
summary(combination_ES)


#### Evaluate combined model on test data ####
prediction_combination_ES <- predict(combination_ES, newdata = elantra.test)
summary(prediction_combination_ES)

#### Quantitative evaluation of combined model ####
SSE_c <- sum((elantra.test$ElantraSales - prediction_combination_ES)^2)
SST_c <- sum((elantra.test$ElantraSales - mean(elantra.train$ElantraSales))^2)
OSR2_c <- 1 - SSE_c / SST_c
OSR2_c


# Load dataset with Real GDP (RGDP) as an additional predictor
elantra3 <- read.csv("elantra_sales_rgdp.csv")
head(elantra3)

#### Train/test split for RGDP dataset ####
elantra3.train <- filter(elantra3, Year < 2015)
elantra3.test  <- filter(elantra3, Year > 2014)
head(elantra3.train)
head(elantra3.test)

#### Model with RGDP as a predictor ####
custom_ES3 <- lm(ElantraSales ~ MonthFactor + ElantraQueries + CPI.Energy + RGDP, data = elantra3.train)
vif(custom_ES3)
ggcoef(custom_ES3, vline_color = "red", vline_linetype = "solid", errorbar_color = "blue", errorbar_height = .25, exclude_intercept = TRUE)
ggsave("images/rgdp_model.png", width = 8, height = 6)
summary(custom_ES3)

#### Evaluate RGDP model on test data ####
prediction_custom_ES <- predict(custom_ES3, newdata = elantra3.test)
summary(prediction_custom_ES)

#### Quantitative evaluation of RGDP model ####
SSE_d <- sum((elantra3.test$ElantraSales - prediction_custom_ES)^2)
SST_d <- sum((elantra3.test$ElantraSales - mean(elantra3.train$ElantraSales))^2)
OSR2_d <- 1 - SSE_d / SST_d
OSR2_d

# Final prediction: Elantra sales for August 2017
summary(seasonal_ES3)
summary(combination_ES)
# Average ElantraQueries for Jan-Jul 2017
i <- 92
# Average CPI.All for Jan-Jul 2017
j <- 244.03
# Actual reported sales
act <- 15127

# MonthFactor coefficients act as binary switches (0 or 1) per month.
# August gets a 1; all others get 0. CPI.All uses the Jan-Jul 2017 average.
Aug2017_ES_combined <- ((-84309.21) - 7011.47*(0) - 5063.52*(0) + 446.33*(0) + 416.636*(0) - 496.86*(0) + 696.16*(0)
                                    - 2.97*(1) - 3122.47*(0) - 6600.71*(0) - 6039.35*(0) - 2505.80*(0) + 452.00*(j))
Error <- abs(Aug2017_ES_combined - act)
Error
####-------------------------------------------------------------------####
