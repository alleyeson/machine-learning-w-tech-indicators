install.packages("e1071") ## svm
library(e1071) ##for svm
getwd()
setwd("/Users/allisonemono/desktop/R-sessions")
install.packages("smooth")
require(smooth)
require(Mcomp)
install.packages("Mcomp")
data_goog_input_train <- read.csv("goog_05_17_input_train_p.csv")
data_goog_input_test <- read.csv("goog_input_18_test_p.csv")
data_goog_out_train <- read.csv("Google_Ouptut_Train_Sign.csv")
data_goog_out_test <- read.csv("Google_test_out_Sign.csv")

head(data_goog_out_train)

####DECLARE WINDOW#
w = 10
head(data_goog_input_train)
N = nrow(data_goog_input_train)

#####Checking validitiy of written indicators ######
######## RUNNING SVM ##################

w = 10
N = nrow(data_goog_input_train)

#################### Indicators ####################################
######### MOVING AVERAGE ###########################
moving_average <- function(x, w) {
  M <- nrow(x)
  x<- data.frame(x)
  move_avg <- matrix(nrow = M-w+1, ncol = 1)
  
  for(i in seq(from =(w), to = M, by = 1) )
  {
    move_avg[i-w+1,] <- mean(x[(i-w):i,])
  }
  
  return(move_avg)
}
######################################################
######### MOMENTUM ##################
momentum_funct <- function(x, w){
  M <- nrow(x)
  x<- data.frame(x)
  mom <- matrix(nrow = M-w+1, ncol = 1)
  
  for(i in seq(from =(w), to = M, by = 1) )
  {
    mom[i-w+1,] <- x[(i),] - x[(i-w+1), ]
  }
  
  return(mom)
}
#############################
######### Stochastic K ##########################
stochastic_K_funct <- function(x,w){
  
  N <- nrow(x)
  x <- data.frame(x)
  
  stochastic_k <- matrix(nrow= N-w+1, ncol = 1)
  for (i in seq(from =w, to = (N), by = 1) ) {
    
    row_max = max(x[(i-w):(i),])
    row_min = min(x[(i-w):(i),])
    
    
    c_t = x[(i),] 
    denum = (row_max-row_min)
    numera = c_t-row_min
    
    if(denum == 0) {
      denum = 100 
    } else {denum = (row_max-row_min)}
    
    stochastic_k[i-w+1,] <- (numera/(denum))*100
  }
  
  return(stochastic_k)
}
####################################################################
############### Accumulative Distribution #############################################
accum_Dist <- function(x,w){
  x <- data.frame(x)
  N <- nrow(x)
  acc <- matrix(nrow = N-w+1, ncol = 1)
  for (i in seq(from =w, to = (N), by = 1)) {
    
    h_i = max(x[1:i, ])
    l_i = min(x[1:i, ])
    
    c_t_1 = x[i-1,]
    
    denum = h_i - l_i 
    numer = h_i - c_t_1
    
    acc[i-w+1,] = numer/denum
    
  }
  
  return(acc)
}
############################################################
################LARRY WILLIAMS#####################
larry_will_func <- function(x, w){
  x<- data.frame(x)
  N <- nrow(x)
  larry <- matrix(nrow = N-w+1, ncol= 1)
  for (i in seq(from =w, to = (N), by = 1)) {
    h_i <- max(x[1:i,])
    l_i  <- min(x[1:i,])
    c_i <- x[i,]
    numerator <- h_i - c_i
    denominator <- h_i - l_i
    
    if(denominator == 0 ){
      denominator = 100
    } else {denominator <- h_i - l_i}
    
    larry[i-w+1,] <- (numerator/denominator)*100
  }
  return(larry)
}
###########################################
## DATA PREP 

data_train_in <- data_goog_input_train[w:N-w+1,]
head(data_train_in)
nrow(data_train_in)
ma <- moving_average(data_goog_input_train,w)
stoch_k <- stochastic_K_funct(data_goog_input_train,w)
mom <- momentum_funct(data_goog_input_train,w)
lar <- larry_will_func(data_goog_input_train,w)
accum <- accum_Dist(data_goog_input_train,w)
nrow(accum)

data_test_input <- data.frame(input = data_train_in,
                           move_average = ma,
                           stochasticK = stoch_k, 
                           momentum_mom = mom, 
                           Larry_will = lar, 
                           accumDist = accum
                           )
head(data_test_input)
ret_train <- data_goog_out_train[w:N-w+1,] 
data_train = data.frame(t = factor(ret_train), data_test_input) ## combine dependant var and indep vars for training 
## replacing former dat data. Put more x as indicators
head(data_train)
#radial Kernel 
fit_goog = svm(factor(t) ~., data = data_train, cross = 10,scale = TRUE, kernel = "radial", cost = 10, gamma = 2.5)

summary(fit_goog)

##polynomial Kernel
fit_goog_poly = svm(factor(t) ~., data = data_train[1:((N-1)/2),], cross = 10, scale = TRUE, kernel = "polynomial", degree = 2, cost = 10, gamma = 2.5)

summary(fit_goog_poly)

################## TEST SET #############################
##head(fit_goog$)

####
pred_radial_svm <- predict(fit_goog, input = data.frame(data_goog_input_test))
?predict

nrow(data.frame(data_goog_input_test))

head(pred_radial_svm)
predict_compare_radial <- data.frame(depen_var = data_goog_out_test, model_ouput = pred_radial_svm[1:nrow(data.frame(data_goog_input_test))] )

write.csv(predict_compare_radial, "Radial_Output_Performance2.csv") #collect ouput to csv file

pred_poly_svm <- predict(fit_goog_poly, input = data.frame(data_goog_input_test))
predict_compare_poly<- data.frame(dep_var = data_goog_out_test, model_poly_out = pred_poly_svm[1:nrow(data.frame(data_goog_input_test))])

head(predict_compare_poly)
write.csv(predict_compare_poly, "Poly_Output_Performance.csv")  ##collect output to csv file

table(pred = pred_radial_svm[1:nrow(data.frame(data_goog_input_test))], true = data_goog_input_test)

head(data_goog_input_test)
M <- nrow(data_goog_input_test)
data_test_in <- data_goog_input_test[w:M-w+1,]
nrow(data.frame(data_test_in))
ma <- moving_average(data_goog_input_test,w)
stoch_k <- stochastic_K_funct(data_goog_input_test,w)
mom <- momentum_funct(data_goog_input_test,w)
lar <- larry_will_func(data_goog_input_test,w)
accum <- accum_Dist(data_goog_input_test,w)

