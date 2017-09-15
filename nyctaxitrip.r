# Abertura dos arquivos
train = read.csv(file = "~/Desktop/train.csv", header = TRUE, sep = ",")
test = read.csv(file = "~/Desktop/test.csv", header = TRUE, sep = ",")

## Pré Processamento
train <- train[3:11]
test <- test[3:8]
train$dropoff_datetime <- NULL
test$passenger_count <- NULL

library(lubridate)

# Divisão data/hora
try <- data.frame(do.call('rbind', strsplit(as.character(train$pickup_datetime),' ',fixed = TRUE)))
trytest <- data.frame(do.call('rbind', strsplit(as.character(test$pickup_datetime),' ',fixed = TRUE)))

try2 <- data.frame(do.call('rbind', strsplit(as.character(try$X2),':',fixed = TRUE)))
trytest2 <- data.frame(do.call('rbind', strsplit(as.character(trytest$X2),':',fixed = TRUE)))

try$X2 <- NULL
trytest$X2 <- NULL

train <- cbind(try,try2,train)
test <- cbind(trytest,trytest2,test)

colnames(train)[1] <- "doy" # Day of Year
colnames(train)[2] <- "hour"
colnames(test)[2] <- "hour"


dow <- 0 # Day of Week
train <- cbind(dow,train)

train$passenger_count <- NULL
train$store_and_fwd_flag <- NULL

# Limpeza das matrizes auxiliares
rm(try,try2,trytest,trytest2)

train$pickup_datetime <- NULL
test$pickup_datetime <- NULL

dow <- 0
test <- cbind(dow,test)

distance <- 0
train <- cbind(distance,train)
train2 <- cbind(distance,train2)
test <- cbind(distance,test)

train$X2 <- NULL
train$X3 <- NULL
test$X2 <- NULL
test$X3 <- NULL
train2 <- train

library(foreach)
library(geosphere)

day_of_year <- 0
train <- cbind(day_of_year,train2)
test <- cbind(day_of_year,test)
colnames(test)[4] <- "doy"

foreach(i=1:length(train$day_of_year), .packages = "geosphere", .verbose = TRUE) %do% {
  train$day_of_year[i] <- as.numeric(strftime(train$doy[i],format="%j"))
  train$dow[i] = as.numeric(train$doy[i]) %% 7
  train$distance[i] <- distHaversine(p1 = c(train$pickup_longitude[i],train$pickup_latitude[i]),
                p2 = c(train$dropoff_longitude[i],train$dropoff_latitude[i]))
}

foreach(i=1:length(test$day_of_year), .packages = "geosphere", .verbose = TRUE) %do% {
  test$day_of_year[i] <- as.numeric(strftime(test$doy[i],format="%j"))
  test$dow[i] = as.numeric(test$doy[i]) %% 7
  test$distance[i] <- distHaversine(p1 = c(test$pickup_longitude[i],test$pickup_latitude[i]),
                                     p2 = c(test$dropoff_longitude[i],test$dropoff_latitude[i]))
}

train$doy <- NULL
train$pickup_latitude <- NULL
train$pickup_longitude <- NULL
train$dropoff_longitude <- NULL
train$dropoff_latitude <- NULL

test$doy <- NULL
test$pickup_latitude <- NULL
test$pickup_longitude <- NULL
test$dropoff_longitude <- NULL
test$dropoff_latitude <- NULL

write.csv(test, file = "test_formated.csv")
write.csv(train, file = "train_formated.csv")