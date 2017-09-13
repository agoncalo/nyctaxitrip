# Abertura do arquivo
train = read.csv(file = "~/Desktop/train.csv", header = TRUE, sep = ",")

## Pré Processamento
train[,6:9] <- scale(train[6:9])
train <- train[3:11]
train$dropoff_datetime <- NULL

# Divisão data/hora
try <- data.frame(do.call('rbind', strsplit(as.character(train$pickup_datetime),'-',fixed = TRUE)))
try2 <- data.frame(do.call('rbind', strsplit(as.character(try$X3),' ',fixed = TRUE)))
hour <- data.frame(do.call('rbind', strsplit(as.character(try2$X2),':',fixed = TRUE)))

try <- cbind(try[1:2],try2[1])
try <- cbind(try, hour)
train <- cbind(try,train[2:8])

colnames(train)[1] <- "year"
colnames(train)[2] <- "month"
colnames(train)[3] <- "day"
colnames(train)[4] <- "hour"
colnames(train)[5] <- "minute"
colnames(train)[6] <- "second"

# Limpeza das matrizes auxiliares
rm(hour,try,try2)

train <- train[2:13]

## Minimos quadrados
lsfit(x = train[1:10], y = train$trip_duration)
