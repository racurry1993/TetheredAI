#install.packages("nflreadr")
#install.packages("nflreadr", repos = c("https://nflverse.r-universe.dev", getOption("repos")))
library(nflreadr)
setwd("C:/Users/rfo7799/Desktop/Git/TetheredAI/NFL/PROD")
schedule <- load_schedules(2025)
write.csv(schedule, '2025_schedule.csv', row.names = FALSE)


