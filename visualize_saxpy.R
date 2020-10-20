library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(purrr)
library(rstudioapi)
library(ggthemes)

setwd(dirname(getActiveDocumentContext()$path))

data <- readLines("saxpy_elapsed_times.tsv") %>%
  str_split(pattern="\t") %>%
  map(~str_match(., "= (.+)")[,2]) %>%
  do.call(rbind, .) %>%
  as_tibble()

colnames(data) <- c("N", "TimeCPU", "TimeGPU")

