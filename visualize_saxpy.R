library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(purrr)
library(tidyr)
library(rstudioapi)
library(ggthemes)

setwd(dirname(getActiveDocumentContext()$path))

data <- readLines("saxpy_elapsed_times.tsv") %>%
  str_split(pattern="\t") %>%
  map(~str_match(., "= (.+)")[,2]) %>%
  do.call(rbind, .) %>%
  as_tibble()

colnames(data) <- c("N", "CPU", "GPU")

data <- data %>%
  pivot_longer(-N, names_to="Device", values_to="Time") %>%
  mutate(N=as.numeric(N), Time=as.numeric(Time))

data %>%
  mutate(LogN=log2(N)) %>%
  ggplot(aes(x=LogN, y=Time, color=Device)) +
  geom_line() +
  xlab(bquote(log[2]~ArraySize)) +
  ylab("Time [ms]") +
  theme_economist()

  