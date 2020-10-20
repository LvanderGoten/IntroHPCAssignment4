library(dplyr)
library(readr)
library(ggplot2)
library(stringr)
library(purrr)
library(rstudioapi)
library(ggthemes)

setwd(dirname(getActiveDocumentContext()$path))

data.host_to_device <- readLines("bandwidthTest_results_host_to_device.tsv") %>%
  str_split(pattern="\n") %>%
  map(str_trim) %>%
  as.character() %>%
  map(~str_replace(., "(\\t)+", ",")) %>%
  str_c(collapse="\n") %>%
  read_csv()

data.device_to_host <- readLines("bandwidthTest_results_device_to_host.tsv") %>%
  str_split(pattern="\n") %>%
  map(str_trim) %>%
  as.character() %>%
  map(~str_replace(., "(\\t)+", ",")) %>%
  str_c(collapse="\n") %>%
  read_csv()

data.device_to_device <- readLines("bandwidthTest_results_device_to_device.tsv") %>%
  str_split(pattern="\n") %>%
  map(str_trim) %>%
  as.character() %>%
  map(~str_replace(., "(\\t)+", ",")) %>%
  str_c(collapse="\n") %>%
  read_csv()

data <- bind_rows(list(HostToDevice = data.host_to_device,
                       DeviceToHost = data.device_to_host,
                       DeviceToDevice = data.device_to_device), .id = "Mode") %>%
  rename("TransferSize"=`Transfer Size (Bytes)`, "Bandwidth"=`Bandwidth(GB/s)`)

data %>%
  ggplot(aes(x=TransferSize, y=Bandwidth)) +
  geom_line() +
  facet_wrap(~Mode) + 
  xlab("Transfer Size [bytes]") +
  ylab("Bandwidth [GB/s]") +
  theme_economist()
