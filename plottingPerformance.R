library(ggplot2)

timeSeries <- read.csv("/home/josura/Projects/LLM/privateGPT/time.csv",sep = ",",header = FALSE)

colnames(timeSeries) <- c("number_of_cores", "modelWeights", "modelEmbeddings", "time")

timeSeries$modelDetails <- paste(timeSeries$modelWeights, timeSeries$modelEmbeddings, sep = ":")


ggplot(timeSeries, aes(x = modelDetails, y = time, fill = modelDetails)) +
  geom_violin(scale = "width") +
  scale_fill_viridis_d() +
  labs(x = "Combined Feature (Model Weights : Model Embeddings)", y = "Time") + 
  geom_boxplot(width=0.1)+
  theme(axis.text.x = element_blank())

library(dplyr)
timeSeries.filtered <- timeSeries %>%
  select(time,modelDetails,modelWeights) %>%
  filter(modelWeights == "GPU/LlamaCppmodels/nous-hermes-13b.ggmlv3.q4_0.bin" | modelWeights == "LlamaCppmodels/nous-hermes-13b.ggmlv3.q4_0.bin")


ggplot(timeSeries.filtered, aes(x = modelDetails, y = time, fill = modelDetails)) +
  geom_violin(scale = "width") +
  scale_fill_viridis_d() +
  labs(x = "Combined Feature (Model Weights : Model Embeddings)", y = "Time") + 
  geom_boxplot(width=0.1)+
  theme(axis.text.x = element_blank())
