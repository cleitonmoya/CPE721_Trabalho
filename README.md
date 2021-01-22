# Predição do Desempenho de Estudantes utilizando Redes Neurais
Trabalho da disciplina CPE721 - Redes Neurais *Feedforward*

## Base de dados: 
- Student Performance
- https://archive.ics.uci.edu/ml/datasets/Student+Performance
- P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In **A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)** pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7. [Link](http://www3.dsi.uminho.pt/pcortez/student.pdf)

## Pré-processamento:
- [pre_processamento.pynb](pre_processamento.pynb) (Python / Jupyter *notebook*)

## Classificação binária
- Classes: 
	- **1**: Aprovado 
	- **-1**: Reprovado
- [nn_bin.m](nn_bin.m): versão básica (Matlab *script*)
- [nn_bin_kfold.m](nn_bin_kfold.m): com validação cruzada k-*fold* (Matlab *script*)