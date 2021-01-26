# Predição do Desempenho de Estudantes utilizando Redes Neurais
Trabalho da disciplina CPE721 - Redes Neurais *Feedforward*

## Base de dados: 
- Student Performance
- https://archive.ics.uci.edu/ml/datasets/Student+Performance
- P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In **A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008)** pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-9077381-39-7. [Link](http://www3.dsi.uminho.pt/pcortez/student.pdf)

## Pré-processamento
- [pre_processamento/pre_processamento.ipynb](pre_processamento/pre_processamento.pynb) (Python / Jupyter *notebook*)
- [pre_processamento/divide_dataset.m](pre_processamento/divide_dataset.m (Matlab *script*)

## Modelos
- Classificação binária:
	- [classificacao_binaria/nn_bin_grid_search.m](classificacao_binaria/nn_bin_grid_search.m) (Matlab *script*)
- Classificação em 5 classes:
	- [classificacao_multiclasse/nn_mul_grid_search.m](classificacao_binaria/nn_mul_grid_search.m) (Matlab *script*)
- Regressão:
	- [classificacao_multiclasse/nn_reg_grid_search.m](classificacao_binaria/nn_reg_grid_search.m) (Matlab *script*)
