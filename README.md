# Classificador de Dígitos MNIST com PyTorch

Este projeto utiliza redes neurais convolucionais (CNN) para classificar dígitos escritos à mão usando o dataset **MNIST**.

O objetivo principal deste projeto é **estudar redes neurais convolucionais** e testar diferentes modelos treinados.

---

# Arquivos do Projeto

## 📓 nmist-nj.ipynb

Este é o **modelo que está funcionando melhor atualmente**.
Ele utiliza uma rede neural mais complexa e apresenta boa performance.

Este arquivo contém:

* Treinamento de uma rede neural convolucional (CNN)
* Teste do modelo
* Avaliação da performance
* Resultados com boa acurácia

Este é o modelo recomendado para uso e testes.

---

## 📓 nmist-pytorch.ipynb

Este foi o **primeiro modelo que eu desenvolvi**, com o objetivo principal de **aprender a usar o PyTorch**, sem foco em otimização ou desempenho.

Atualmente:

* A acurácia é menor que a do `nmist-nj.ipynb`
* Pode precisar de ajustes na arquitetura
* Pode precisar de mais épocas de treino
* Pode precisar de ajuste de parâmetros

Este arquivo foi mantido para aprendizado e comparação.

⚠️ **Não recomendado para uso principal.**

---

## 🎨 draw_predict_pygame.py

Este script permite **desenhar números com o mouse** e testar o modelo treinado.

### Funcionamento:

1. O usuário desenha um número na tela
2. O desenho é convertido para o formato do MNIST
3. O modelo tenta prever qual número foi desenhado
4. O resultado é exibido na tela

### Objetivo deste arquivo:

* Testar o modelo de forma interativa
* Ver como o modelo se comporta com números desenhados manualmente
* Demonstrar o uso prático do modelo treinado

---

# Objetivo do Projeto

Este projeto foi criado para:

* Aprender a usar **PyTorch**
* Estudar **redes neurais convolucionais (CNN)**
* Testar diferentes arquiteturas de rede
* Criar uma forma interativa de testar modelos treinados
