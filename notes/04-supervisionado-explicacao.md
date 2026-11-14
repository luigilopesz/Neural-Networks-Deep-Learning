# Projeto 2 (Classificação): explicação para a equipe

Relatório: `docs/projects/classification/index.md`
Código: `docs/projects/classification/code/`
Enunciado salvo em `specs/insper-2026.2/projects/supervised.html`

> O enunciado exige a pasta `docs/projects/classification/` (não `supervised/`).
> Prazo oficial: **13/11/2026 (sexta), commits até 23:59**.

---

## 1. O resultado, em 5 linhas

1. Pergunta: "a estação vai alagar **amanhã**?" (classificação binária; 2,2 % de dias com
   enchente, desbalanceamento de 45 : 1).
2. A rede (MLP 21 → 64 → 32 → 1, PyTorch) tem **PR-AUC de teste 0,0900 ± 0,0012** (3 seeds).
   A regressão logística tem 0,0863.
3. A rede ganha de todos os baselines, e o ganho é **3× maior que a variação entre seeds**.
   Ou seja, é real.
4. Mas o ganho é pequeno: **+4,27 %**. O critério que fixamos antes de treinar pedia +10 %, então
   **o critério não foi cumprido**. O relatório diz isso abertamente.
5. O erro está concentrado no calendário: a rede **perde 100 % das enchentes de outubro**
   (138 de 138), porque aprendeu que enchente é coisa de julho a setembro.

---

## 2. Como rodar do zero

```bash
# na raiz do repositório
python3.12 -m venv env                      # ou: uv venv env --python 3.12
env/bin/pip install -r requirements.txt     # torch já está no requirements (VAE)
# dataset: precisa estar em docs/projects/data/ASIA_FLOOD_25YEAR_DATASET/data/
#   (baixe com kagglehub; a pasta está no .gitignore, NUNCA commitar)
cd docs/projects/classification/code
../../../../env/bin/python model.py          # autoteste (segundos)
../../../../env/bin/python select_on_val.py  # passos 0-4, ~1 minuto
../../../../env/bin/python test_once.py      # passos 5-6, ~30 segundos
```

Para baixar o dataset:

```python
import kagglehub
kagglehub.dataset_download("aliahmadmphil/asia-flood-25-year-flood-risk-atlas-2000-2024")
# copie a pasta ASIA_FLOOD_25YEAR_DATASET para docs/projects/data/
```

Os scripts são **determinísticos**. Rodar de novo dá exatamente os mesmos números; nós
conferimos que os JSONs saem idênticos byte a byte. Por isso eles:

- usam seed 42;
- rodam em 1 thread de CPU;
- usam `torch.use_deterministic_algorithms(True)`.

---

## 3. O que cada arquivo faz

### `common.py`: dados, métricas e figuras

- Importa o `data.py` e o `pipeline.py` do EDA **sem mudar nada**.
- `load_splits()` faz quatro coisas:
  1. carrega o CSV e cria o alvo do dia seguinte (`shift(-1)` por estação);
  2. separa treino (2000–2016), validação (2017–2020) e teste (2021–2024);
  3. ajusta (`fit`) o pipeline **só no treino**;
  4. transforma os três conjuntos em matrizes de 21 colunas.
- `scores()` calcula PR-AUC, ROC-AUC e PR-AUC só dentro da monção.
- `threshold_for_recall()` acha o **maior** limiar que ainda dá recall ≥ 80 %. Quanto maior o
  limiar, maior a precisão.
- `at_threshold()` calcula a matriz de confusão e precisão, recall e F1 das duas classes, mais
  o F1 macro.

### `model.py`: a rede e o loop de treino

- `MLP`: camadas `Linear → ReLU → Dropout`, terminando em **1 logit**, sem sigmoid. A
  inicialização é He para ReLU.
- `train()`, passo a passo:
  1. fixa a seed (inicialização, máscaras de dropout, ordem dos mini-batches);
  2. calcula `pos_weight = negativos / positivos` do treino, que dá ≈ 46,29;
  3. cria o otimizador AdamW e a loss `BCEWithLogitsLoss(pos_weight=46,29)`;
  4. a cada época:
     - percorre o treino em mini-batches de 1024 (zera o gradiente, faz forward, calcula a
       loss, faz backward e dá o step);
     - desliga o dropout e mede a loss de treino, a loss de validação e a **PR-AUC de
       validação**;
  5. se a PR-AUC de validação melhorou, guarda uma cópia dos pesos; se não melhorou por
     8 épocas seguidas, **para** (early stopping);
  6. no fim, **restaura os melhores pesos**.
- `python model.py` confere duas coisas: que 21-64-32-1 tem 3521 parâmetros, e que duas
  execuções com a mesma seed dão o mesmo resultado.

### `select_on_val.py`: passos 0 a 4 (só treino e validação; o teste nem é carregado)

1. **Passo 0:** imprime os tamanhos e as taxas de enchente. Imprime também a métrica, a regra
   do limiar e o critério de sucesso, **antes** de treinar qualquer coisa.
2. **Passo 1:** avalia os 3 baselines na validação.
3. **Passo 4 (busca):** treina 16 configurações, ordena pela PR-AUC de validação e escolhe a
   primeira. As 16 combinam:
   - 4 regularizações: nenhuma, weight decay, dropout, ambos;
   - × 2 arquiteturas: 21-32-1 e 21-64-32-1;
   - × 2 learning rates: 1e-3 e 3e-3.
4. **Passos 2 e 3:** treina de novo a configuração escolhida, guardando as probabilidades de
   cada época (para o GIF). Imprime a arquitetura, os parâmetros, a tabela de todas as
   épocas e o limiar de 80 % de recall.
5. Gera a Figura 3.1, a Figura 4.1 e a Animação 3.2.
6. Salva `results_select.json` (configuração escolhida e limiares). É o "congelamento".

### `test_once.py`: passos 5 e 6 (o único script que lê o teste)

1. Lê a configuração congelada do JSON. **Nada é escolhido aqui.**
2. Treina de novo a regressão logística e pega o limiar dela na validação.
3. Treina a MLP com seeds **42, 43 e 44**. Cada uma usa early stopping na validação e pega o
   limiar na validação; depois mede no teste.
4. Imprime a tabela do teste (todos os modelos), a média ± desvio das 3 seeds, o relatório
   por classe e se o critério foi cumprido.
5. Faz a análise de erros:
   - erro por bacia e por mês;
   - média das medições por tipo de resultado (TP, FN, FP, TN);
   - 5 piores falsos negativos e 5 piores falsos positivos;
   - importância por permutação, medida **na validação**.
6. Gera as Figuras 5.1–5.3 e 6.1–6.2 e salva `results_test.json`.

---

## 4. Por que cada decisão

| decisão | por quê (resposta curta) |
|---|---|
| **PR-AUC como métrica principal** | Só 2,2 % são enchentes. A acurácia engana: "nunca alaga" tem 97,8 %. A ROC-AUC também fica otimista quando há muitos negativos fáceis. A PR-AUC mede o que importa: quando dou alarme, quão frequentemente acerto, para cada nível de recall. A linha de base dela é a taxa de positivos (0,022). |
| **Limiar com 80 % de recall, escolhido na validação** | Perder uma enchente (ninguém é avisado) é muito pior que um alarme falso. Por isso fixamos primeiro quanto queremos pegar (80 %) e depois vemos quanta precisão isso custa. O limiar vem da validação para o teste continuar "limpo". |
| **`pos_weight` = 46,29 na loss** | É o jeito simples de lidar com o desbalanceamento: errar uma enchente pesa na loss como errar ~46 dias secos. Não precisa reamostrar nada. O efeito colateral é que as saídas ficam infladas (não são probabilidades calibradas; ver Figura 5.3). |
| **Split cronológico** | Dias vizinhos são quase iguais. Num split aleatório, o teste teria "cópias" do treino e o resultado seria inflado. Isso foi provado no EDA (Figura 1.5). |
| **Pipeline do EDA sem mudanças** | O enunciado pede para reaproveitar. O `fit` é só no treino, para não vazar estatísticas da validação e do teste. |
| **Colunas de vazamento fora** | `flood_risk_score > 7` reproduz o alvo em 100 % das linhas, e `severity_level` é o mesmo score em faixas. Seria colar a resposta. |
| **Baselines nessa ordem** | Classe majoritária: mostra a armadilha da acurácia. Gate de monção: é a regra grátis do EDA (só alaga na monção) e pega 100 % com 5 % de precisão. Logística: é a referência linear "de verdade". A rede só vale se ganhar dela. |
| **PyTorch e não o MLP NumPy do exercício** | A rubrica pede AdamW (ou momentum), dropout, weight decay e `BCEWithLogitsLoss`. O MLP do exercício só tem tanh e gradiente descendente simples; implementar tudo à mão seria mais código para explicar. |
| **ReLU + inicialização He** | A ReLU não satura, então o gradiente não some. A He mantém a variância estável em camadas ReLU. |
| **21 → 64 → 32 → 1** | Duas camadas conseguem aprender curvas, como "o risco sobe e desce ao longo do ano". Tem 3.521 parâmetros para 310.500 linhas (88 linhas por parâmetro), então não dá para decorar o treino. |
| **AdamW, lr 1e-3, batch 1024** | O Adam ajusta o passo por peso. O "W" é o weight decay desacoplado, que é exatamente a regularização que comparamos. 1e-3 é o padrão do Adam. Batch 1024 dá ~22 enchentes por batch e é rápido em CPU. |
| **Early stopping pela PR-AUC de validação, paciência 8** | Para no ponto em que a validação deixa de melhorar e devolve os melhores pesos. A PR-AUC oscila de uma época para outra, por isso a paciência de 8. |
| **3 seeds no teste** | Para saber se a diferença para a logística é maior que o "ruído" de treinar de novo. Foi: o gap é 3× o desvio. |
| **Critério de +10 % fixado antes** | Para não ajustar a meta depois de ver o resultado. Não cumprimos, e está escrito. |

---

## 5. O que cada figura mostra (uma frase cada)

| figura | o que mostra | conclusão |
|---|---|---|
| **3.1** curvas de treino | Loss de treino × validação por época; PR-AUC de validação por época | Converge sem overfitting forte; a PR-AUC estabiliza entre 0,0851 e 0,0893 a partir da época 11; melhor época = 18 |
| **3.2** animação (GIF) | A curva precisão-recall da validação época a época | O ganho da rede sobre a logística aparece no recall alto, onde um sistema de alerta opera |
| **4.1** regularização | PR-AUC e gap (val − treino) para nenhuma / weight decay / dropout / ambos | As diferenças (+0,0005 a +0,0010) são menores que o ruído entre seeds (0,0015): aqui a regularização quase não importa |
| **5.1** PR e ROC no teste | As curvas da MLP e dos 3 baselines | A MLP fica igual ou acima da logística e bem acima do gate de monção |
| **5.2** matrizes de confusão | MLP e logística no limiar de 80 % de recall | A MLP dá 1.445 alarmes falsos a menos, perdendo 22 enchentes a mais |
| **5.3** calibração | Probabilidade prevista × taxa real | As saídas são rankings, não probabilidades: a média prevista é 9,9× a taxa real, por causa do `pos_weight` |
| **6.1** erros por subgrupo | Taxa de perda e precisão por bacia e por mês | O erro é de calendário (outubro 100 %, junho 95 %), não de geografia (bacias entre 11 e 31 %) |
| **6.2** importância por permutação | Quanto a PR-AUC cai ao embaralhar cada feature | `day_of_year` é a mais importante; bacia é a menos |

---

## 6. Perguntas prováveis da prova oral

**Por que não acurácia?**
Porque 97,8 % dos dias não têm enchente. Um modelo que nunca dá alarme tem 97,8 % de acurácia e
não serve para nada.

**O que é PR-AUC e qual é o "zero" dela?**
É a área sob a curva precisão × recall. Um modelo aleatório tem PR-AUC igual à fração de
positivos: 0,022 na validação, 0,023 no teste. A nossa MLP tem 0,090, ou seja, ~4× o
aleatório.

**Por que a PR-AUC é tão baixa se a ROC-AUC é 0,87?**
A ROC-AUC conta os muitos negativos fáceis, como os dias fora da monção, que qualquer modelo
acerta. A PR-AUC só olha a qualidade dos alarmes. Com só as medições do dia, muitos dias secos
de monção parecem dias de enchente. As médias de TP e FP são quase iguais (Tabela de apoio,
§6).

**Como vocês escolheram o limiar?**
Na validação: o maior limiar que ainda pega 80 % das enchentes. Para a MLP (seed 42) deu
0,7731. Ele foi aplicado no teste sem mudar nada, e no teste o recall foi 78,5 %.

**O teste foi usado para escolher alguma coisa?**
Não. Só o `test_once.py` lê o teste, e ele lê a configuração e a regra do limiar de um JSON
gerado antes. Rodamos o script mais de uma vez para ajustar o texto e o layout das figuras,
mas os números saíram idênticos byte a byte.

**O que é `pos_weight` e qual o efeito colateral?**
É um peso extra na loss para os positivos (46,29 = negativos/positivos do treino). O modelo
passa a "prestar atenção" nas enchentes. O efeito colateral é que as saídas ficam
descalibradas: a média prevista é 0,232 contra uma taxa real de 0,023.

**Por que `BCEWithLogitsLoss` e não sigmoid + BCE?**
Ela junta a sigmoid e a BCE numa conta numericamente estável, sem `log(0)`. Por isso a rede
devolve o logit cru.

**Quantos parâmetros e por que isso importa?**
São 3.521: 21·64+64 + 64·32+32 + 32+1. Com 310.500 linhas de treino, dá 88 linhas por
parâmetro. A rede é pequena demais para decorar, e por isso quase não há overfitting.

**Onde começa o overfitting (Figura 3.1)?**
Não há um ponto claro. O gap entre validação e treino cresce devagar (de 0,023 para 0,046) e
a loss de validação fica estável depois da época 18. O limite vem das features, não da rede.

**Que regularizações vocês compararam e qual ganhou?**
Weight decay (1e-2, via AdamW) e dropout (0,2), contra nenhuma e contra as duas juntas. O
dropout ganhou (0,0893 contra 0,0883), mas a diferença é menor que o desvio entre seeds.
Dissemos isso no relatório: não é prova de que dropout ajuda aqui.

**Quantas configurações na busca?**
16 (4 regularizações × 2 arquiteturas × 2 learning rates), ordenadas pela PR-AUC de validação.

**A rede valeu a pena?**
Pouco. Ela ganha da logística por +4,27 % (critério: +10 %), mas o ganho é 3× o desvio entre
seeds, então é real. No ponto de operação, são 10,5 alarmes falsos por enchente pega contra
11,4 da logística.

**O que a rede aprendeu que a logística não aprendeu?**
Uma "lombada" no `day_of_year`: o risco sobe e desce ao longo do ano. A logística só consegue
"quanto maior, mais risco". Por isso `day_of_year` é a feature mais importante na permutação,
mesmo tendo AUC sozinha baixa no EDA (0,63).

**Isso bate com o EDA?**
Quase tudo bate: bacia é irrelevante, nível do rio é redundante com chuva, e chuva, umidade e
temperatura importam. A diferença é `day_of_year`: no EDA a AUC dela é baixa porque a relação
não é monótona; a rede consegue usar a curva.

**Qual o maior erro e qual a hipótese?**
Outubro: 100 % das 138 enchentes perdidas. No treino, só 4,4 % das enchentes estão em junho
ou outubro e 95,6 % em julho–setembro. A rede aprendeu "enchente = jul–set", e o limiar de
80 % de recall é atingido só com esses meses. As enchentes perdidas não são dias secos: têm
57,8 mm de chuva em média, contra 38,3 mm nas enchentes pegas. O calendário "venceu" a chuva.

**E a pior enchente perdida (score 0,000)?**
É a única enchente fora da monção em 25 anos (Indus, 22/05/2021): dia seco de primavera.
Nenhum modelo com essas features pega isso.

**Quem é afetado por um erro?**
Uma enchente perdida atinge quem mora rio abaixo (sem alerta). Um alarme falso custa dinheiro
e confiança, e o excesso de alarmes faz as pessoas ignorarem os avisos. No nosso caso, o
operador precisa saber que o modelo é quase cego em junho e outubro.

**O que fariam com mais tempo?**

1. Criar features de janela (chuva acumulada de 7 dias, rio subindo), que o plano previa e o
   pipeline do EDA não tem.
2. Usar um limiar por mês.
3. Calibrar as saídas (Platt).
4. Testar gradient boosting.
5. Rodar a busca com 3 seeds por configuração.

**Por que os números de validação (+15 %) foram melhores que os de teste (+4 %)?**
Escolhemos a melhor de 16 configurações na validação, então a validação fica otimista por
construção. É exatamente por isso que o teste fica separado até o fim.

**O dataset é real?**
Não, é sintético (o README diz que foi gerado). Os métodos são reais; os números descrevem um
simulador.

---

## 7. Números para saber de cor

| | valor |
|---|---|
| linhas treino / val / teste | 310.500 / 73.050 / 73.000 |
| taxa de enchente (treino / val / teste) | 2,11 % / 2,22 % / 2,33 % |
| `pos_weight` | 46,29 |
| parâmetros da rede | 3.521 (88,2 linhas por parâmetro) |
| configuração escolhida | 21-64-32-1, ReLU, dropout 0,2, sem weight decay, AdamW lr 1e-3, batch 1024 |
| melhor época / parou na | 18 / 26 |
| PR-AUC de validação: MLP / logística | 0,0893 / 0,0776 (+15,03 %) |
| PR-AUC de teste: MLP (3 seeds) / logística / gate / majoritária | 0,0900 ± 0,0012 / 0,0863 / 0,0556 / 0,0233 |
| ROC-AUC de teste: MLP / logística | 0,8733 ± 0,0008 / 0,8629 |
| limiar MLP (seed 42) / recall e precisão no teste | 0,7731 / 78,51 % e 8,69 % |
| critério | +4,27 % contra +10 % exigido, **não cumprido**; gap = 3,0× o desvio entre seeds |
| maior erro | outubro, 100 % das 138 enchentes perdidas; por bacia: Brahmaputra, 30,77 % |
| feature mais importante | `day_of_year` (queda de 0,0415 na PR-AUC, 46,4 %) |
