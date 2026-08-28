# 🏦 Dataset real: German Credit

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · **German Credit** · [Modelo](modelo.md) · [Métricas](metricas.md) · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · [Serviço](servico.md) · [API](api.md) · [Testes](testes.md)

---

## 🏦 Dataset real: German Credit

Todo o laboratório até aqui rodou sobre dados que **este projeto inventou**. Isso foi útil — dá para conferir se a rede aprendeu, porque a regra que gerou os rótulos está a três linhas de distância. O gerador sintético hoje [injeta ruído e desbalanceamento](dados-sinteticos.md#-geração-dos-dados-sintéticos) de propósito, e isso derruba a AUC de `0.9994` para `0.9770`; mas o ruído continua sendo *o ruído que nós escolhemos*, na quantidade que nós escolhemos.

O **German Credit** troca esse conforto por realidade: 1.000 solicitações de crédito de verdade, coletadas por Hans Hofmann na Universidade de Hamburgo e publicadas em 1994. Ninguém escolheu a regra que separa bom de mau pagador — e boa parte dela simplesmente **não está nas colunas**.

```bash
npm run fetch:german     # baixa da UCI e converte (o CSV já vem versionado)
npm start                # dataset real, codificação one-hot — o padrão
npm run start:ordinal    # mesmas colunas, codificação ordinal (comparação)
npm run start:synthetic  # o laboratório sintético continua a um argumento
```

| | Sintético | German Credit |
| --- | ---: | ---: |
| Clientes | 1.200 | 1.000 |
| Colunas usadas | 4 | 19 (de 20) |
| Entradas da rede | 4 | 57 (one-hot) |
| Alto risco | 15,8% | 30% |
| Origem dos rótulos | fórmula conhecida | comportamento real |
| Ruído | [injetado de propósito](dados-sinteticos.md#-geração-dos-dados-sintéticos) e mensurável | todo o que a realidade tem, em quantidade desconhecida |

### As 19 colunas usadas

Das 20 do arquivo original, o projeto usa 19 — **sete numéricas** e **doze qualitativas**:

| Numéricas | Faixa | | Qualitativas | Níveis |
| --------- | ----- | --- | ------------ | -----: |
| `durationMonths` | 4 – 72 meses | | `checkingStatus` | 4 |
| `creditAmount` | 250 – 18.424 DM | | `creditHistory` | 5 |
| `installmentRate` | 1 – 4 (% da renda) | | `purpose` | 10 |
| `residenceSince` | 1 – 4 anos | | `savingsStatus` | 5 |
| `age` | 19 – 75 anos | | `employmentYears` | 5 |
| `existingCredits` | 1 – 4 | | `otherDebtors` | 3 |
| `dependents` | 1 – 2 | | `property` | 4 |
| | | | `otherInstallments` | 3 |
| | | | `housing` | 3 |
| | | | `job` | 4 |
| | | | `telephone` | 2 |
| | | | `foreignWorker` | 2 |

As doze qualitativas somam **50 níveis**. Com one-hot, o vetor de entrada tem `7 + 50 = 57` posições.

> ⚖️ A vigésima coluna — **atributo 9, estado civil e sexo** — é lida e vai para o CSV, mas **não entra no modelo**. Ela tem outro papel: [auditar](#a-coluna-que-o-modelo-não-recebeu) as decisões depois que já foram tomadas.

### Por que one-hot e não ordinal

A versão anterior codificava as qualitativas como inteiro: `checkingStatus` virava `0, 1, 2, 3`. Isso é conveniente e, na maioria das colunas, é uma **mentira**.

Dizer que `purpose = 3` fica entre `2` e `4` afirma que "eletrodoméstico", "rádio/TV" e "reparos" estão em uma escala — e não estão. Não existe ordem entre finalidades de empréstimo. A rede recebia uma relação que ninguém quis afirmar.

One-hot desfaz a suposição. Cada categoria vira uma coluna própria:

```javascript
//   purpose = 3  ->  ordinal: [0.333]
//                    one-hot: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
const oneHotEncode = (size, index) =>
  Array.from({ length: size }, (unused, position) => (position === index ? 1 : 0));
```

Nenhuma categoria fica "maior" que outra, e a rede aprende **um peso independente para cada uma** em vez de um peso único multiplicado por um número arbitrário.

Duas colunas continuam sendo tratadas como número, de propósito: `installmentRate` (1 a 4, percentual da renda comprometida) e `residenceSince` (anos no endereço) têm ordem de verdade. A UCI as documenta como numéricas, e é o que são.

> 📐 Com todos os níveis presentes há colinearidade perfeita — as quatro colunas de `checkingStatus` sempre somam 1, então uma é dedutível das outras. É a *dummy variable trap*, e em regressão linear ela quebra a inversão da matriz. Em rede neural com viés e ativação não-linear isso não é um problema prático, e é o padrão do Keras — por isso o projeto mantém todos os níveis.

### O CSV guarda códigos, não features

O arquivo continua com uma coluna por atributo, e a qualitativa é gravada como o **índice** do código:

```text
durationMonths,creditAmount,...,checkingStatus,creditHistory,purpose,...,personalStatus,risk
6,1169,...,0,4,3,...,2,0
```

O `3` em `purpose` **não é uma quantidade** — é "A43", rádio/TV. O one-hot acontece na hora de montar o vetor, não na hora de gravar. É a mesma decisão de [guardar dados brutos](dados-sinteticos.md#o-csv-guarda-dados-brutos): se o arquivo já viesse com 57 colunas expandidas, trocar de codificação exigiria reexportar tudo.

### Agora a normalização precisa ser medida

Com dados inventados, as faixas eram conhecidas de antemão. Com dados reais, não:

```javascript
const fitMinMaxScaler = (customers, featureNames) => { /* mede mínimo e amplitude */ };

const applyMinMaxScaler = (scaler, customer) => /* aplica o que foi medido */;
```

E medir tem hora certa — **depois** de separar treino e teste:

```javascript
const { trainCustomers, testCustomers } = stratifiedSplitCustomers(customers);

const scaler = source.fitScaler(trainCustomers);           // só o treino
const toVector = (customer) => source.toVector(customer, scaler);
```

Se o `min`/`max` saísse do dataset inteiro, o maior empréstimo do conjunto de teste passaria a influenciar a escala aplicada no treino. Isso é **vazamento (*data leakage*)**, e o efeito é sempre o mesmo: o modelo parece melhor na avaliação do que será diante de dados que nunca viu.

É exatamente a dívida que o aviso lá de [Features de entrada](dados-sinteticos.md#-features-de-entrada) tinha registrado — e que só o dataset real obrigou a pagar.

### Uma fonte, três datasets

Trocar de dataset não exigiu tocar em treino, matriz de confusão, ROC ou escolha de limiar. Tudo que muda de um para o outro ficou em um objeto:

```javascript
const createGermanSource = ({
  id,
  label,
  encoding,
  regularization = { l2: L2_LAMBDA, dropout: DROPOUT_RATE },
}) => ({
  id,
  label,
  encoding,
  regularization,
  csvPath: GERMAN_CSV_PATH,
  featureNames: germanFeatureNames(encoding),

  ensure: (filePath = GERMAN_CSV_PATH) => { /* confere que o arquivo existe */ },
  read: () => readCustomersCsv(GERMAN_CSV_PATH),

  // Só as numéricas passam pelo min-max: escalar um código de
  // categoria seria escalar um rótulo.
  fitScaler: (customers) => fitMinMaxScaler(customers, GERMAN_NUMERIC),
  toVector: (customer, scaler) => toGermanVector(customer, scaler, encoding),

  audit: auditByGroup,
  sampleCustomer: { /* cliente de exemplo para a inferência */ },
});
```

São **três** fontes registradas — `synthetic`, `german` e `german-ordinal` —, e as duas do German Credit saem da mesma fábrica, diferindo só no `encoding`.

A fonte sintética cumpre o mesmo contrato, com duas diferenças que valem ler:

```javascript
  // A escala é CONHECIDA porque nós geramos os dados: "ajustar" aqui é
  // devolver as constantes, e o argumento é ignorado de propósito.
  fitScaler: () => null,
  toVector: (customer) => toFeatureVector(customer),

  // 225 parâmetros para 768 linhas: a capacidade já cabe no dado, e
  // frear aqui só cobra. A medição está em Regularização.
  regularization: { l2: 0, dropout: 0 },
```

A `regularization` entrou no contrato pelo mesmo motivo que a `fitScaler`: a intensidade certa depende da razão entre parâmetros e linhas, que é [propriedade do dataset](modelo.md#cada-dataset-pede-uma-dose-diferente), não do laboratório.

E a rede continua precisando saber pouquíssimo sobre qual dataset está em uso — o tamanho da entrada e a dose dos freios:

```javascript
const { l2, dropout } = { ...source.regularization, ...overrides };
const model = buildModel(source.featureNames.length, { l2, dropout });
```

### O resultado — e por que ele é a melhor parte

```text
Test accuracy: 0.7150
Baseline (classe majoritária): 0.7000
AUC: 0.7477
```

**A acurácia caiu de `0.94` no sintético para `0.72` aqui.** E o número logo abaixo dela é o que importa: `0.7000` é o que se consegue **chutando "baixo risco" para todo mundo**, sem olhar feature nenhuma. O treino inteiro comprou 1,5 ponto percentual nesta divisão — e a [validação cruzada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado) mostra que esta divisão é das piores: na média de k dobras o ganho é de quase cinco pontos.

Pior: no limiar `0.5`, o modelo pega **24 dos 60** maus pagadores do conjunto de teste — recall de `0.4000`.

```text
Matriz de confusão (limiar 0.5):
           | Predito BAIXO | Predito ALTO
-----------+---------------+-------------
Real BAIXO |      119 (TN) |      21 (FP)
Real ALTO  |       36 (FN) |      24 (TP)
```

Se a acurácia fosse a única métrica do projeto, esse modelo passaria por bom. **É por isso que todas as outras existem.**

E a AUC de `0.7477` diz que não é o modelo que está inútil — ele ordena os clientes bem acima do acaso. O que está errado é o corte:

```text
Ajuste do limiar (FP custa 1, FN custa 5):
Estratégia     | Limiar |    FPR |    TPR | FP | FN | Custo
---------------+--------+--------+--------+----+----+------
Padrão (0.5)   | 0.5000 | 0.1500 | 0.4000 | 21 | 36 |   201
Youden (max J) | 0.2591 | 0.4071 | 0.8500 | 57 |  9 |   102
Menor custo    | 0.2591 | 0.4071 | 0.8500 | 57 |  9 |   102
```

Descer o corte de `0.5` para `0.2591` derruba os falsos negativos de **36 para 9** e o custo de **201 para 102** — 49% mais barato, **sem retreinar nada**. O modelo sempre soube ordenar; faltava alguém escolher onde cortar. (Nesta execução as duas estratégias caíram no mesmo ponto da curva; nem sempre caem, e a [seção do limiar](metricas.md#-ajuste-do-limiar-de-decisão) mostra quando divergem.)

> 💡 Os custos `1` e `5` não são invenção deste projeto: são a **matriz de custo oficial do dataset**, publicada junto com ele. A UCI documenta que classificar um mau pagador como bom custa 5 vezes mais que o contrário. A `chooseThresholdByCost`, escrita antes de o dataset real entrar no projeto, já estava calibrada para ele.

### One-hot melhorou o modelo? Não.

A justificativa acima é de **correção**, não de desempenho. Vale medir a diferença em vez de supor — 15 sementes, mesma arquitetura, mudando só a codificação e o conjunto de colunas:

| Variante | Entradas | Parâmetros | AUC | Custo mínimo |
| -------- | -------: | ---------: | --: | -----------: |
| ordinal, 8 colunas (versão anterior, medição histórica) | 8 | 289 | `0.7793` ± 0.0057 | `101.4` ± 2.2 |
| ordinal, 19 colunas, sem freio | 19 | 465 | `0.7726` ± 0.0087 | `102.8` ± 2.4 |
| ordinal, 19 colunas (o que `start:ordinal` roda) | 19 | 465 | `0.7692` ± 0.0079 | `102.8` ± 2.4 |
| one-hot, 19 colunas, sem freio | 57 | 1.073 | `0.7762` ± 0.0071 | `100.6` ± 3.0 |
| **one-hot, 19 colunas** (o que `npm start` roda) | **57** | **1.073** | `0.7734` ± 0.0071 | `100.9` ± 2.6 |

*(média ± erro padrão sobre 15 sementes de embaralhamento — variar a divisão, e não os pesos, é o protocolo certo aqui, pelo motivo que a seção [Uma divisão não é o dataset](#uma-divisão-não-é-o-dataset) mede. A primeira linha vem de uma medição anterior, com outro conjunto de sementes; as quatro últimas são do mesmo conjunto, comparáveis entre si.)*

**Nenhuma diferença sobrevive ao erro padrão.** Nem a codificação correta, nem 11 colunas a mais, nem [regularização](modelo.md#-regularização-l2-e-dropout) moveram a AUC de forma distinguível de ruído — as quatro variantes comparáveis cabem em `0.007`, contra um erro padrão de `0.008`.

Três leituras disso, todas úteis:

1. **A codificação não era o gargalo.** A AUC de ~`0.78` é aproximadamente o teto publicado para o German Credit — a literatura reporta `0.76`–`0.80` para praticamente qualquer método, de regressão logística a *gradient boosting*. O limite está no sinal disponível nos dados, não em como as colunas são representadas.

2. **Mais features com o mesmo dado não é ganho automático.** O modelo saltou de 289 para 1.073 parâmetros treinando com as mesmas 640 linhas efetivas, e a capacidade extra foi para decorar: a diferença treino−teste é `0.0707` no one-hot contra `0.0224` no ordinal. A [regularização](modelo.md#-regularização-l2-e-dropout) corta essa diferença pela metade — e, como a tabela acima mostra, **sem que a AUC note**. Decorar era real e não estava custando nada de mensurável.

3. **Correção e desempenho são eixos separados.** One-hot continua sendo a representação certa para `purpose`, mesmo sem mexer no número. Afirmar uma ordem que não existe é errado independentemente de a métrica notar.

Um detalhe que o protocolo torna visível: **na divisão fixa do projeto, a variante ordinal vai melhor** — AUC `0.7496` ± 0.0013 contra `0.7387` ± 0.0013 do one-hot, uma diferença de várias vezes o erro padrão. Isso não contradiz a tabela acima; confirma o que a seção [Uma divisão não é o dataset](#uma-divisão-não-é-o-dataset) mede. Uma diferença que some ao trocar a divisão é uma propriedade **daquele sorteio**, não das codificações, e tratá-la como resultado seria exatamente o erro que a média de 15 sementes existe para evitar.

> 🔬 Para reproduzir a comparação: `npm start` roda one-hot e `npm run start:ordinal` roda a codificação anterior sobre as mesmas 19 colunas. A variante ordinal existe no código **só para isso** — para que a frase "one-hot é melhor" possa ser medida em vez de repetida.

### A coluna que o modelo não recebeu

O atributo 9 é estado civil **e sexo**: `A91` homem divorciado, `A92` mulher, `A93` homem solteiro, `A94` homem casado/viúvo. Dá para recuperar o sexo dele — `A92` é o único código feminino que aparece nos 1.000 registros.

Ele nunca entrou no modelo, e continua fora. Mas tirar a coluna resolve o problema?

```text
Auditoria por sexo — limiar único (o modelo nunca recebeu esta coluna):
Grupo    |   N | Inadimp. real | Marcados ALTO | FN não pegos
---------+-----+---------------+---------------+-------------
Mulheres |  66 |         30.3% |         57.6% |        10.0%
Homens   | 134 |         29.9% |         52.2% |        17.5%

Razão de aprovação (regra dos 4/5): 0.888
```

**Não resolve.** As taxas-base dos dois grupos nesta divisão são praticamente iguais — `30,3%` contra `29,9%` —, e mesmo assim o modelo marca mulheres como alto risco com mais frequência, sem nunca ter visto a coluna. Ele reconstrói o sinal por tabela: idade, moradia, tempo de emprego e valor do crédito carregam a informação, e a rede a recompõe sozinha. Dá para medir **quanto**: uma rede igual, treinada para prever o sexo a partir das mesmas features, chega a [AUC `0.699`](mitigacao.md#por-que-a-disparidade-existe).

Nesta execução a razão fica em `0.888`, acima do patamar dos quatro quintos. Em outras, não fica — e é por isso que a subseção seguinte existe.

É o resultado clássico de que *fairness through unawareness* não funciona — remover o atributo protegido não remove a disparidade, só a torna mais difícil de enxergar. Por isso a coluna fica no CSV: para medir depois, não para decidir antes.

#### Um número só não basta

O `N` das mulheres no hold-out é 66, e os pesos iniciais são aleatórios — essa razão balança bastante entre execuções. Média de 15 divisões:

| | Mulheres | Homens |
| --- | ---: | ---: |
| Inadimplência **real** | 35,2% ± 1,3 | 27,7% ± 0,5 |
| Marcados ALTO pelo modelo | 68,0% ± 3,0 | 65,5% ± 2,5 |

**Razão de aprovação: `0.9219` ± 0.0441** — abaixo de `0.80` em 2 das 15 execuções, e variando de `0.71` a `1.33` entre elas.

E aqui é preciso ser honesto sobre o que o número diz e o que não diz. As mulheres do dataset **de fato** têm taxa de inadimplência maior (35,2% contra 27,7%). A razão entre as taxas-base é `1.27`; a razão entre as taxas de marcação é `1.04`. Ou seja: **o modelo é bem menos desigual que os próprios dados** — ele atenua a diferença, não a amplifica.

> 🔬 A [regularização](modelo.md#-regularização-l2-e-dropout) não mexeu nisso: nas mesmas 15 divisões, a razão é `0.8995` ± 0.0460 sem os freios e `0.9219` ± 0.0441 com eles — indistinguíveis. Disparidade não é *overfitting*, e não sai pelo mesmo remédio.

Então há discriminação? Depende do critério, e é isso que torna o caso interessante. Auditando os **1.000** clientes de uma vez, com validação cruzada de 5 dobras e 10 repetições — o único recorte grande o bastante para separar efeito de ruído:

- **Paridade demográfica** (mesma taxa de aprovação nos dois grupos) → **falha**: razão `0.8408` ± 0.0122, abaixo de `0.80` em 2 das 10 repetições.
- **Igualdade de erros** (mesmo recall nos dois grupos) → **passa**: o inadimplente escapa em `7,7%` ± 0,6 dos casos entre as mulheres e `9,5%` ± 0,6 entre os homens.

Os dois critérios são incompatíveis entre si quando as taxas-base diferem — é um resultado provado, não uma limitação deste projeto. Escolher qual vale é uma decisão de política, não de engenharia.

> ⚖️ Medir e parar aqui deixaria o problema documentado e intacto. A seção [Mitigação da disparidade](mitigacao.md#-mitigação-da-disparidade) age sobre o número — leva a razão de `0.8408` a `1.0184` sem retreinar nada — e mede o que isso cobra.

O que o laboratório entrega é a **medição**. Quem decide o que fazer com ela precisa de mais contexto do que um `README` tem: por que as taxas-base diferem (o dataset é de 1994, quando crédito para mulheres casadas dependia de autorização do marido na Alemanha), se a diferença é causal ou reflexo de discriminação histórica já embutida nos rótulos, e o que a lei aplicável exige.

> ⚠️ Um modelo treinado em rótulos históricos aprende as decisões do passado, inclusive as injustas. Se em 1994 mulheres recebiam menos crédito e por isso apareciam mais como inadimplentes, o rótulo já vem contaminado — e nenhuma escolha de features conserta um rótulo enviesado.

### Comparação lado a lado

Média de **25 execuções sobre a divisão que o projeto fixa** — a mesma que `npm start` usa, com a mesma configuração de treino, então os baselines abaixo são exatamente os que aparecem no seu terminal. Só a inicialização dos pesos varia:

| Métrica | Sintético | German Credit |
| ------- | --------: | ------------: |
| Entradas da rede | 4 | 57 |
| Baseline (classe majoritária) | `0.8423` | `0.7000` |
| Test accuracy | `0.9416` ± 0.0067 | `0.7088` ± 0.0026 |
| **Ganho sobre o baseline** | **+9,9 pts** | **+0,9 pt** |
| AUC | `0.9733` ± 0.0007 | `0.7387` ± 0.0013 |
| Recall no limiar `0.5` | `0.6295` ± 0.0427 | `0.4053` ± 0.0073 |
| Custo no limiar `0.5` | `70.4` ± 8.1 | `201.0` ± 1.9 |
| Custo no limiar escolhido | `20.1` ± 0.9 | `105.4` ± 0.7 |

Duas linhas resumem a diferença entre um laboratório e um problema real.

O **ganho sobre o baseline quase desaparece**: `0.7088` contra `0.7000` de quem chuta "bom pagador" para todos os 200 clientes do teste, sem olhar coluna nenhuma. Menos de um ponto — e o **recall no limiar `0.5`** cai de `0.63` para `0.41`: o corte herdado deixa passar seis em cada dez maus pagadores.

Um relatório que parasse na acurácia concluiria que o modelo é inútil — e estaria errado por dois motivos independentes.

O primeiro é a **AUC de `0.7387`**: ele *ordena* os clientes bem acima do acaso, e o que falta não é sinal, é **régua**. No limiar escolhido, o mesmo modelo, sem retreinar, derruba o custo de `201.0` para `105.4`.

O segundo é que **esta divisão é ruim**. A [validação cruzada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado) mede `0.7499` ± 0.0018 contra o mesmo baseline `0.7000` — quase cinco pontos de ganho, não um. Os dois números estão certos e respondem perguntas diferentes; o honesto é o segundo, e ele só existe porque a pergunta foi feita a k divisões em vez de uma.

Este é o resultado mais útil do projeto inteiro, e ele precisou da [matriz de confusão](metricas.md#-matriz-de-confusão), da [ROC](metricas.md#-curva-roc-e-auc), do [ajuste do limiar](metricas.md#-ajuste-do-limiar-de-decisão) e da [validação cruzada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado) para ser dizível: **acurácia quase empatada com o baseline, AUC de `0.74`, e um ganho real de cinco pontos que uma única divisão escondeu**. Cada número diz uma coisa diferente sobre o mesmo modelo, e nenhuma das conclusões seria visível sem as outras métricas ao lado.

O dataset sintético não estava errado — ele estava **fácil**, e continua sendo o mais fácil dos dois mesmo depois do [ruído injetado](dados-sinteticos.md#-geração-dos-dados-sintéticos). A diferença é que agora dá para dizer *quanto* mais fácil, e por quê: no sintético o ruído é conhecido e limitado a 2 pontos irredutíveis; no German Credit ninguém sabe onde fica o teto.

### Uma divisão não é o dataset

Os números acima descrevem **a divisão que o projeto fixa** (semente `42`). Vale saber o quanto eles dependem dela. Sorteando 15 divisões diferentes, todas [estratificadas](validacao-cruzada.md#-validação-cruzada-e-split-estratificado):

| | Sintético | German Credit |
| --- | ---: | ---: |
| Baseline | `0.8423` **em todas** | `0.7000` **em todas** |
| Acurácia média | `0.9477` ± 0.0043 | `0.7467` ± 0.0058 |
| Pior / melhor divisão | `0.9212` / `0.9710` | `0.7150` / `0.7900` |
| AUC média | `0.9716` ± 0.0036 | `0.7781` ± 0.0083 |
| Pior / melhor divisão | `0.9382` / `0.9903` | `0.7210` / `0.8318` |

Três coisas ficam claras de uma vez:

**A divisão importa mais que a inicialização.** A AUC do German Credit vai de `0.72` a `0.83` conforme o sorteio — onze pontos de amplitude, contra `± 0.0013` entre inicializações de peso na mesma divisão. O acaso do corte domina o acaso do treino por uma ordem de grandeza.

**O baseline parou de se mexer, e isso foi conquistado.** Ele sai `0.7000` nas quinze divisões, não em média: em cada uma. Com o corte cru que o projeto usava antes, o mesmo experimento entregava de `0.6500` a `0.7600`, porque a proporção de maus pagadores no teste mudava a cada sorteio — e comparar acurácias contra uma régua que oscila onze pontos não significa nada. Foi o [split estratificado](validacao-cruzada.md#-validação-cruzada-e-split-estratificado) que travou o número.

**A semente `42` calhou de ser ruim** — e não por pouco. Na média das divisões o modelo real acerta `0.7467`; na divisão fixa, `0.7088`, abaixo da **pior** das quinze. **Os dois números estão certos**; eles respondem perguntas diferentes: "o que esperar de uma divisão qualquer?" e "o que esta divisão dá?".

Isso é um resultado sobre o **tamanho do dataset**, não sobre o modelo: 200 linhas de teste, das quais 60 são positivas, não sustentam três casas decimais. É o argumento concreto para a [validação cruzada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado), que troca "a estimativa de um sorteio" pela "média de vários" — e que, medida 10 vezes, concorda com a média das 15 divisões dentro do erro padrão: `0.7499` ± 0.0018 contra `0.7467` ± 0.0058.

E é por isso que a semente continua fixa. Sabendo que existe uma divisão que dá `0.7900`, a tentação de procurá-la é real; uma semente congelada no código é a defesa mais barata contra escolher o resultado depois de ver os resultados.

### Reprodutibilidade

O arquivo real chega na ordem em que foi coletado, e essa ordem não é aleatória. O pipeline embaralha antes de dividir — com **semente fixa**:

```javascript
const SHUFFLE_SEED = 42;

const customers = shuffle(await source.read(), createRandom(SHUFFLE_SEED));
```

`Math.random()` não aceita semente, então o projeto traz um gerador próprio (*mulberry32*, 32 bits). Sem ele, cada execução mediria um recorte diferente do dataset e nenhum número desta seção se reproduziria.

Os pesos iniciais da rede continuam aleatórios, então as métricas ainda oscilam um pouco entre execuções — mas os **dados** não.

### Procedência

| | |
| --- | --- |
| **Fonte** | [UCI ML Repository — Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| **Autor** | Prof. Dr. Hans Hofmann, Universität Hamburg (1994) |
| **Licença** | CC BY 4.0 |
| **Arquivo versionado** | `data/german-credit.csv` — 1.000 linhas, 9 colunas, ~23 KB |
| **Como regerar** | `npm run fetch:german` |

O CSV convertido é versionado para que `npm start` funcione **offline** logo depois do clone. O `scripts/fetch-german.js` existe para auditar a origem e regerar o arquivo: ele baixa da UCI e aplica `parseGermanCsv`, a mesma função coberta pelos testes.

---

[⬅️ Voltar ao README](../README.md)
