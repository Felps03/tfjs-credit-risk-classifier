# 🧠 TFJS Credit Risk Classifier

Laboratório didático de classificação de risco de crédito com **Node.js**, **TensorFlow.js** e uma rede neural **MLP (Multilayer Perceptron)**.

Todo o ciclo de um projeto de Machine Learning supervisionado está aqui, do dado bruto até a inferência — sobre um **dataset real de crédito** ou sobre um dataset sintético gerado aqui. O código vive em `src/`, um módulo por seção, **numerados na ordem em que se leem**: `00-constants` → `18-main`. O `index.js` é só a porta de entrada.

```mermaid
flowchart LR
    A(["🎯 Problema"]) --> B["🏦 Dados<br/>German Credit (UCI)<br/>ou sintéticos"]
    B --> B2["📄 CSV<br/>dados brutos em disco"]
    B2 --> S["✂️ Split<br/>treino / teste"]
    S --> C["⚙️ Pré-processamento<br/>escala medida no treino"]
    C --> D["🏋️ Treinamento<br/>MLP 57 → 16 → 8 → 1"]
    D --> E{"📊 Validação<br/>val_loss melhorou?"}
    E -->|"sim — próxima época"| D
    E -->|"não há 5 épocas — early stopping"| F["🧪 Teste<br/>matriz, F1, AUC<br/>e ajuste do limiar"]
    F --> G["💾 Persistência<br/>model.save / loadLayersModel"]
    G --> H(["🔮 Predição"])

    classDef start fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef step  fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,color:#1e293b
    classDef check fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef final fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class A start
    class B,B2,S,C,D step
    class E check
    class F,G step
    class H final
```

O laço entre **Treinamento** e **Validação** é o coração do processo: a cada época o modelo é medido em dados que não usou para ajustar pesos, e o *early stopping* corta o ciclo quando essa medida para de melhorar. **Teste**, **Persistência** e **Predição** acontecem uma única vez, depois que o treino terminou.

> ⚠️ Finalidade **exclusivamente educacional**. O dataset real é de 1994 e serve de estudo, não de base para decisões financeiras.

**Neste README:** [Documentação](#-documentação) · [Objetivo](#-objetivo) · [Início rápido](#-início-rápido) ([estrutura](#estrutura), [testes](#testes)) · [A API](#-a-api) · [Limitações conhecidas](#-limitações-conhecidas) · [Próximas evoluções](#-próximas-evoluções) · [Conceitos](#-conceitos-demonstrados) · [Aviso](#-aviso) · [Referências](#-referências)

---

## 📚 Documentação

O `README` cobre o essencial: o que o projeto é, como rodar e o que ainda falta. O resto — a medição de cada decisão — está em `docs/`:

| Documento | O que tem lá |
| --------- | ------------ |
| **[🧪 Dados: features, geração sintética e CSV](docs/dados-sinteticos.md)** | Como as features são definidas, como o dataset sintético é gerado (ruído, desbalanceamento, semente) e como os dados vão e voltam do CSV. |
| **[🏦 Dataset real: German Credit](docs/german-credit.md)** | O dataset da UCI, a codificação das colunas qualitativas, a auditoria por sexo e a comparação lado a lado com o sintético. |
| **[🧠 O modelo: arquitetura, regularização e treinamento](docs/modelo.md)** | A MLP, a comparação de oito topologias que termina em empate, os dois freios contra overfitting medidos em grade, a configuração de treino e como os dados são divididos. |
| **[📊 Métricas de avaliação e ajuste do limiar](docs/metricas.md)** | Matriz de confusão, precision/recall/F1, curva ROC e AUC, e a escolha do corte por custo. |
| **[🔁 Validação cruzada e split estratificado](docs/validacao-cruzada.md)** | Por que um hold-out não basta, o que a estratificação trava e o que k dobras mudaram na conclusão do projeto. |
| **[⚖️ Mitigação da disparidade](docs/mitigacao.md)** | O limiar por grupo: o que ele corrige, o que ele cobra e por que o padrão é desligado. |
| **[🔮 Inferência, persistência e memória](docs/inferencia.md)** | Prever para um cliente novo, salvar e recarregar o modelo, e o gerenciamento de tensores. |
| **[🌐 O serviço: API REST e o pacote servido](docs/servico.md)** | O endpoint `POST /risk-score`, o `metadata.json` que impede o *training-serving skew*, a validação do payload e o campo que a API recusa. |
| **[🧩 API do módulo e exemplo de saída](docs/api.md)** | Tudo que o `index.js` exporta, e uma execução completa comentada. |
| **[✅ Testes e solução de problemas](docs/testes.md)** | A tabela de cobertura da suíte, alvo por alvo, e o que fazer quando o `tfjs-node` não instala. |

---

## 🎯 Objetivo

Treinar uma rede neural capaz de estimar a probabilidade de um cliente ser de **alto risco** de inadimplência.

A rede recebe as características financeiras do cliente — **57** no dataset real, **4** no sintético — e devolve **um único número entre `0` e `1`**:

| Saída da rede | Interpretação          | Classificação no corte padrão |
| ------------: | ---------------------- | ----------------------------- |
|        `0.12` | 12% de chance de risco | ✅ BAIXO RISCO |
|        `0.91` | 91% de chance de risco | ⚠️ ALTO RISCO  |

O corte **padrão** é `0.5`, e ele é o ponto de partida — não o que o projeto entrega:

```javascript
const DECISION_THRESHOLD = 0.5;

const classify = (probability) =>
  (probability >= DECISION_THRESHOLD ? 'ALTO RISCO' : 'BAIXO RISCO');
```

Meio a meio é o corte de quem não sabe o preço de errar. **Este projeto sabe**: no German Credit, deixar passar um inadimplente custa **5×** o que custa recusar um bom pagador, e é a matriz de custo oficial do dataset que diz isso. Com essa assimetria o corte desce — a rede passa a sinalizar mais, trocando falsos negativos caros por falsos positivos baratos.

E ele desce **onde pode**: numa fatia de calibração separada do treino, que o teste nunca vê. Escolher o corte no mesmo conjunto em que ele é medido publica um número que não se sustenta, e este projeto fazia exatamente isso até pouco tempo atrás. A diferença aparece impressa em toda execução:

```text
Custo por cliente: 0.406 na calibração (onde o corte foi escolhido) · 0.520 no teste (onde ele vale).
```

Por isso o limiar é [medido e gravado no pacote](docs/metricas.md#-ajuste-do-limiar-de-decisão) a cada treino, viaja junto dos pesos e é devolvido em toda resposta da API. Ele é uma **escolha de negócio**, não uma verdade do modelo — e uma escolha que muda de valor quando os dados mudam é uma escolha que não pode ficar chumbada no código.

---

## 🚀 Início rápido

**Requisitos:** Node.js **20 ou 22** (LTS). O projeto traz um `.nvmrc`, então basta:

```bash
nvm use
```

O `@tensorflow/tfjs-node` baixa binários nativos na instalação — o primeiro `npm install` demora.

> ⛔ **Não use Node 23 ou superior.** Veja [Solução de problemas](docs/testes.md#-solução-de-problemas).

```bash
git clone https://github.com/Felps03/tfjs-credit-risk-classifier.git
cd tfjs-credit-risk-classifier
npm install
npm start
```

`npm start` roda sobre o **[German Credit](docs/german-credit.md#-dataset-real-german-credit)**, um dataset real da UCI. Os dois CSVs vêm versionados, então isso funciona **offline** logo após o clone.

```bash
npm start                # dataset real (German Credit) — padrão
npm run start:synthetic  # dataset sintético gerado pelo projeto
npm run fetch:german     # rebaixa o dataset real da UCI e reconverte
npm run seed             # regenera o dataset sintético a partir do código
```

Uma execução mede **uma** divisão. Para a estimativa com barra de erro — k treinos, cada cliente pontuado por um modelo que não o viu — existe um comando à parte:

```bash
npm run cv               # 5 dobras estratificadas no dataset real
npm run cv:synthetic     # o mesmo no sintético
node index.js --cv=10    # escolhendo o k
```

É [essa medição](docs/validacao-cruzada.md#-validação-cruzada-e-split-estratificado) que responde "quanto o modelo acerta?" sem depender de qual sorteio calhou.

Os dois freios contra *overfitting* são ajustáveis pela linha de comando, para que o efeito possa ser **visto** em vez de lido — e a arquitetura também:

```bash
node index.js --l2=0 --dropout=0   # a rede sem regularização nenhuma
node index.js --dropout=0.5        # só dropout, e forte
node index.js --units=64,32        # outra topologia
node index.js --units=0            # nenhuma camada oculta: regressão logística
```

Escolher uma delas por convenção seria estranho num projeto que mede o resto, então há um comando que compara oito topologias por validação cruzada:

```bash
npm run arquiteturas               # da regressão logística à rede de 15.745 parâmetros
```

O [resultado](docs/modelo.md#-comparando-arquiteturas) é a parte interessante: as oito **empatam**. Uma regressão logística de 58 parâmetros, sem camada oculta nenhuma, entrega a mesma acurácia que a rede 271× maior — porque o gargalo são as 1.000 linhas, não a capacidade.

O desbalanceamento — 30% de inadimplentes — também é atacável durante o treino, e não só depois dele pelo limiar:

```bash
node index.js --balancear        # peso por classe no `fit`
npm run cv -- --balancear        # o efeito medido em 5 dobras, que é o que vale
```

O [resultado é negativo e vale a leitura](docs/modelo.md#-peso-por-classe-o-desbalanceamento-atacado-durante-o-treino): a acurácia **cai** de forma medível e o custo não melhora. O padrão é desligado por isso.

E a disparidade medida pela auditoria pode ser **corrigida** em vez de só reportada, com um limiar por grupo calibrado no treino:

```bash
npm start -- --mitigar             # a decisão passa a ler o sexo do cliente
```

A comparação entre as duas políticas aparece em toda execução, com ou sem a flag; o que a flag muda é qual delas decide. O padrão é **desligado**, e [a seção sobre mitigação explica por quê](docs/mitigacao.md#-mitigação-da-disparidade) — com números.

Depois de treinar, o modelo pode ser **servido**. O `npm start` salva um pacote — pesos, escala medida no treino e limiar escolhido — e a API sobe a partir dele:

```bash
npm run serve                      # http://localhost:3000
npm run serve -- --port=8080
```

O mesmo endereço abre uma **página** com três modos sobre a mesma rede:

![A página do fluxo: passar o mouse por um campo acende o caminho dele pela rede, editar um valor repontua o cliente na hora, e os três modos mostram a mesma rede decidindo, aprendendo e sendo medida.](docs/fluxo-da-analise.gif)

- **Análise** — o caminho do dado, dos campos que entraram até a probabilidade que saiu. Passar o mouse por um campo **acende o caminho dele**: uma coluna numérica vai para a escala min–max, uma qualitativa vai para o one-hot, e as duas só se encontram na primeira camada oculta. Os 19 campos são **editáveis**: baixar o prazo de 48 para 6 meses repontua o cliente e move as barras na hora.
- **Treinamento** — o laço que produziu os pesos, passo a passo — passo à frente, erro medido, passo atrás, pesos ajustados —, ao lado da **curva real** daquele treino: a perda de treino caindo, a de validação achatando, e o ponto onde o *early stopping* cortou.
- **Avaliação** — quanto ele acerta **ao lado do piso da classe majoritária**, a matriz de confusão no limiar escolhido, os três cortes candidatos com o preço de cada um, e a auditoria por sexo com a regra dos quatro quintos.

Trocar de modo **não descarta a simulação**: o cliente editado continua lá quando se volta para a análise. A página consome as mesmas rotas que qualquer outro cliente da API e [está documentada junto do serviço](docs/servico.md#-a-página-do-fluxo).

```bash
curl -X POST localhost:3000/risk-score -H 'content-type: application/json' -d '{
  "durationMonths": 48, "creditAmount": 9000, "installmentRate": 4,
  "residenceSince": 2, "age": 24, "existingCredits": 2, "dependents": 1,
  "checkingStatus": 0, "creditHistory": 1, "purpose": 0, "savingsStatus": 0,
  "employmentYears": 1, "otherDebtors": 0, "property": 3,
  "otherInstallments": 0, "housing": 0, "job": 1,
  "telephone": 0, "foreignWorker": 0 }'
```

```json
{ "riskProbability": 0.741019, "classification": "HIGH_RISK", "threshold": 0.158696,
  "model": { "source": "german", "features": 57, "savedAt": "2026-08-28T10:47:53.549Z" } }
```

A parte difícil de servir um modelo não é o HTTP — é que os pesos sozinhos não bastam. [A seção sobre o serviço](docs/servico.md#-o-problema-que-a-api-expõe) mostra por quê.

Cada fonte declara a **sua** dose — o dataset real vem com os freios ligados, o sintético não —, e as flags sobrescrevem o que a fonte declara. Cada execução imprime `Diferença treino − teste` logo abaixo das acurácias. É o termômetro do *overfitting*, e é o número que [a regularização existe para encolher](docs/modelo.md#-regularização-l2-e-dropout).

Os dados **não** mudam entre execuções, e agora nem entre máquinas: o CSV real é fixo, e o sintético é gerado por um [PRNG com semente](docs/dados-sinteticos.md#-geração-dos-dados-sintéticos) — `npm run seed` reproduz o arquivo versionado byte a byte. Os pesos iniciais da rede continuam aleatórios, então as métricas oscilam um pouco a cada rodada; é esperado, e é por isso que os números desta documentação são **médias de 15 execuções** com erro padrão.

### Estrutura

```text
tfjs-credit-risk-classifier/
├── index.js               # porta de entrada: lê os argumentos e reexporta o que `src/` define
├── src/                   # um módulo por seção, numerados na ordem em que se leem
│   ├── 00-constants.js    # faixas, sementes, limiar, doses de regularização, topologia
│   ├── 01-preprocess.js   # normalização do sintético, vetor de features e `classify`
│   ├── 02-synthetic.js    # PRNG com semente, ruído, desbalanceamento e geração dos clientes
│   ├── 03-csv.js          # ida e volta do CSV: escrita com precisão por coluna e leitura
│   ├── 04-german.js       # parse do arquivo bruto da UCI e conversão para clientes
│   ├── 05-scaler.js       # min-max ajustado SÓ no treino
│   ├── 06-encoding.js     # one-hot e ordinal das colunas qualitativas
│   ├── 07-audit.js        # auditoria por sexo e mitigação por limiar de grupo
│   ├── 08-sources.js      # as três fontes: sintética, German one-hot e German ordinal
│   ├── 08a-cli.js         # leitura das flags (`--source`, `--cv`, `--units`, `--l2`, …)
│   ├── 09-split.js        # embaralhamento, split estratificado e dobras
│   ├── 10-model.js        # construção da MLP, regularizadores e configuração de treino
│   ├── 11-persistence.js  # `model.save` e `tf.loadLayersModel`
│   ├── 12-inference.js    # predição para um cliente novo
│   ├── 13-confusion.js    # matriz de confusão
│   ├── 13a-format.js      # tabelas de texto para o relatório
│   ├── 14-metrics.js      # precision, recall e F1
│   ├── 15-roc.js          # curva ROC e AUC
│   ├── 16-threshold.js    # matriz de custo e escolha do limiar
│   ├── 16a-evaluate.js    # baseline da classe majoritária e avaliação no teste
│   ├── 17-cross-validation.js  # k dobras estratificadas, média e erro padrão
│   ├── 17a-architectures.js    # comparação das oito topologias
│   ├── 18-main.js         # o fluxo completo de uma execução
│   ├── 19-artifacts.js    # o pacote servido: pesos + scaler + limiar + contrato
│   ├── 20-contract.js     # validação do payload que chega pela API
│   ├── 21-api.js          # o servidor `node:http` e as três rotas
│   └── 22-web.js          # os arquivos da página, servidos pelo mesmo processo
├── web/                   # a página do fluxo: sem build, sem dependência
│   ├── index.html
│   ├── styles/            # tokens (cores, espaçamento, tema claro/escuro) e layout
│   └── js/                # api → mappers → componentes (custom elements)
├── scripts/
│   ├── fetch-german.js    # baixa o German Credit da UCI e converte
│   ├── seed.js            # regera o CSV sintético (ruído e balanço configuráveis)
│   └── serve.js           # sobe a API a partir do pacote salvo
├── test/
│   └── index.test.js      # testes com o runner nativo do Node
├── docs/                  # a documentação longa: medições, tabelas e decisões
│   ├── dados-sinteticos.md
│   ├── german-credit.md
│   ├── modelo.md
│   ├── metricas.md
│   ├── validacao-cruzada.md
│   ├── mitigacao.md
│   ├── inferencia.md
│   ├── servico.md
│   ├── api.md
│   ├── testes.md
│   └── fluxo-da-analise.gif   # a página em funcionamento, no topo deste README
├── package.json
├── package-lock.json
├── .nvmrc                 # versão do Node suportada
├── .gitignore
├── data/
│   ├── german-credit.csv  # dataset REAL (UCI/Statlog), convertido e versionado
│   └── customers.csv      # dataset sintético versionado, reproduzível pela semente
├── model/                 # gerado por `npm start` — ignorado pelo git
│   ├── model.json         # topologia + training config
│   ├── weights.bin        # pesos e estado do otimizador
│   └── metadata.json      # scaler, ordem das features e limiar — o contrato dos pesos
└── README.md
```

### Testes

```bash
npm test           # roda a suíte uma vez
npm run test:watch # re-executa a cada alteração
```

São **458 testes** no runner nativo do Node (`node:test` + `node:assert`) — nenhuma dependência de desenvolvimento —, escritos em **Given / When / Then**. A [tabela de cobertura completa](docs/testes.md), alvo por alvo, está na documentação.

> ⛔ Se o `npm install` ou o `npm start` quebrarem, o culpado quase sempre é a versão do Node: veja [Solução de problemas](docs/testes.md#-solução-de-problemas).

---

## 🌐 A API

```http
POST /risk-score
Content-Type: application/json
```

As sete numéricas em unidades brutas e as doze qualitativas como índice do código na lista da UCI — 19 campos, os mesmos 19 que viram as 57 entradas da rede:

```json
{
  "durationMonths": 48, "creditAmount": 9000, "installmentRate": 4,
  "residenceSince": 2, "age": 24, "existingCredits": 2, "dependents": 1,

  "checkingStatus": 0, "creditHistory": 1, "purpose": 0, "savingsStatus": 0,
  "employmentYears": 1, "otherDebtors": 0, "property": 3,
  "otherInstallments": 0, "housing": 0, "job": 1,
  "telephone": 0, "foreignWorker": 0
}
```

```json
{
  "riskProbability": 0.741019,
  "classification": "HIGH_RISK",
  "threshold": 0.158696,
  "model": { "source": "german", "features": 57, "savedAt": "2026-08-28T10:47:53.549Z" }
}
```

O limiar viaja na resposta porque ele é uma escolha de negócio, não uma propriedade do modelo — e não é o `0.5` herdado, é o que a [matriz de custo escolheu](docs/metricas.md#-ajuste-do-limiar-de-decisão).

```mermaid
flowchart LR
    A(["🖥️ Cliente HTTP"]) --> B["🌐 API node:http<br/>POST /risk-score"]
    B --> V{"🛡️ Contrato<br/>19 campos válidos?"}
    V -->|"não"| X(["📤 400<br/>lista de erros"])
    V -->|"sim"| C["⚙️ Pré-processamento<br/>scaler do metadata.json"]
    C --> D["🧠 Modelo carregado<br/>tf.loadLayersModel"]
    D --> E["📈 Probabilidade<br/>0.741019"]
    E --> F(["📤 200<br/>HIGH_RISK"])

    classDef edge fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0c4a6e
    classDef step fill:#f1f5f9,stroke:#64748b,stroke-width:1.5px,color:#1e293b
    classDef key  fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef bad  fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class A,F edge
    class B,D,E step
    class C,V key
    class X bad
```

O ponto crítico: o serviço precisa aplicar **exatamente a mesma normalização** do treino. Divergência entre treino e inferência (*training-serving skew*) é uma das falhas mais comuns em ML em produção, e a pior característica dela é não dar erro — a rede recebe números que não significam o que os pesos aprenderam, devolve uma probabilidade plausível, e o serviço responde `200 OK`.

Por isso o `npm start` não salva mais só os pesos. Ele salva um **pacote**: `model.json`, `weights.bin` e um `metadata.json` com o `scaler` medido naquele treino, a ordem exata das 57 features e o limiar escolhido. Na subida, o serviço **recusa** o pacote se a rede esperar outro número de entradas ou se a lista de features gravada não bater com a que o código gera hoje — o caso em que o tamanho do vetor continua certo e o significado de cada posição, não.

E a validação do payload, que o CSV nunca precisou ter, aqui é obrigatória: um `age` ausente viraria `NaN`, e `NaN >= limiar` é `false` — o cliente sairia classificado como baixo risco por não ter mandado a idade. [Os detalhes estão na documentação do serviço](docs/servico.md#-por-que-validar-se-o-csv-nunca-foi-validado), inclusive por que `personalStatus` é recusado em vez de ignorado, e por que a API serve só a política de limiar único.

---

## ⚠️ Limitações conhecidas

Coisas deliberadamente simplificadas — cada uma é um bom exercício de correção:

| Simplificação                                                  | Por que importaria em produção                              |
| -------------------------------------------------------------- | ----------------------------------------------------------- |
| **1.073 parâmetros para 640 linhas** de treino efetivo           | Capacidade muito acima do dado disponível. A [regularização](docs/modelo.md#-regularização-l2-e-dropout) corta a diferença treino−teste pela metade, mas **não** recupera AUC — o gargalo é o dado, não a capacidade |
| Dataset real com apenas **1.000 linhas**                         | Pouco dado para uma rede neural — boa parte da variação entre execuções vem daí |
| Todos os níveis one-hot mantidos (*dummy variable trap*)         | Inofensivo em rede neural, quebraria uma regressão linear |
| `savingsStatus` trata "desconhecido" como mais um nível          | Ausência de dado não é uma categoria como as outras; o certo seria um indicador de faltante separado |
| Mitigação da disparidade **desligada por padrão**                | O limiar por grupo existe e funciona ([medido](docs/mitigacao.md#-mitigação-da-disparidade)), mas exige ler o atributo protegido na decisão e, no *hold-out* de 200 linhas, corrige com a dose errada. Ligá-la é decisão de política |
| Mitigação limitada à **paridade demográfica**                    | Igualar aprovação é um critério entre vários; igualar FNR ou odds daria outra política, e as duas não podem valer juntas com taxas-base diferentes |
| Nada é feito contra o **vazamento do atributo protegido**         | O sexo é previsível a partir das features do modelo com AUC `0.699`; a mitigação compensa o efeito no fim da linha, sem remover a causa |
| CSV sem validação de esquema, faixas ou valores ausentes        | Dado real vem com coluna faltando, texto onde deveria haver número e `NaN` |
| Custos de FP e FN fixos no código (`1` e `5`)                    | Aqui vêm da matriz oficial do dataset; em produção viriam de ticket médio, taxa de recuperação e margem |
| `npm start` continua reportando **uma** divisão                  | O relatório completo — matriz, curva, limiar, auditoria — roda sobre um hold-out só, e a divisão fixa é reconhecidamente ruim. A estimativa honesta está a um comando de distância ([`npm run cv`](docs/validacao-cruzada.md#-validação-cruzada-e-split-estratificado)), mas não é o padrão |
| Regularização escolhida em uma grade medida no **mesmo** protocolo de avaliação | Trinta configurações comparadas sem conjunto de validação separado; o certo seria escolher em um conjunto e reportar em outro — o mesmo problema do [limiar](docs/metricas.md#-ajuste-do-limiar-de-decisão), algumas linhas acima |
| Tratamento do **desbalanceamento** limitado a `classWeight`      | `--balancear` existe e está [medido](docs/modelo.md#-peso-por-classe-o-desbalanceamento-atacado-durante-o-treino) — o resultado é pior, e por isso o padrão é desligado. Reamostragem, SMOTE e *focal loss* continuam sem medida |
| Ruído sintético **normal e independente** por coluna             | Erro real é enviesado (renda é subdeclarada, não sorteada em torno da verdade) e correlacionado entre colunas |
| Ruído de rótulo **simétrico**                                    | Na prática um mau pagador registrado como bom é bem mais comum que o contrário |

Duas limitações da versão anterior **deixaram de existir** com o dataset real:

| Era limitação | O que resolveu |
| ------------- | -------------- |
| ~~Normalização com constantes fixas em vez de estatísticas do treino~~ | `fitMinMaxScaler`, [medido só no treino](docs/german-credit.md#agora-a-normalização-precisa-ser-medida) |
| ~~Split por fatiamento sem embaralhar antes~~ | `shuffle` com [semente fixa](docs/german-credit.md#reprodutibilidade) antes do corte |
| ~~Categorias codificadas como ordinais~~ | [One-hot](docs/german-credit.md#por-que-one-hot-e-não-ordinal) nas 12 qualitativas |
| ~~8 das 20 colunas aproveitadas~~ | 19 colunas; a vigésima é [auditoria, não feature](docs/german-credit.md#a-coluna-que-o-modelo-não-recebeu) |
| ~~Dataset sintético limpo e quase equilibrado~~ | [Ruído e desbalanceamento](docs/dados-sinteticos.md#-geração-dos-dados-sintéticos) injetados, com o efeito de cada um medido |
| ~~Nenhuma defesa contra overfitting além do early stopping~~ | [L2 e dropout](docs/modelo.md#-regularização-l2-e-dropout), com a dose declarada por fonte e medida em grade |
| ~~CSV sintético gerado com `Math.random()`, irreprodutível~~ | Gerador com semente: `npm run seed` reconstrói o arquivo versionado byte a byte |
| ~~Split simples em vez de estratificado~~ | [`stratifiedSplitCustomers`](docs/validacao-cruzada.md#-validação-cruzada-e-split-estratificado): o baseline do teste passou a sair `0.7000` em **todas** as divisões |
| ~~Sem validação cruzada~~ | `npm run cv`: k dobras estratificadas, média com erro padrão, e a auditoria de disparidade rodando sobre os 1.000 clientes |
| ~~Modelo salvo sem versionar o scaler junto~~ | [`metadata.json`](docs/servico.md#-a-correção-o-modelo-como-pacote): escala, ordem das features e limiar viajam com os pesos, e a carga recusa o pacote se qualquer um dos três não bater |
| ~~Limiar escolhido no mesmo conjunto em que é medido~~ | [Fatia de calibração](docs/metricas.md#onde-o-corte-é-escolhido): o treino se parte em 640 para ajustar os pesos e 160 para escolher o corte, e o teste só é tocado na hora de reportar. Vale para o fluxo principal **e** para cada dobra da validação cruzada |
| ~~A camada da página sem teste nenhum~~ | `web/package.json` declara os módulos como ESM, e as funções puras de `mappers.js` e `domain.js` passaram a ser cobertas pelo mesmo `npm test` — sem build e sem dependência |

---

## 🛠️ Próximas evoluções

**Métricas e avaliação**
- [x] ~~matriz de confusão~~ — feito, veja [Matriz de confusão](docs/metricas.md#-matriz-de-confusão);
- [x] ~~precision, recall e F1-score~~ — feito, veja [Precision, recall e F1-score](docs/metricas.md#-precision-recall-e-f1-score);
- [x] ~~curva ROC e AUC~~ — feito, veja [Curva ROC e AUC](docs/metricas.md#-curva-roc-e-auc);
- [x] ~~ajuste do limiar de decisão a partir da curva~~ — feito, veja [Ajuste do limiar de decisão](docs/metricas.md#-ajuste-do-limiar-de-decisão).

**Modelo e dados**
- [x] ~~carregar dados de um CSV~~ — feito, veja [Carregando dados de um CSV](docs/dados-sinteticos.md#-carregando-dados-de-um-csv);
- [x] ~~usar um dataset real de crédito~~ — feito, veja [Dataset real: German Credit](docs/german-credit.md#-dataset-real-german-credit);
- [x] ~~codificar as categóricas com *one-hot* em vez de ordinal, e aproveitar as colunas restantes~~ — feito, veja [Por que one-hot e não ordinal](docs/german-credit.md#por-que-one-hot-e-não-ordinal);
- [x] ~~adicionar ruído e desbalanceamento aos dados sintéticos~~ — feito, veja [Geração dos dados sintéticos](docs/dados-sinteticos.md#-geração-dos-dados-sintéticos);
- [x] ~~regularização L2 e dropout~~ — feito, veja [Regularização: L2 e dropout](docs/modelo.md#-regularização-l2-e-dropout);
- [x] ~~mitigar a disparidade medida, não só reportá-la~~ — feito, veja [Mitigação da disparidade](docs/mitigacao.md#-mitigação-da-disparidade);
- [x] ~~validação cruzada e split estratificado~~ — feito, veja [Validação cruzada e split estratificado](docs/validacao-cruzada.md#-validação-cruzada-e-split-estratificado);
- [x] ~~comparar arquiteturas diferentes~~ — feito, veja [Comparando arquiteturas](docs/modelo.md#-comparando-arquiteturas).

**Produto**
- [x] ~~testes automatizados~~ — feito, veja [Testes](#testes);
- [x] ~~salvar e recarregar o modelo (`model.save` / `tf.loadLayersModel`)~~ — feito, veja [Persistência do modelo](docs/inferencia.md#-persistência-do-modelo);
- [x] ~~API REST com endpoint `POST /risk-score`~~ — feito, veja [O serviço](docs/servico.md#-o-serviço-api-rest-pacote-servido-e-contrato-de-entrada);
- [x] ~~frontend para simular clientes~~ — feito, veja [A página do fluxo](docs/servico.md#-a-página-do-fluxo);
- [ ] inferência no navegador com TensorFlow.js;
- [ ] autenticação e log estruturado das decisões servidas.

---

## 📚 Conceitos demonstrados

`Redes neurais` · `Perceptron` · `MLP` · `Dense Layers` · `ReLU` · `Sigmoid` · `Classificação binária` · `Binary Crossentropy` · `Adam` · `Learning rate` · `Batch size` · `Epoch` · `Train/Validation/Test` · `Early Stopping` · `Overfitting` · `Normalização` · `Min-max scaling` · `Data leakage` · `Codificação ordinal` · `One-hot encoding` · `Dummy variable trap` · `Baseline da classe majoritária` · `Desbalanceamento de classes` · `Ruído de medição` · `Ruído de rótulo` · `Erro irredutível` · `Box-Muller` · `PRNG com semente` · `Quantil` · `Matriz de confusão` · `Precision/Recall/F1` · `Curva ROC` · `AUC` · `Matriz de custo` · `Ajuste de limiar` · `Reprodutibilidade` · `Validação cruzada k-fold` · `Amostragem estratificada` · `Regularização L2` · `Dropout` · `Fairness` · `Disparate impact` · `Regra dos quatro quintos` · `Paridade demográfica` · `Igualdade de erros` · `Mitigação por pós-processamento` · `Limiar por grupo` · `Inferência` · `Gerenciamento de tensores` · `Model serving` · `Training-serving skew` · `Versionamento de artefatos` · `Validação de esquema` · `API REST` · `Testes automatizados`

---

## 🔒 Aviso

Projeto criado para **estudo de redes neurais e TensorFlow.js**. O modelo **não deve ser usado para decisões financeiras reais**.

O dataset real é de **1994**, tem 1.000 registros de um único banco alemão e reflete as práticas de concessão daquele contexto — inclusive as discriminatórias. Ele serve para estudar o método, não para tirar conclusões sobre crédito hoje.

A [auditoria por sexo](docs/german-credit.md#a-coluna-que-o-modelo-não-recebeu) incluída aqui é uma demonstração didática de uma técnica, não um parecer. Avaliar viés em um sistema de crédito real exige análise causal, contexto jurídico e revisão humana — nada disso cabe em um `README`.

O mesmo vale, com força redobrada, para a [mitigação](docs/mitigacao.md#-mitigação-da-disparidade). Aplicar um limiar diferente por sexo é usar o atributo protegido na decisão, o que em vários ordenamentos jurídicos é ilegal **mesmo com intenção corretiva**. A flag `--mitigar` existe para que o efeito e o preço possam ser medidos, não para ser ligada em produção.

Modelos de crédito em produção exigem, entre outros pontos: dados representativos, validação estatística, análise de viés, explicabilidade, governança, monitoramento contínuo, segurança e conformidade regulatória.

---

## 📖 Referências

- *Deep Learning* — Ian Goodfellow, Yoshua Bengio, Aaron Courville
- *Deep Learning with Python* — François Chollet
- [TensorFlow.js — Documentação](https://www.tensorflow.org/js)
- [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node)
- Hofmann, H. (1994). **Statlog (German Credit Data)**. UCI Machine Learning Repository. [DOI: 10.24432/C5NC77](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — CC BY 4.0
- Barocas, S., Hardt, M., Narayanan, A. — [*Fairness and Machine Learning*](https://fairmlbook.org/) (sobre por que os critérios de justiça são incompatíveis entre si quando as taxas-base diferem)
- Box, G. E. P., Muller, M. E. (1958). **A Note on the Generation of Random Normal Deviates**. *Annals of Mathematical Statistics*, 29(2) — a transformação usada para gerar o [ruído de medição](docs/dados-sinteticos.md#ruído-de-medição-um-teto-que-nenhum-modelo-ultrapassa)
- Frénay, B., Verleysen, M. (2014). **Classification in the Presence of Label Noise: a Survey**. *IEEE Transactions on Neural Networks and Learning Systems*, 25(5) — por que rótulo errado dói mais na classe minoritária
- He, H., Garcia, E. A. (2009). **Learning from Imbalanced Data**. *IEEE Transactions on Knowledge and Data Engineering*, 21(9) — o problema que a [taxa de 15,8%](docs/dados-sinteticos.md#desbalanceamento-o-limiar-virou-um-quantil) introduz de propósito
- Hardt, M., Price, E., Srebro, N. (2016). **Equality of Opportunity in Supervised Learning**. *NeurIPS* — o pós-processamento por limiar de que a [mitigação](docs/mitigacao.md#-mitigação-da-disparidade) deste projeto é a variante mais simples
- Kohavi, R. (1995). **A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection**. *IJCAI* — por que k dobras **estratificadas**, e não um hold-out, estimam acurácia
- Srivastava, N. et al. (2014). **Dropout: A Simple Way to Prevent Neural Networks from Overfitting**. *JMLR*, 15 — o freio que [não acrescenta parâmetro nenhum](docs/modelo.md#-regularização-l2-e-dropout)

---

## 📄 Licença

Uso livre para fins de estudo e experimentação.
