# 🌐 O serviço: API REST, pacote servido e contrato de entrada

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · [German Credit](german-credit.md) · [Modelo](modelo.md) · [Métricas](metricas.md) · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · **Serviço** · [API do módulo](api.md) · [Testes](testes.md)

---

## 🚀 Subir

```bash
npm start        # treina e salva o pacote em `model/`
npm run serve    # sobe a API lendo esse pacote
```

```text
API ouvindo em http://localhost:3000
  fonte:    german (57 features)
  limiar:   0.1099 — menor custo (FP=1, FN=5)
  treinado: 2026-08-28T02:37:23.428Z

  POST /risk-score   pontua um cliente
  GET  /schema       o contrato de entrada
  GET  /health       o pacote carregado
```

O mesmo endereço aberto no navegador mostra [a página do fluxo](#-a-página-do-fluxo): os campos que entraram, as etapas que os relacionaram e a probabilidade que saiu.

Os dois comandos são separados de propósito. Treinar demora minutos, acontece uma vez e produz um artefato; servir sobe em segundos a partir desse artefato e responde milhares de vezes. Misturar os dois é o caminho mais curto para um serviço que retreina a cada *deploy* — e que, portanto, responde diferente a cada *deploy*.

`--port=8080` escolhe a porta. `--port=0` pede uma livre ao sistema, que é como os testes sobem o serviço sem disputar a 3000 com nada.

---

## 🧨 O problema que a API expõe

A parte difícil de servir um modelo não é o HTTP. São **oito linhas** de `node:http` e nenhuma dependência nova.

A parte difícil é que, até aqui, o modelo salvo estava **incompleto** — e ninguém percebia, porque quem recarregava era o mesmo processo que tinha acabado de treinar:

```javascript
await saveModel(model);        // pesos
const loaded = await loadModel();
predictRisk(loaded, cliente, toVector);   // `toVector` ainda estava na memória
```

Esse `toVector` carrega o `scaler` — o mínimo e a amplitude de cada coluna numérica, **medidos no conjunto de treino**. Um serviço não tem isso. Ele sobe horas depois, em outra máquina, e recebe um cliente em unidades brutas:

```json
{ "durationMonths": 48, "creditAmount": 9000, "age": 24 }
```

Para virar o vetor que a rede espera, `creditAmount = 9000` precisa virar `(9000 − min) / range` com **exatamente** o `min` e o `range` daquele treino. Se o serviço remedir a escala com os dados que estiver vendo, ou pior, chutar constantes, a rede recebe números que não significam o que os pesos aprenderam.

E aqui está o ponto: **nada quebra**. Não há exceção, não há `NaN`, não há log de erro. A rede recebe `0.9` onde deveria receber `0.4`, devolve uma probabilidade plausível, e o serviço responde `200 OK`. É o *training-serving skew*, e ele não falha — ele **degrada em silêncio**.

Era a última limitação da lista do README que ainda não tinha sido corrigida, e a API é o que a tornou impossível de adiar.

---

## 📦 A correção: o modelo como pacote

O modelo deixa de ser um arquivo de pesos e passa a ser um **pacote**. Junto de `model.json` e `weights.bin` passa a viver um terceiro arquivo:

```text
model/
├── model.json      # topologia + training config
├── weights.bin     # pesos e estado do otimizador
└── metadata.json   # o contrato que os pesos pressupõem
```

```json
{
  "version": 1,
  "savedAt": "2026-08-28T02:37:23.428Z",
  "source": "german",
  "encoding": "onehot",
  "featureNames": ["durationMonths", "creditAmount", "…", "foreignWorker=A202"],
  "scaler": {
    "featureNames": ["durationMonths", "creditAmount", "…"],
    "min":   { "durationMonths": 4,  "creditAmount": 250,   "age": 19 },
    "range": { "durationMonths": 68, "creditAmount": 18174, "age": 56 }
  },
  "threshold": 0.109897,
  "thresholdStrategy": "menor custo (FP=1, FN=5)",
  "training": { "customers": 800, "units": [16, 8], "l2": 0.003, "dropout": 0.2 }
}
```

Quatro coisas que os pesos sozinhos não dizem:

| Campo | Sem ele |
| ----- | ------- |
| `scaler` | O serviço normaliza diferente do treino. É o *skew*. |
| `featureNames` | A ordem das 57 entradas é o que liga cada valor ao peso certo. Trocar duas de lugar não dá erro nenhum. |
| `threshold` | `0.31` não significa nada sem o corte. O laboratório **escolhe** o corte pela matriz de custo; servir com `0.5` jogaria fora essa escolha. |
| `source` / `encoding` | Qual fonte e qual codificação reconstroem o `toVector` certo. |

E quem serve não precisa saber que nada disso existe:

```javascript
const artifacts = await loadArtifacts();
artifacts.predict(cliente);   // scaler, ordem e codificação já embutidos
```

É por **não precisar saber** que o serviço não tem como errar.

### As três checagens de carga

`loadArtifacts` recusa o pacote antes da primeira predição, porque depois dela ninguém mais percebe:

| Checagem | O que pega |
| -------- | ---------- |
| `version` | O formato do pacote mudou desde que este modelo foi salvo. |
| `inputs[0].shape[1]` vs `featureNames.length` | A rede espera 57 entradas e o pacote declara outra coisa. |
| `source.featureNames` vs `metadata.featureNames` | **A mais sutil.** Alguém acrescentou uma coluna, mudou a ordem de `GERMAN_CATEGORICAL` ou trocou a codificação. O vetor pode até continuar com 57 posições — mas cada posição passou a significar outra coisa para os mesmos pesos. |

A terceira é a única que pega o caso em que o tamanho bate e o **significado** não. É exatamente o *skew* de novo, com outra roupa.

### O limiar `Infinity`

O primeiro ponto de toda curva ROC é `{ fpr: 0, tpr: 0, threshold: Infinity }` — o corte que não aprova ninguém. Ele quase nunca ganha a comparação por custo. "Quase nunca" não é "nunca", e `JSON.stringify(Infinity)` é `null`.

Um limiar `null` no pacote faria `probabilidade >= null` virar `probabilidade >= 0`: **todo** cliente sairia como alto risco, sem erro nenhum aparecer. `saveArtifacts` recusa gravar um limiar que não esteja em `[0, 1]` — descobrir isso na hora de escrever é mais barato do que descobrir pela taxa de aprovação do serviço.

---

## 📥 O contrato de entrada

```http
POST /risk-score
Content-Type: application/json
```

As sete numéricas em unidades **brutas** e as doze qualitativas como **índice do código** na lista da UCI:

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
  "riskProbability": 0.781903,
  "classification": "HIGH_RISK",
  "threshold": 0.109897,
  "model": { "source": "german", "features": 57, "savedAt": "2026-08-28T02:37:23.428Z" }
}
```

O limiar viaja na resposta porque ele é uma **escolha de negócio**, não uma propriedade do modelo: sem ele, `0.78` não diz se o cliente foi aprovado. E ele não é o `0.5` herdado — é o `0.109897` que a matriz de custo escolheu, gravado no pacote.

> A probabilidade e o limiar são arredondados na **mesma** casa (6). Arredondar só um dos dois produziria, na fronteira, um JSON que se contradiz: `riskProbability` igual ao `threshold` e classificação `LOW_RISK`.

### `GET /schema`

Descobrir o contrato batendo no `400` é um jeito ruim de integrar, então o serviço o publica:

```json
{
  "source": "german",
  "threshold": 0.109897,
  "request": {
    "numeric": ["durationMonths", "creditAmount", "installmentRate", "…"],
    "categorical": {
      "purpose": { "range": [0, 9], "codes": ["A40", "A41", "…", "A410"] },
      "telephone": { "range": [0, 1], "codes": ["A191", "A192"] }
    },
    "rejected": ["personalStatus"]
  }
}
```

---

## 🛡️ Por que validar, se o CSV nunca foi validado

O README lista, entre as limitações, que o CSV entra **sem** validação de esquema. Continua verdade, e continua defensável: aquele CSV foi escrito por este projeto.

Um endpoint HTTP inverte isso. A entrada passa a vir de fora, em JSON, escrita por alguém que não leu o código — e cada suposição não checada vira um jeito diferente de a rede receber lixo e responder um número que parece legítimo:

| Entrada | Sem validação |
| ------- | ------------- |
| `age` ausente | `undefined − min / range` é `NaN`. O `NaN` atravessa o tensor, a rede devolve `NaN`, e `NaN >= limiar` é `false`: o cliente sai **BAIXO RISCO** por não ter mandado a idade. |
| `"durationMonths": "48"` | Funciona por coerção em quase toda conta e falha justamente onde não deveria. |
| `"durationMonth": 48` (sem o "s") | Ignorado em silêncio; o campo certo fica ausente e cai no primeiro caso. |
| `"purpose": 99` | One-hot de índice 99 em vetor de 10 posições: dez zeros. A categoria some. |

Nenhum desses é sofisticado. É só a diferença entre um script e um serviço: **o script pode confiar na entrada, o serviço não pode.**

A validação devolve os erros **todos** de uma vez, com os códigos válidos junto:

```json
{
  "error": "Requisição inválida.",
  "details": [
    "`age`: esperado um número; recebido \"24\" (string).",
    "`purpose`: esperado um inteiro entre 0 e 9 (A40, A41, A42, A43, A44, A45, A46, A48, A49, A410); recebido 99 (number).",
    "`durationMonth`: campo desconhecido."
  ]
}
```

### O campo que é recusado, não esquecido

`personalStatus` — a coluna que carrega o sexo — não está entre os campos aceitos, e não é por descuido:

```json
{ "error": "Requisição inválida.", "details": [
  "`personalStatus` não é aceito: é o atributo protegido, e o modelo nunca o recebeu. A auditoria usa essa coluna; a decisão, não."
] }
```

Ignorá-lo em silêncio seria pior do que recusá-lo: quem chama ficaria achando que mandou algo que foi usado.

Pelo mesmo motivo, **a API serve apenas a política de limiar único**. A [mitigação por grupo](mitigacao.md#-mitigação-da-disparidade) existe, funciona e está medida — mas servi-la exigiria ler o atributo protegido em cada requisição, que é precisamente o ato que o README argumenta ser ilegal em vários ordenamentos, mesmo com intenção corretiva. O `--mitigar` continua onde deve estar: no laboratório, onde o efeito e o preço podem ser medidos.

---

## 📋 As respostas

| Código | Quando |
| -----: | ------ |
| `200` | Cliente pontuado. |
| `400` | Corpo vazio, JSON malformado, ou o payload não passou no contrato (`details` traz a lista). |
| `404` | Rota desconhecida — a resposta lista as que existem. |
| `405` | Método errado na rota certa; vem com o cabeçalho `Allow`. |
| `413` | Corpo acima de 16 KB. |
| `415` | Sem `content-type: application/json` — quase sempre é JSON com o cabeçalho esquecido. |
| `500` | Erro interno. A stack vai para o log, **não** para a resposta: devolvê-la entregaria caminho de arquivo e versão de biblioteca a quem só mandou um POST. |

Dois detalhes que só aparecem quando se escreve o servidor à mão:

**O teto do corpo vem antes da validação de esquema.** A validação protege contra um objeto errado; ela não protege contra o que nunca chega a ser um objeto. Um POST de 50 MB vira 50 MB de `Buffer` antes de qualquer `JSON.parse` rodar.

**Recusar não é o mesmo que desligar.** A primeira versão deste código chamava `req.destroy()` ao estourar o teto — direto e errado: a conexão morria antes de a resposta sair, e o cliente via um *reset* em vez do `413`, ficando sem saber por que foi recusado. O certo é parar de acumular (`req.pause()`), responder, e fechar a conexão **na resposta** (`connection: close`) — porque a requisição ficou pela metade e o resto do upload seria lido como se fosse a próxima.

---

## 🖥️ A página do fluxo

`http://localhost:3000/` não devolve JSON: devolve uma página com três modos sobre o mesmo pacote — **Análise**, o caminho que o dado percorre até virar uma probabilidade; **Treinamento**, como a rede aprendeu a percorrê-lo; e **Avaliação**, quanto ela acerta e quem ela penaliza.

```text
DADOS UTILIZADOS          PROCESSAMENTO                    RESULTADOS DA ANÁLISE

Prazo do contrato  ──┐   Preparo   oculta 1   oculta 2       Risco de inadimplência  78,2%
Valor do crédito   ──┤                                       ████████████████████░░░░░
Idade              ──┼──▶  ○ ──────  ○ ────────  ○ ──┐              ┆ limiar 11,0%
Conta corrente     ──┤   min–max     ○           ○   ├──▶ ○ ──▶
Finalidade         ──┤                ○          ○   │  sigmoide   Probabilidade de adimplência
Moradia            ──┴──▶  ○ ──────   ○ ─────────────┘                                21,8%
                         one-hot                             █████░░░░░░░░░░░░░░░░░░░░
```

**Os círculos do meio não são enfeite.** Cada coluna é uma etapa que existe no pipeline, com o nome que ela tem no código: a escala min–max de `05-scaler.js`, o one-hot de `06-encoding.js`, as camadas ocultas de `10-model.js` e a sigmoide. E as ligações da esquerda também são verdade: as sete colunas numéricas vão para o min–max, as doze qualitativas vão para o one-hot, e as duas coisas **só se encontram na primeira camada oculta** — é ali que os sinais começam a ser combinados.

Uma camada de 16 unidades vira 4 círculos, e a legenda diz exatamente isso (`16 unidades · 4 representadas`). Desenhar 4 bolinhas e chamá-las de "16 unidades" seria mentir em silêncio; desenhar 16 seria uma parede ilegível.

**São dois resultados, não três.** O modelo tem uma saída — uma probabilidade —, e a segunda barra é o complemento dela. Uma terceira faixa apresentaria uma decisão de layout como se fosse uma decisão do modelo.

### Simular um cliente

Os 19 campos são **editáveis**, e cada alteração repontua. Mudar a conta corrente de "saldo negativo" para "200 DM ou mais" derruba o risco na hora — é a variável mais forte do German Credit, e vê-la mexer explica mais do que qualquer tabela de pesos.

**Nenhum campo é escrito à mão na página.** O tipo de cada controle sai do contrato: as sete numéricas viram `<input type="number">`, as doze qualitativas viram `<select>` cujas opções são os códigos do `GET /schema` traduzidos pelo dicionário de domínio. Um retreino que mude as colunas muda o formulário sozinho.

`personalStatus` continua na tela e continua **não editável** — porque não é enviado. É a mesma decisão de sempre, agora visível: o campo que a auditoria usa e a decisão não.

Três detalhes que só aparecem quando alguém usa de verdade:

**Debounce.** Sem ele, digitar `9000` dispara quatro requisições, e as três primeiras pontuam clientes que ninguém quis simular.

**Ordem das respostas.** O debounce reduz as requisições; não as ordena. Duas em voo podem voltar trocadas, e a resposta antiga sobrescrever a nova — a tela mostraria o resultado de um cliente que já não está no formulário. Cada pedido leva um número de sequência, e só o último escreve.

**Campo vazio não é enviado.** Apagar um número para digitar outro deixaria o campo em `null` por um instante; mandar isso renderia um `400` correto e um efeito ruim — a probabilidade sumindo da tela no meio da edição. O estado inválido fica local até virar um número de novo.

### Fora da faixa vista no treino

O `GET /schema` passou a publicar `observedRange`: o mínimo e o máximo que cada coluna numérica teve no treino, reconstruídos do `scaler` do pacote.

```json
"observedRange": {
  "durationMonths": { "min": 4,   "max": 72    },
  "creditAmount":   { "min": 250, "max": 18424 },
  "age":            { "min": 19,  "max": 75    }
}
```

Ele **não valida nada**. `creditAmount: 500000` continua sendo aceito, continua saindo de `[0, 1]` de propósito, e continua sendo pontuado. O que mudou é que agora a página consegue **avisar**: o campo fica amarelo e mostra `treino: 250 a 18424`.

Era a última limitação da lista deste documento que ninguém conseguia ver acontecer. Ela não foi corrigida — um valor absurdo ainda recebe uma probabilidade de aparência normal —, mas deixou de ser silenciosa para quem usa a tela.

### Modo treinamento

A mesma rede tem duas coisas para contar: **como ela decide** e **como ela aprendeu a decidir**. O seletor no topo troca entre as duas, e a coluna do meio — os nós e as ligações — é a mesma nas duas. É de propósito: não são duas telas, é a mesma rede vista por outro ângulo.

```text
   Passo à frente  │  Erro medido  │  Passo atrás  │  Pesos ajustados
   ────────────────┴───────────────┴───────────────┴─────────────────
                            ▶ época 7 de 25

  COMO FOI TREINADO          O LAÇO DO TREINO           A CURVA DO TREINO
                                                          0.78 ╲
  800 clientes de treino      ○ ─── ○ ─── ○                    ╲╲___ treino
  57 → 16 → 8 → 1          ╱  │ ╳  │ ╳  │  ╲                    ╲   ‾‾‾──
  1.073 parâmetros        ●   ○ ─── ○ ─── ○   ●                  ╲__
  Lotes de 32              ╲  │ ╳  │ ╳  │  ╱                        ‾‾──── validação
  L2 0.003 · dropout 0.2      ○ ─── ○ ─── ○                  0.42        ┆ melhor: 20
  Parou na época 25 de 40                                      1        25
```

**A curva é medida; a animação é esquema.** A distinção é explícita na tela porque as duas coisas têm status diferentes:

- Os pontos da curva vêm do `history` que o `model.fit` devolveu, época por época, gravado no pacote por `npm start`. Nenhum é suavizado ou interpolado — se a linha sobe e desce, é porque subiu e desceu.
- A animação dos quatro passos é o **procedimento**, não os valores. Não existe registro dos pesos época a época, então fingir que os círculos mostram os pesos mudando seria inventar. O que se anima é o algoritmo; o que se mede está na curva ao lado.

O passo atrás inverte o sentido dos pontos que percorrem as ligações. Não há um segundo conjunto de curvas para isso: é a mesma animação ao contrário, porque é literalmente o que a retropropagação faz — o erro volta pelas **mesmas** ligações por onde o dado veio.

**As duas linhas são o argumento do projeto em uma imagem.** A de treino continua caindo até o fim; a de validação achata na época 20 e não melhora mais. A distância entre as duas é o modelo decorando em vez de aprender, e as cinco épocas entre a melhor validação e a parada são exatamente a paciência do *early stopping*. É o laço do fluxograma do README, com números.

Sob `prefers-reduced-motion` o laço não roda sozinho: a curva aparece inteira e a barra de épocas continua funcionando. A informação é a mesma; o que some é o movimento.

### O histórico no pacote

O `model.fit` sempre devolveu o histórico e ele sempre foi descartado. Agora ele vai para o `metadata.json`, com quatro casas decimais:

```json
"training": {
  "customers": 800, "units": [16, 8], "l2": 0.003, "dropout": 0.2,
  "epochs": 40, "batchSize": 32, "validationSplit": 0.2, "patience": 5,
  "history": {
    "loss":    [0.7539, 0.7111, 0.689,  "…", 0.5334],
    "valLoss": [0.6407, 0.5809, 0.55,   "…", 0.4479]
  }
}
```

Custa alguns kilobytes e paga por si: é a **única evidência** do que aconteceu durante o treino. Os hiperparâmetros entram junto pelo mesmo motivo que o `scaler` entrou — descrever um treino de memória é o jeito mais fácil de descrever outro.

Um pacote salvo antes disso devolve o bloco sem `history`, e a tela diz isso em vez de desenhar uma curva vazia: *"rode `npm start` para treinar de novo"*.

### Modo avaliação

O terceiro modo responde a pergunta que os outros dois não respondem: **o modelo presta?**

```text
   Padrão (0.5)      │  Youden (max J)   │  ▸ Menor custo
   custo 180          │  custo 127         │  custo 107

  QUANTO ELE ACERTA        ONDE ELE ERRA            QUEM ELE PENALIZA

  A rede         72,0%     73 TN   │   67 FP        Mulheres      n=66
  ██████████████░░░░       ────────┼────────        marcados  65,2% ████████
  Chutar         70,0%      8 FN   │   52 TP        real      30,3% ████
  █████████████░░░░░                                Homens        n=134
                           precision 43,7%          marcados  53,7% ██████
  AUC 0,7417               recall    86,7%          real      29,9% ████
  treino−teste 7,6%        F1        58,1%          razão de aprovação 0,805
```

**A acurácia nunca aparece sozinha.** Ela aparece encostada no piso da classe majoritária — a taxa de quem chuta "bom pagador" para todo mundo sem olhar para nenhuma feature. 72% parece ótimo até estar ao lado de 70%, e a distância entre os dois é tudo que o treino acrescentou. Publicar um sem o outro seria publicar meia verdade.

**Os dois erros não custam igual, e a matriz mostra isso.** Recusar quem pagaria (FP) custa 1; deixar passar quem não paga (FN) custa 5. É essa assimetria que puxou o limiar de 0,5 para 0,225 — e a faixa do topo mostra os três cortes candidatos com o preço de cada um, com o que está decidindo em destaque. É a única parte da tela em que o limiar deixa de ser um número dado e vira uma **escolha com alternativas**.

**A auditoria fecha o argumento do `personalStatus`.** A tela já dizia, na coluna de entrada, que o campo é recusado porque "a auditoria usa essa coluna; a decisão, não". Faltava mostrar a auditoria. Agora ela está lá: taxa de marcação por grupo ao lado da inadimplência real do grupo, os inadimplentes que passaram batido, e a razão de aprovação com o veredito da regra dos quatro quintos.

As manchetes desses modos citam números — "Acertar 72% parece bom. Chutar acerta 70%" — e **são montadas a partir do pacote carregado**, não escritas à mão. Um retreino muda os números na tela e no texto junto; escrevê-los à mão faria a página mentir na primeira vez que alguém rodasse `npm start`.

### O que o pacote passou a guardar

Assim como o histórico de treino, tudo isto era calculado, impresso no terminal e descartado:

```json
"evaluation": {
  "baseline": 0.7, "testAccuracy": 0.72, "trainAccuracy": 0.7962,
  "testLoss": 0.536, "testCustomers": 200, "auc": 0.7417,
  "confusion": { "truePositives": 52, "trueNegatives": 73,
                 "falsePositives": 67, "falseNegatives": 8 },
  "metrics":   { "precision": 0.437, "recall": 0.8667, "f1Score": 0.581 },
  "costs":     { "falsePositive": 1, "falseNegative": 5 },
  "thresholds": [ { "label": "Menor custo", "threshold": 0.225, "cost": 107 } ],
  "audit": { "politica": "Limiar único", "women": {…}, "men": {…},
             "approvalRatio": 0.805 }
}
```

Duas escolhas que valem registro. A matriz gravada é a do limiar **escolhido**, não a do `0.5` herdado — era a única que existia apenas dentro de um `console.log`. E um limiar `Infinity` (o ponto da ROC que não aprova ninguém) é gravado como `null` de propósito: `JSON.stringify` já o transformaria em `null` de qualquer forma, e um número falso no lugar seria pior.

### De onde vêm os dados

A página é só mais um cliente da API, e usa as mesmas rotas que qualquer outro:

```text
GET  /schema      o contrato, a forma da rede e um cliente de exemplo
POST /risk-score  esse cliente, pontuado
```

Foi por isso que o `GET /schema` ganhou quatro campos: `thresholdStrategy` (o limiar sozinho não explica por que não é `0.5`), `model.units` (sem a topologia, não há como descrever o caminho em vez de supô-lo), `observedRange` (a faixa vista no treino, por coluna numérica) e `example` — um payload que **funciona**, com os campos recusados já removidos. Publicar um exemplo que o próprio serviço devolveria com `400` seria pior do que não publicar nenhum.

Nada disso é opcional para a página funcionar depois de um retreino que mudou as colunas: ela não tem um payload escrito à mão em lugar nenhum.

### O que ela é por dentro

Zero dependências e zero build, pelo mesmo motivo que a API não tem framework:

```text
web/
├── index.html
├── styles/tokens.css   cores, espaçamento, raio, sombra e tema claro/escuro
├── styles/app.css      layout
└── js/
    ├── api.js          fetch — devolve os DTOs como o serviço os escreve
    ├── domain.js       o vocabulário: `checkingStatus = 0` → "saldo negativo"
    ├── mappers.js      DTO → dados do componente (funções puras)
    ├── mock.js         temporário, só para `?mock=1`
    └── components/     custom elements
```

A separação que importa é `api → mappers → componente`: o componente recebe `inputs`, `layers`, `connections` e `results`, e **nunca** um corpo de resposta HTTP. Trocar a API é trocar o mapper.

As ligações são um SVG absoluto sobre o container, e nenhuma coordenada é escrita à mão: cada ponta procura o elemento com o `data-anchor` correspondente e mede onde ele está. Um `ResizeObserver` redesenha tudo quando qualquer coluna muda de tamanho — é o que faz o mesmo código produzir o fluxo horizontal no desktop e o vertical no mobile, sem um segundo layout.

Os pontos que percorrem as ligações somem inteiros sob `prefers-reduced-motion`; as 44 ligações continuam lá. E o SVG nunca é a única fonte da informação: as entradas são uma `<ul>`, as etapas são uma `<ol>` na ordem certa e os resultados trazem o número em texto. Se nada disso renderizar, a página continua dizendo a mesma coisa.

```bash
npm run serve            # a API e a página
open http://localhost:3000/?mock=1   # a página sem o serviço, para ver o layout
```

---

## 🧪 Como isso é testado sem treinar nada

A camada HTTP não sabe o que é um modelo. `createRoutes` recebe um `predict` pronto e um limiar, e nunca toca em tensor, `scaler` ou codificação — a mesma injeção que `predictRisk` já usava para o `toVector`.

O resultado é que a API inteira — os 200, os 400, o 413, o 500 — é testada com um pacote de mentira:

```javascript
const artifacts = {
  metadata: { source: 'synthetic', featureNames: [...], threshold: 0.3, savedAt: '…' },
  source: SYNTHETIC_SOURCE,
  predict: () => 0.9,
};
```

Nenhuma rede é treinada, o `npm test` continua rodando em segundos, e o teste do `500` verifica algo que só dá para verificar assim: que a mensagem de erro interna **não** aparece na resposta.

---

## 🔭 O que falta

A API responde, valida e não sofre *skew*. O que ela ainda não faz, e que um serviço de verdade faria:

- **autenticação** — qualquer um que alcance a porta pontua clientes;
- **rate limiting** — nada impede 10 mil requisições por segundo;
- **log estruturado das decisões** — sem registrar entrada, score e limiar de cada resposta, não há como auditar o modelo em produção nem detectar que a distribuição dos clientes mudou;
- **versionamento do endpoint** — trocar o modelo troca a resposta, e quem integra não tem como saber;
- **o próprio serviço avisar sobre entrada fora da faixa vista no treino** — a página já avisa, porque o `GET /schema` publica `observedRange`; a **resposta** do `POST /risk-score` ainda não. Um `creditAmount` de 500.000 DM sai de `[0, 1]` de propósito (é o certo para o laboratório) e volta com uma probabilidade de aparência perfeitamente normal para quem integra sem abrir o navegador.

O primeiro item da lista é o mais barato de fazer e o mais fácil de esquecer.
