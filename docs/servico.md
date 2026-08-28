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
  "savedAt": "2026-08-28T10:47:53.549Z",
  "source": "german",
  "encoding": "onehot",
  "featureNames": ["durationMonths", "creditAmount", "…", "foreignWorker=A202"],
  "scaler": {
    "featureNames": ["durationMonths", "creditAmount", "…"],
    "min":   { "durationMonths": 4,  "creditAmount": 250,   "age": 19 },
    "range": { "durationMonths": 68, "creditAmount": 18174, "age": 56 }
  },
  "threshold": 0.158696,
  "thresholdStrategy": "menor custo (FP=1, FN=5) em 160 clientes de calibração",
  "training": { "customers": 800, "fitCustomers": 640, "calibrationCustomers": 160,
                "balanced": false, "units": [16, 8], "l2": 0.003, "dropout": 0.2 }
}
```

Quatro coisas que os pesos sozinhos não dizem:

| Campo | Sem ele |
| ----- | ------- |
| `scaler` | O serviço normaliza diferente do treino. É o *skew*. |
| `featureNames` | A ordem das 57 entradas é o que liga cada valor ao peso certo. Trocar duas de lugar não dá erro nenhum. |
| `threshold` | `0.31` não significa nada sem o corte. O laboratório **escolhe** o corte pela matriz de custo, numa [fatia de calibração separada do teste](metricas.md#onde-o-corte-é-escolhido); servir com `0.5` jogaria fora essa escolha. |
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
  "riskProbability": 0.741019,
  "classification": "HIGH_RISK",
  "threshold": 0.158696,
  "model": { "source": "german", "features": 57, "savedAt": "2026-08-28T10:47:53.549Z" }
}
```

O limiar viaja na resposta porque ele é uma **escolha de negócio**, não uma propriedade do modelo: sem ele, `0.74` não diz se o cliente foi aprovado. E ele não é o `0.5` herdado — é o `0.158696` que a matriz de custo escolheu **numa fatia de calibração que o teste nunca viu**, gravado no pacote.

> A probabilidade e o limiar são arredondados na **mesma** casa (6). Arredondar só um dos dois produziria, na fronteira, um JSON que se contradiz: `riskProbability` igual ao `threshold` e classificação `LOW_RISK`.

### `GET /schema`

Descobrir o contrato batendo no `400` é um jeito ruim de integrar, então o serviço o publica:

```json
{
  "source": "german",
  "threshold": 0.158696,
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
- **o próprio serviço avisar sobre entrada fora da faixa vista no treino** — o `GET /schema` publica `observedRange`, então quem integra *pode* comparar; a **resposta** do `POST /risk-score` não diz nada. Um `creditAmount` de 500.000 DM sai de `[0, 1]` de propósito (é o certo para o laboratório) e volta com uma probabilidade de aparência perfeitamente normal.

O primeiro item da lista é o mais barato de fazer e o mais fácil de esquecer.
