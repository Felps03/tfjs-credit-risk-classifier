# 🧪 Dados: features, geração sintética e CSV

[⬅️ README](../README.md) · **Dados** · [German Credit](german-credit.md) · [Modelo](modelo.md) · [Métricas](metricas.md) · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · [Serviço](servico.md) · [API](api.md) · [Testes](testes.md)

---

## 📥 Features de entrada

O cliente é descrito por quatro campos brutos:

| Campo               | Descrição                           | Faixa gerada     |
| ------------------- | ----------------------------------- | ---------------- |
| `income`            | Renda mensal do cliente             | `2.000 – 15.000` |
| `debtRatio`         | Percentual de endividamento         | `0.0 – 1.0`      |
| `latePayments`      | Quantidade de pagamentos atrasados  | `0 – 5`          |
| `creditUtilization` | Percentual de utilização do crédito | `0.0 – 1.0`      |

Dois deles precisam ser **normalizados** antes de entrar na rede, para que todas as features fiquem na mesma escala (`0` a `1`) e nenhuma domine o gradiente só por ter números maiores:

```javascript
const normalizeIncome = (income) => (income - INCOME_MIN) / INCOME_RANGE;

const normalizeLatePayments = (latePayments) => latePayments / MAX_LATE_PAYMENTS;
```

`debtRatio` e `creditUtilization` já nascem entre `0` e `1` e vão direto.

O vetor que a rede realmente enxerga é montado por uma única função:

```javascript
const toFeatureVector = ({
  income,
  debtRatio,
  latePayments,
  creditUtilization,
}) => [
  normalizeIncome(income),
  debtRatio,
  normalizeLatePayments(latePayments),
  creditUtilization,
];
```

> 🔑 `toFeatureVector` é a **fonte única de verdade** da normalização: quem a chama é tanto a geração do dataset quanto a inferência do cliente novo. Duplicar essa lógica em dois lugares é a origem clássica do *training-serving skew* — o modelo passa a receber, em produção, números em escala diferente da que viu no treino.

> 💡 Aqui as constantes de normalização (`2000`, `13000`, `5`) são conhecidas de antemão porque nós geramos os dados. **Em um projeto real elas devem ser calculadas apenas no conjunto de treino** e depois reaplicadas em validação/teste — caso contrário há vazamento de dados (*data leakage*).
>
> É exatamente o que o [dataset real](german-credit.md#agora-a-normalização-precisa-ser-medida) obrigou a fazer: lá as faixas são **medidas**, e só sobre o treino.

As features acima descrevem o dataset sintético. O dataset real tem [as suas próprias dezenove](german-credit.md#as-19-colunas-usadas) — e o pipeline atende as duas sem saber qual está em uso.

---

## 🧪 Geração dos dados sintéticos

O projeto cria **1.200 clientes** com características aleatórias, e uma regra determinística define o rótulo de cada um. Só que gerar dados *perfeitos* ensina pouco: um dataset limpo e equilibrado faz qualquer modelo parecer excelente e esconde justamente os dois problemas que aparecem em quase todo projeto real.

Então o gerador cria os dois de propósito, cada um com um botão próprio:

| Parâmetro | Padrão | O que simula |
| --------- | -----: | ------------ |
| `positiveRate` | `0.15` | **Desbalanceamento** — inadimplente é minoria |
| `featureNoise` | `0.05` | **Ruído de medição** — o dado observado não é o dado real |
| `labelNoise` | `0.02` | **Ruído de rótulo** — o desfecho registrado está errado |

São argumentos, não constantes escondidas. Dá para desligar um de cada vez e ver o efeito isolado — é exatamente o que a [tabela do 2×2](#o-que-cada-botão-fez-com-as-métricas) mais abaixo faz.

### A verdade que ninguém observa

A mudança estrutural está aqui: o gerador passou a distinguir **o cliente verdadeiro** do **cliente medido**.

```javascript
// 1. O estado VERDADEIRO do cliente. No mundo real ele existe e ninguém
//    o enxerga; aqui ele existe, é usado para rotular, e é descartado.
const truths = Array.from({ length: total }, () => ({
  income: INCOME_MIN + random() * INCOME_RANGE,
  debtRatio: random(),
  latePayments: Math.floor(random() * (MAX_LATE_PAYMENTS + 1)),
  creditUtilization: random(),
}));

// 2. O corte que produz a taxa de inadimplência pedida.
const scores = truths.map(riskScore);
const cut = quantile(scores, 1 - positiveRate);

// 3. O que vai para o arquivo é a MEDIDA e o desfecho REGISTRADO.
return truths.map((truth, index) => {
  const label = scores[index] > cut ? 1 : 0;
  const mistaken = random() < labelNoise;

  return {
    ...measureCustomer(truth, featureNoise, gaussian),
    risk: mistaken ? 1 - label : label,
  };
});
```

O rótulo é calculado sobre `truth`, e o que chega ao CSV é `measureCustomer(truth, ...)`. **A resposta certa depende de um valor que nunca entra no arquivo.** É isso que cria um teto: nem a fórmula que gerou os rótulos consegue reconstruí-los a partir do que o modelo vê.

A regra em si não mudou — e a rede continua sem vê-la:

```javascript
const riskScore = (customer) => {
  const [income, debtRatio, latePayments, creditUtilization] =
    toFeatureVector(customer);

  return (
    1.4 * debtRatio +
    1.2 * latePayments +
    1.0 * creditUtilization -
    0.8 * income
  );
};
```

### Desbalanceamento: o limiar virou um quantil

Antes o corte era a constante `RISK_RULE_THRESHOLD = 1.35`, escolhida na mão, e a taxa de inadimplência era o que desse — cerca de 46%. Um número mágico cujo efeito só aparecia rodando.

Agora o corte é **derivado da taxa que se pede**:

```javascript
const quantile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp(Math.floor(fraction * sorted.length), 0, sorted.length - 1);

  return sorted[index];
};

const cut = quantile(scores, 1 - positiveRate);   // 0.15 → percentil 85
```

`positiveRate: 0.15` produz 15% de inadimplentes porque *é assim que quantil funciona*, não porque alguém calibrou um número até a conta fechar. Desbalancear mais é só empurrar a fração para cima.

Com 15% de positivos, o dataset ganha a propriedade que faltava: **a acurácia deixa de significar alguma coisa sozinha**. Chutar "bom pagador" para todo mundo já acerta `0.8423` no conjunto de teste. É por isso que o projeto imprime o [baseline da classe majoritária](metricas.md#-matriz-de-confusão) ao lado da acurácia — sem ele, `0.9563` parece ótimo, e são só 11 pontos de ganho sobre não fazer nada.

### Ruído de medição: um teto que nenhum modelo ultrapassa

Cada coluna recebe um desvio normal proporcional à sua faixa e volta para dentro dos limites válidos:

```javascript
const measureCustomer = (customer, noise, gaussian) => {
  const measured = Object.fromEntries(
    Object.entries(SYNTHETIC_BOUNDS).map(([column, [lowest, highest]]) => {
      const drift = gaussian() * noise * (highest - lowest);

      return [column, clamp(customer[column] + drift, lowest, highest)];
    }),
  );

  return { ...measured, latePayments: Math.round(measured.latePayments) };
};
```

Três decisões pequenas que importam:

- **A normal vem de Box-Muller**, não de `Math.random()`. Ruído de medição costuma ser a soma de muitos erros pequenos e independentes, e o Teorema Central do Limite diz que essa soma tende à normal.
- **O `clamp` não é detalhe**: sem ele apareceriam renda negativa e utilização de crédito de 130%, e a normalização passaria a devolver valores fora de `[0, 1]`.
- **`latePayments` é arredondado** — "2,3 atrasos" não existe no sistema de nenhum banco.

O efeito é medível sem treinar nada. Aplicando a **própria regra que gerou os rótulos** aos dados medidos, ela erra:

| Dataset | A regra recupera o rótulo |
| ------- | ------------------------: |
| Sem ruído nenhum | `100,0%` |
| Só `featureNoise: 0.05` | `98,0%` |
| Medição + rótulo (o CSV atual) | `96,8%` |

Esses 2 pontos da segunda linha são **irredutíveis**. Não existe arquitetura, otimizador ou quantidade de épocas que os recupere, porque a informação não está no arquivo. É o análogo controlado do que o [German Credit](german-credit.md#-dataset-real-german-credit) mostra sem pedir licença: o teto raramente é o modelo.

Dois testes guardam exatamente essa afirmação — um exige teto `< 100%` com ruído, o outro exige teto `= 100%` sem ele.

### Ruído de rótulo: a resposta certa também erra

`labelNoise: 0.02` troca 2% dos desfechos. Um bom pagador que ficou desempregado, uma baixa lançada na conta errada, um `1` digitado onde era `0`.

Em um dataset desbalanceado isso tem um efeito assimétrico que vale notar. Como 85% das linhas são negativas, trocar 2% delas gera *muito* mais positivos falsos do que se perde de positivos verdadeiros — no arquivo versionado, **12 rótulos viraram `1` e só 2 viraram `0`**. (São 14 trocas em 1.200 linhas, e não as 24 que 2% sugerem: cada linha é um sorteio independente, e esta semente calhou de dar poucas. Em 40 sementes a taxa média é `0.0196`.) A taxa de inadimplência sobe de `14,9%` para `15,8%`, e cerca de **um em cada dezesseis positivos do arquivo é puro ruído**.

Rótulo minoritário é caro justamente por isso: a classe rara é a mais fácil de contaminar, porque a classe abundante é grande o bastante para inundá-la com os próprios erros.

### O que cada botão fez com as métricas

Cada variante foi escrita em CSV e lida de volta, e treinada com a **configuração exata de `npm start`** — mesmas épocas, mesmo *early stopping*, mesma [divisão estratificada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado). Média de **15 execuções**, com dataset e divisão fixos e só a inicialização dos pesos variando:

| Dataset | Baseline | Acurácia | Ganho | AUC | Recall `0.5` | Custo `0.5` | Custo mín. |
| ------- | -------: | -------: | ----: | --: | -----------: | ----------: | ---------: |
| limpo + equilibrado *(o de antes)* | `0.5021` | `0.9881` ± 0.0016 | **+48,6 pts** | `0.9997` ± 0.0001 | `0.9900` | `7.7` | `1.5` |
| limpo + desbalanceado | `0.8506` | `0.9690` ± 0.0031 | **+11,8 pts** | `0.9960` ± 0.0005 | `0.8000` | `36.3` | `9.2` |
| ruidoso + equilibrado | `0.5042` | `0.9572` ± 0.0014 | **+45,3 pts** | `0.9938` ± 0.0002 | `0.9501` | `34.0` | `12.7` |
| **ruidoso + desbalanceado** *(atual)* | `0.8423` | `0.9563` ± 0.0032 | **+11,4 pts** | `0.9739` ± 0.0012 | `0.7228` | `52.7` | `19.1` |

Cinco leituras que só aparecem por causa do 2×2:

**1. A acurácia ficou igual e o modelo piorou muito.** Compare as duas linhas ruidosas: ao desbalancear, a acurácia vai de `0.9572` para `0.9563` — a mesma, dentro do erro padrão. No mesmo passo a AUC cai de `0.9938` para `0.9739` e o recall despenca de `0.9501` para `0.7228`. **O placar não se mexeu e o modelo ficou nitidamente pior.** Não há paradoxo: desbalancear aumenta a fatia de negativos, e negativos são fáceis — o que a rede perdeu em positivos, ela recuperou em linhas triviais. Este é o argumento inteiro contra reportar acurácia sozinha, em duas linhas de tabela.

**2. A AUC é quase indiferente à proporção; a acurácia é refém dela.** Olhe a coluna do ganho: `+48,6`, `+11,8`, `+45,3`, `+11,4` — ela salta 37 pontos só porque a raridade dos positivos mudou, sem que nada do modelo mude junto. A AUC, no mesmo 2×2, anda de `0.9997` a `0.9739`. É consequência da definição: ela compara pares positivo–negativo, então a *proporção* entre as classes sai da conta. Entre os dois botões, o ruído é o que ela sente mais — desbalancear sozinho custa `0.0037` de AUC, ruído sozinho custa `0.0059` —, mas os dois efeitos são pequenos perto do que fazem com a acurácia.

**3. O recall desaba e a precision sobe.** `0.9900` → `0.7228` de recall, contra `0.9862` → `1.0000` de precision. Com positivos raros, a rede aprende que apostar em "alto risco" quase nunca compensa, e passa a só marcar quando tem muita certeza. Fica **cautelosa** — e cautela, num modelo de crédito, é deixar passar **mais de um em cada quatro** inadimplentes. A precision perfeita não é virtude: é o sintoma. Um modelo que nunca erra ao acusar é um modelo que quase não acusa.

**4. É por isso que o limiar precisa ser escolhido.** O corte de menor custo cai de `0.4807` para `0.2517`: o desbalanceamento empurra as probabilidades todas para baixo, e o `0.5` herdado deixa de ser um lugar razoável para cortar. Ajustá-lo derruba o custo de `52.7` para `19.1` — **64% a menos**, sem treinar nada de novo. No dataset limpo e equilibrado, a mesma manobra poupava `6.2` unidades de custo; aqui poupa `33.6`.

**5. E o erro padrão dobrou.** Repare nas duas linhas desbalanceadas: `± 0.0031` e `± 0.0032` de acurácia, contra `± 0.0016` e `± 0.0014` nas equilibradas. O mesmo gerador, a mesma arquitetura — só a raridade dos positivos mudou. Com 38 positivos no conjunto de teste, cada acerto ou erro move o recall em 2,6 pontos, e o *early stopping* passa a interromper o treino em épocas diferentes a cada execução. **Desbalancear não piora só o modelo: piora a sua capacidade de medi-lo.**

O dataset sintético continua sendo mais fácil que o real, e deve mesmo — ele existe para que o pipeline seja verificável. Mas parou de ser fácil de graça.

---

## 📄 Carregando dados de um CSV

Gerar o dataset em memória é conveniente, mas esconde a etapa que todo projeto real tem: **os dados chegam de um arquivo**. O pipeline agora passa por disco — o CSV é escrito uma vez e, a partir dele, tudo é lido.

```javascript
ensureCsv();                              // cria só se ainda não existir
const dataset = await loadDatasetCsv();   // daqui em diante, vem do arquivo
```

### O CSV guarda dados BRUTOS

Esta é a decisão que sustenta a seção inteira:

```text
income,debtRatio,latePayments,creditUtilization,risk
11465.21,0.764368,0,0.941555,1
4230.33,0.640292,1,0.280455,0
```

Renda em reais, atrasos em contagem — **não** os valores normalizados. Salvar já normalizado congelaria `INCOME_MIN` e `INCOME_RANGE` dentro do arquivo: qualquer ajuste na normalização exigiria reexportar tudo, e um CSV antigo lido com constantes novas produziria features silenciosamente erradas.

Com dados brutos, o arquivo é a **fonte**, e a normalização continua sendo do código.

### A refatoração que isso exigiu

Antes, `createDataset` gerava clientes brutos e os descartava, devolvendo só as features normalizadas. Para exportar, foi preciso separar as duas responsabilidades:

| Função | Devolve |
| ------ | ------- |
| `createCustomers` | Clientes em unidades brutas — `{ income, debtRatio, latePayments, creditUtilization, risk }` |
| `toDataset` | Clientes → `{ features, labels }` prontos para o TensorFlow |
| `createDataset` | `toDataset(createCustomers(total))` — a API antiga, intacta |

O ganho: **um só caminho de normalização**, seja o dado gerado em memória ou lido do arquivo.

```javascript
const loadDatasetCsv = async (filePath = CSV_PATH) =>
  toDataset(await readCustomersCsv(filePath));
```

`loadDatasetCsv` devolve exatamente o mesmo formato de `createDataset` — `splitDataset` e o treino funcionam sem qualquer adaptação.

### Lendo com tf.data.csv

O `tfjs-node` registra o esquema `file://` também aqui:

```javascript
const dataset = tf.data.csv(`file://${path.resolve(filePath)}`, {
  columnConfigs: { risk: { isLabel: true } },
});

const rows = await dataset.toArray();
```

Dois comportamentos úteis: o parse numérico já vem pronto (nada de `parseFloat`), e as colunas são casadas **pelo nome**, não pela posição — um CSV com as colunas em outra ordem carrega igual. Há teste para isso.

### Precisão é uma escolha

CSV é texto, então cada coluna grava um número fixo de casas:

```javascript
const CSV_PRECISION = {
  income: 2,          // centavos bastam para renda
  debtRatio: 6,
  latePayments: 0,    // contagem, não fração
  creditUtilization: 6,
  risk: 0,
};
```

Isso mantém o arquivo legível e evita `3.0000000000000004` em uma coluna de contagem. Em troca, a ida e volta **não é bit a bit** — é por isso que o teste de round-trip compara com tolerância explícita em vez de igualdade, ao contrário do teste de [persistência do modelo](inferencia.md#-persistência-do-modelo), onde os pesos voltam idênticos.

### O arquivo é versionado

`data/customers.csv` **está no repositório** (36 KB). É o que dá ao laboratório algo que ele não tinha: um dataset **estável**.

Antes, cada execução sorteava 1200 clientes novos, então os números desta documentação nunca batiam com os da sua tela. Agora o dataset é o mesmo — o que varia entre execuções é só a inicialização dos pesos.

Para isso funcionar, a geração precisa ser **idempotente**:

```javascript
const ensureCsv = (filePath = CSV_PATH, total = SYNTHETIC_TOTAL) => {
  if (fs.existsSync(filePath)) {
    return { path: filePath, created: false };
  }

  writeCustomersCsv(createCustomers(total), filePath);

  return { path: filePath, created: true };
};
```

Sem esse `existsSync`, versionar o CSV seria um incômodo: todo `npm start` reescreveria o arquivo e deixaria 1200 linhas de diff aleatório no `git status`.

| Comando | Efeito no CSV |
| ------- | ------------- |
| `npm start` | Usa o que está lá; só cria se o arquivo faltar |
| `npm run seed` | **Regenera** a partir do código — mudança deliberada, que você commita se quiser |

E como o gerador tem semente, `npm run seed` é **reprodutível**: rodar duas vezes produz o mesmo arquivo, byte a byte. Isso transforma o CSV versionado em algo auditável — quem clona reconstrói o arquivo a partir do código e confere que ninguém o editou à mão. Um teste faz exatamente essa comparação e falha se o gerador mudar sem que o CSV seja regerado.

Para experimentar sem tocar no código, cada parâmetro do gerador é uma flag:

```bash
npm run seed -- --seed=99                        # outros clientes
npm run seed -- --feature-noise=0 --label-noise=0  # dataset limpo
npm run seed -- --positive-rate=0.5              # dataset equilibrado
```

Um dataset real — que você recebe em vez de gerar — entraria exatamente aqui, e o `createCustomers` sairia do caminho.

---

[⬅️ Voltar ao README](../README.md)
