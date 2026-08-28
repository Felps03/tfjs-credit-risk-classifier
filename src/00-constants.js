const path = require('node:path');

// --------------------------------------------------
// 0. Constantes do laboratório
// --------------------------------------------------
// Faixas conhecidas de antemão porque nós geramos os dados.
// Em um projeto real, estas estatísticas seriam calculadas
// APENAS sobre o conjunto de treino.
const INCOME_MIN = 2000;
const INCOME_RANGE = 13000;
const MAX_LATE_PAYMENTS = 5;

// Configuração do gerador sintético. Um dataset de brinquedo limpo e
// balanceado ensina o pipeline, mas esconde os dois problemas que
// aparecem em quase todo projeto real — e que só se enfrenta medindo:
//
//   ruído          o banco não observa a verdade, observa uma MEDIDA dela.
//                  Renda declarada, atraso registrado com erro, utilização
//                  medida no dia errado. Isso cria um teto: acima dele
//                  nenhum modelo chega, porque a informação não existe.
//   desbalanceamento  inadimplente é minoria. Quando 85% dos clientes são
//                  bons, "chutar bom para todo mundo" já acerta 85% — e a
//                  acurácia deixa de significar qualquer coisa.
const SYNTHETIC_SEED = 7;
const SYNTHETIC_TOTAL = 1200;

// Fração de inadimplentes ALVO. O limiar da regra deixa de ser um número
// fixo e passa a ser o quantil que produz exatamente esta taxa.
const SYNTHETIC_POSITIVE_RATE = 0.15;

// Desvio padrão do ruído de medição, como fração da faixa de cada coluna.
// 0.05 sobre a renda = R$ 650 de erro típico.
const SYNTHETIC_FEATURE_NOISE = 0.05;

// Fração de rótulos trocados: o desfecho registrado está errado. Um bom
// pagador que ficou desempregado, uma baixa lançada na conta errada.
const SYNTHETIC_LABEL_NOISE = 0.02;

// Limiar de decisão do classificador: escolha de negócio, não do modelo.
const DECISION_THRESHOLD = 0.5;

// Regularização: os dois freios contra decorar o treino.
//
// O modelo real tem 1.073 parâmetros para 640 linhas de treino efetivo.
// Com mais parâmetros do que dados, decorar é o caminho mais barato para
// baixar o erro — e a conta chega no teste, não no treino. Os dois freios
// atacam o mesmo problema por caminhos diferentes:
//
//   L2       soma λ·Σw² à loss do treino. Peso grande passa a custar caro,
//            então a rede só paga por um se ele render redução de erro
//            suficiente. Achata os pesos; não zera nenhum.
//   dropout  desliga uma fração das unidades a cada passo do treino.
//            Nenhuma unidade pode contar com uma vizinha específica, então
//            a informação precisa ficar distribuída em vez de concentrada.
//
// Os valores não vieram de convenção: saíram de uma varredura de 6 × 5
// combinações sobre 15 divisões, medindo a diferença treino−teste. O
// README registra a grade inteira.
const L2_LAMBDA = 0.003;
const DROPOUT_RATE = 0.2;

// A arquitetura padrão: duas camadas ocultas, afunilando.
//
// Ela ficou onde estava por uma razão medida, não por convenção. Oito
// topologias foram comparadas por validação cruzada, da regressão
// logística sem camada oculta nenhuma até uma rede de 15.745
// parâmetros: entre as SETE redes, nenhuma se distingue de nenhuma. A
// documentação traz a tabela.
//
// Trocar o padrão por outra dessas sete seria trocar por ruído — que é
// exatamente o erro contra o qual o resto do projeto argumenta.
const HIDDEN_UNITS = [16, 8];

// Pasta onde o modelo treinado é persistido (ignorada pelo git).
const MODEL_DIR = path.join(__dirname, '..', 'model');

// CSV com os dados brutos. Em um projeto real ele viria de fora
// (export de um banco, entrega de um parceiro); aqui é gerado para o
// laboratório continuar auto-contido.
const CSV_PATH = path.join(__dirname, '..', 'data', 'customers.csv');
const CSV_LABEL_COLUMN = 'risk';
const CSV_COLUMNS = [
  'income',
  'debtRatio',
  'latePayments',
  'creditUtilization',
  CSV_LABEL_COLUMN,
];

// Casas decimais por coluna. CSV é texto: a precisão é uma ESCOLHA, e
// arredondar aqui é o que um export de banco de dados faria.
const CSV_PRECISION = {
  income: 2,
  debtRatio: 6,
  latePayments: 0,
  creditUtilization: 6,
  risk: 0,
};

// --------------------------------------------------
// Dataset REAL: German Credit (UCI / Statlog)
// --------------------------------------------------
// 1000 solicitações de crédito reais, coletadas por Hans Hofmann na
// Universidade de Hamburgo e publicadas em 1994. É o contraponto do
// dataset sintético: aqui ninguém escolheu a regra que separa bom de mau
// pagador — ela precisa ser descoberta, e boa parte dela simplesmente
// não está nas colunas.

const GERMAN_CSV_PATH = path.join(__dirname, '..', 'data', 'german-credit.csv');
const GERMAN_SOURCE_URL = 'https://archive.ics.uci.edu/static/public/144/data.csv';

// Colunas com magnitude de verdade: um valor maior significa "mais".
// Só estas passam pelo min-max — as outras não têm o que escalar.
const GERMAN_NUMERIC = [
  'durationMonths',
  'creditAmount',
  'installmentRate',
  'residenceSince',
  'age',
  'existingCredits',
  'dependents',
];

// Colunas qualitativas, com os códigos na ordem documentada pela UCI.
//
// ATENÇÃO ao que o inteiro significa daqui em diante: ele é um CÓDIGO,
// não uma quantidade. `purpose = 3` não é o triplo de `purpose = 1`; são
// "rádio/TV" e "carro usado". É exatamente por isso que estas colunas
// viram one-hot antes de entrar na rede.
const GERMAN_CATEGORICAL = {
  checkingStatus: ['A11', 'A12', 'A13', 'A14'],
  creditHistory: ['A30', 'A31', 'A32', 'A33', 'A34'],
  purpose: ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
  savingsStatus: ['A61', 'A62', 'A63', 'A64', 'A65'],
  employmentYears: ['A71', 'A72', 'A73', 'A74', 'A75'],
  otherDebtors: ['A101', 'A102', 'A103'],
  property: ['A121', 'A122', 'A123', 'A124'],
  otherInstallments: ['A141', 'A142', 'A143'],
  housing: ['A151', 'A152', 'A153'],
  job: ['A171', 'A172', 'A173', 'A174'],
  telephone: ['A191', 'A192'],
  foreignWorker: ['A201', 'A202'],
};

// Atributo 9 do arquivo original: estado civil e SEXO.
//
// Ele é lido e vai para o CSV, mas NÃO entra no modelo. Usar sexo para
// negar crédito é discriminação e é ilegal em vários países. A coluna
// serve a um propósito diferente: AUDITAR as decisões do modelo depois
// que ele já decidiu sem ela. Veja `auditByGroup`.
//
// A95 (mulher solteira) não aparece nos 1000 registros; só A91–A94.
const GERMAN_AUDIT_COLUMN = 'personalStatus';
const GERMAN_AUDIT_CODES = ['A91', 'A92', 'A93', 'A94'];

// A92 é o único código feminino presente no arquivo. Os outros três são
// recortes de estado civil masculino — ou seja, dá para recuperar o sexo
// desta coluna, e é isso que a auditoria usa.
const FEMALE_CODE = 'A92';

// Uma linha do CSV: as numéricas, os códigos das qualitativas, a coluna
// de auditoria e o rótulo. 21 colunas — 19 delas viram features.
const GERMAN_COLUMNS = [
  ...GERMAN_NUMERIC,
  ...Object.keys(GERMAN_CATEGORICAL),
  GERMAN_AUDIT_COLUMN,
  CSV_LABEL_COLUMN,
];

// Todos os valores gravados são inteiros: meses, marcos, contagens,
// idades e códigos de categoria.
const GERMAN_PRECISION = Object.fromEntries(
  GERMAN_COLUMNS.map((column) => [column, 0]),
);

// --------------------------------------------------
// Serviço HTTP
// --------------------------------------------------
// Porta padrão do `npm run serve`, e o teto do corpo da requisição.
// Um cliente do German Credit tem 19 campos: 16 KB é folga de sobra e
// ainda assim recusa um POST de 50 MB antes de ele virar memória.
const API_PORT = 3000;
const API_BODY_LIMIT = 16 * 1024;

// Semente do embaralhamento. Fixa de propósito: sem ela cada execução
// mediria um recorte diferente do dataset e nenhum número deste projeto
// se reproduziria. Trocar a semente é trocar o experimento.
const SHUFFLE_SEED = 42;

module.exports = {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  SYNTHETIC_SEED,
  SYNTHETIC_TOTAL,
  SYNTHETIC_POSITIVE_RATE,
  SYNTHETIC_FEATURE_NOISE,
  SYNTHETIC_LABEL_NOISE,
  DECISION_THRESHOLD,
  L2_LAMBDA,
  DROPOUT_RATE,
  HIDDEN_UNITS,
  MODEL_DIR,
  CSV_PATH,
  CSV_LABEL_COLUMN,
  CSV_COLUMNS,
  CSV_PRECISION,
  GERMAN_CSV_PATH,
  GERMAN_SOURCE_URL,
  GERMAN_NUMERIC,
  GERMAN_CATEGORICAL,
  GERMAN_AUDIT_COLUMN,
  GERMAN_AUDIT_CODES,
  FEMALE_CODE,
  GERMAN_COLUMNS,
  GERMAN_PRECISION,
  SHUFFLE_SEED,
  API_PORT,
  API_BODY_LIMIT,
};
