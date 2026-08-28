// --------------------------------------------------
// MOCK — temporário, e só para ver a tela sem o serviço
// --------------------------------------------------
// Isto NÃO é usado quando a API responde. Ele existe por duas razões:
// abrir a página com `?mock=1` sem ter treinado nada, e servir de
// referência da forma exata dos DTOs.
//
// A forma é a parte que importa: os dois objetos abaixo são cópias
// literais do que `GET /schema` e `POST /risk-score` devolvem — mesmos
// nomes, mesmos tipos, mesma ordem. É por isso que `toFlowData` não sabe
// dizer se veio daqui ou da rede, e é por isso que trocar o mock pela
// API é uma linha em `app.js` e nada além.
//
// Para remover: apague este arquivo e o ramo `?mock=1` de `app.js`.

const SCHEMA = {
  source: 'german',
  label: 'German Credit — UCI/Statlog (Hofmann, 1994), one-hot',
  encoding: 'onehot',
  threshold: 0.109897,
  thresholdStrategy: 'menor custo (FP=1, FN=5)',
  model: {
    features: 57,
    units: [16, 8],
    savedAt: '2026-08-28T02:37:23.428Z',
  },
  request: {
    numeric: [
      'durationMonths', 'creditAmount', 'installmentRate',
      'residenceSince', 'age', 'existingCredits', 'dependents',
    ],
    categorical: {
      checkingStatus: { range: [0, 3], codes: ['A11', 'A12', 'A13', 'A14'] },
      creditHistory: { range: [0, 4], codes: ['A30', 'A31', 'A32', 'A33', 'A34'] },
      purpose: {
        range: [0, 9],
        codes: ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
      },
      savingsStatus: { range: [0, 4], codes: ['A61', 'A62', 'A63', 'A64', 'A65'] },
      employmentYears: { range: [0, 4], codes: ['A71', 'A72', 'A73', 'A74', 'A75'] },
      otherDebtors: { range: [0, 2], codes: ['A101', 'A102', 'A103'] },
      property: { range: [0, 3], codes: ['A121', 'A122', 'A123', 'A124'] },
      otherInstallments: { range: [0, 2], codes: ['A141', 'A142', 'A143'] },
      housing: { range: [0, 2], codes: ['A151', 'A152', 'A153'] },
      job: { range: [0, 3], codes: ['A171', 'A172', 'A173', 'A174'] },
      telephone: { range: [0, 1], codes: ['A191', 'A192'] },
      foreignWorker: { range: [0, 1], codes: ['A201', 'A202'] },
    },
    rejected: ['personalStatus'],
  },
  example: {
    durationMonths: 48,
    creditAmount: 9000,
    installmentRate: 4,
    residenceSince: 2,
    age: 24,
    existingCredits: 2,
    dependents: 1,
    checkingStatus: 0,
    creditHistory: 1,
    purpose: 0,
    savingsStatus: 0,
    employmentYears: 1,
    otherDebtors: 0,
    property: 3,
    otherInstallments: 0,
    housing: 0,
    job: 1,
    telephone: 0,
    foreignWorker: 0,
  },
};

const SCORE = {
  riskProbability: 0.781204,
  classification: 'HIGH_RISK',
  threshold: 0.109897,
  model: { source: 'german', features: 57, savedAt: SCHEMA.model.savedAt },
};

// O atraso é de propósito: sem ele o estado de carregamento some antes de
// aparecer, e um esqueleto que ninguém vê não foi testado por ninguém.
export const fetchAnalysisMock = async () => {
  await new Promise((resolve) => { setTimeout(resolve, 550); });

  return { schema: SCHEMA, score: SCORE, customer: SCHEMA.example };
};

// --------------------------------------------------
// Pontuação de mentira, para o formulário reagir sem serviço
// --------------------------------------------------
// ISTO NÃO É O MODELO. É uma soma ponderada com meia dúzia de pesos
// escritos à mão, cuja única função é fazer as barras se moverem quando
// alguém mexe num campo com `?mock=1`. Os números que ela produz não
// significam nada — a rede de verdade tem 1.073 parâmetros e está em
// `model/weights.bin`.
//
// Os pesos seguem a direção intuitiva de cada coluna (prazo longo piora,
// poupança alta melhora) só para o comportamento não ser absurdo na tela.
// A direção estar certa é coincidência de bom senso, não medição.
const PESOS_FALSOS = {
  durationMonths: 0.030,
  creditAmount: 0.00006,
  installmentRate: 0.25,
  age: -0.030,
  checkingStatus: -0.55,
  savingsStatus: -0.22,
  employmentYears: -0.16,
  creditHistory: -0.14,
  otherInstallments: -0.20,
  housing: -0.18,
  existingCredits: 0.10,
  dependents: 0.12,
  residenceSince: -0.03,
};

const sigmoide = (x) => 1 / (1 + Math.exp(-x));

export const scoreMock = async (customer) => {
  await new Promise((resolve) => { setTimeout(resolve, 120); });

  const soma = Object.entries(PESOS_FALSOS).reduce(
    (total, [campo, peso]) => total + peso * (customer[campo] ?? 0),
    -1.1,
  );
  const riskProbability = Number(sigmoide(soma).toFixed(6));

  return {
    riskProbability,
    classification: riskProbability >= SCHEMA.threshold ? 'HIGH_RISK' : 'LOW_RISK',
    threshold: SCHEMA.threshold,
    model: SCORE.model,
  };
};
