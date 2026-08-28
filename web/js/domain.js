// --------------------------------------------------
// Vocabulário do domínio
// --------------------------------------------------
// O serviço fala `durationMonths` e `checkingStatus = 0`; uma pessoa não.
// Este módulo é a única camada que sabe traduzir um do outro, e é de
// propósito que ele seja só DADO: nenhum componente monta texto de campo,
// e trocar o dataset é trocar este dicionário — não a visualização.
//
// Os códigos e o que cada um significa vêm da documentação original do
// German Credit (Hofmann, 1994, UCI/Statlog). O valor monetário está em
// DM (marco alemão), porque é a moeda em que os dados foram coletados —
// exibir "R$" seria converter uma unidade que ninguém converteu.

// Ícones em traço, 24×24, para herdar `currentColor` e a espessura do
// resto da interface. Cada entrada é uma lista de `d`, porque alguns
// desenhos precisam de mais de um traço.
export const ICONS = {
  calendar: ['M8 2v4M16 2v4M3 10h18', 'M5 4h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z'],
  money: ['M2 7h20v10H2z', 'M12 9a3 3 0 1 1 0 6 3 3 0 0 1 0-6z', 'M5 10v.01M19 14v.01'],
  gauge: ['M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z', 'M13.4 10.6 19 5', 'M3.5 18a9 9 0 1 1 17 0'],
  pin: ['M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11z', 'M12 10a2.5 2.5 0 1 0 0 .01z'],
  user: ['M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2', 'M12 3a4 4 0 1 1 0 8 4 4 0 0 1 0-8z'],
  users: ['M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2', 'M9 3a4 4 0 1 1 0 8 4 4 0 0 1 0-8z', 'M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75'],
  layers: ['m12 2 9 5-9 5-9-5z', 'M3 12l9 5 9-5', 'M3 17l9 5 9-5'],
  bank: ['M3 21h18', 'M4 10h16', 'm12 3 9 5H3z', 'M6 10v11M10 10v11M14 10v11M18 10v11'],
  clock: ['M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18z', 'M12 7v5l3 2'],
  tag: ['M7.5 7.5h.01', 'M20.6 12.6 12 21.2a2 2 0 0 1-2.8 0l-6.4-6.4a2 2 0 0 1-.6-1.4V4a1 1 0 0 1 1-1h9.4a2 2 0 0 1 1.4.6l6.6 6.6a2 2 0 0 1 0 2.8z'],
  wallet: ['M3 7a2 2 0 0 1 2-2h12v2', 'M3 7h16a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z', 'M17 13h.01'],
  briefcase: ['M3 8h18v12H3z', 'M8 8V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v3', 'M3 13h18'],
  handshake: ['M12 6 9 9l3 3 3-3z', 'M2 12l4-4 6 6', 'M22 12l-4-4-4 4', 'M6 16l3 3'],
  key: ['M12.6 11.4a5 5 0 1 0-7.2 7.2 5 5 0 0 0 7.2-7.2z', 'M12.6 11.4 21 3', 'M17 7l2.5 2.5'],
  card: ['M2 6h20v12H2z', 'M2 10h20', 'M6 15h4'],
  home: ['m3 10 9-7 9 7v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z', 'M9 22V12h6v10'],
  phone: ['M8 2h8a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z', 'M12 18h.01'],
  globe: ['M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18z', 'M3 12h18', 'M12 3a15 15 0 0 1 0 18 15 15 0 0 1 0-18'],
  shield: ['M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z'],
  field: ['M4 6h16M4 12h16M4 18h10'],
  alert: ['M12 3 2 20h20z', 'M12 9v5', 'M12 17.5v.01'],
  check: ['M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18z', 'm8.5 12 2.5 2.5 4.5-5'],
};

// Formatadores das colunas numéricas. Cada um recebe o valor BRUTO — o
// mesmo que vai no POST — e devolve o texto. Nenhum deles arredonda o
// que será enviado: formatar é uma decisão de exibição, e ela não pode
// vazar para o payload.
const inteiro = new Intl.NumberFormat('pt-BR');

const FORMATTERS = {
  durationMonths: (v) => `${v} meses`,
  creditAmount: (v) => `${inteiro.format(v)} DM`,

  // 1 a 4 não são reais nem por cento: são faixas ordenadas de
  // comprometimento da renda, e escrever "4" sozinho sugeriria o
  // contrário.
  installmentRate: (v) => `faixa ${v} de 4`,
  residenceSince: (v) => `faixa ${v} de 4`,
  age: (v) => `${v} anos`,
  existingCredits: (v) => `${v} ${v === 1 ? 'crédito' : 'créditos'}`,
  dependents: (v) => `${v} ${v === 1 ? 'dependente' : 'dependentes'}`,
};

// Nome, ícone e — nas qualitativas — o que cada código quer dizer, na
// MESMA ordem da lista publicada em `GET /schema`. A ordem importa: o
// serviço recebe o ÍNDICE, então uma linha fora de lugar aqui renomeia
// silenciosamente a categoria na tela.
export const FIELDS = {
  durationMonths: { label: 'Prazo do contrato', icon: 'calendar', group: 'numeric' },
  creditAmount: { label: 'Valor do crédito', icon: 'money', group: 'numeric' },
  installmentRate: { label: 'Comprometimento da renda', icon: 'gauge', group: 'numeric' },
  residenceSince: { label: 'Tempo de residência', icon: 'pin', group: 'numeric' },
  age: { label: 'Idade', icon: 'user', group: 'numeric' },
  existingCredits: { label: 'Créditos neste banco', icon: 'layers', group: 'numeric' },
  dependents: { label: 'Dependentes', icon: 'users', group: 'numeric' },

  checkingStatus: {
    label: 'Conta corrente',
    icon: 'bank',
    group: 'categorical',
    codes: ['saldo negativo', 'até 200 DM', '200 DM ou mais', 'não possui'],
  },
  creditHistory: {
    label: 'Histórico de crédito',
    icon: 'clock',
    group: 'categorical',
    codes: [
      'sem crédito anterior',
      'tudo quitado neste banco',
      'em dia até agora',
      'atrasos no passado',
      'crítico / crédito em outro banco',
    ],
  },
  purpose: {
    label: 'Finalidade',
    icon: 'tag',
    group: 'categorical',
    codes: [
      'carro novo', 'carro usado', 'móveis', 'rádio / TV', 'eletrodomésticos',
      'reparos', 'educação', 'requalificação', 'negócio próprio', 'outros',
    ],
  },
  savingsStatus: {
    label: 'Poupança',
    icon: 'wallet',
    group: 'categorical',
    codes: ['menos de 100 DM', '100 a 500 DM', '500 a 1.000 DM', '1.000 DM ou mais', 'não informado'],
  },
  employmentYears: {
    label: 'Tempo de emprego',
    icon: 'briefcase',
    group: 'categorical',
    codes: ['desempregado', 'menos de 1 ano', '1 a 4 anos', '4 a 7 anos', '7 anos ou mais'],
  },
  otherDebtors: {
    label: 'Outros devedores',
    icon: 'handshake',
    group: 'categorical',
    codes: ['nenhum', 'co-solicitante', 'avalista'],
  },
  property: {
    label: 'Patrimônio',
    icon: 'key',
    group: 'categorical',
    codes: ['imóvel', 'poupança / seguro de vida', 'carro ou outro', 'não possui'],
  },
  otherInstallments: {
    label: 'Outros financiamentos',
    icon: 'card',
    group: 'categorical',
    codes: ['em banco', 'em lojas', 'nenhum'],
  },
  housing: {
    label: 'Moradia',
    icon: 'home',
    group: 'categorical',
    codes: ['alugada', 'própria', 'cedida'],
  },
  job: {
    label: 'Ocupação',
    icon: 'briefcase',
    group: 'categorical',
    codes: [
      'não qualificada, não residente',
      'não qualificada, residente',
      'qualificada',
      'gestão / autônomo qualificado',
    ],
  },
  telephone: {
    label: 'Telefone registrado',
    icon: 'phone',
    group: 'categorical',
    codes: ['não', 'sim'],
  },
  foreignWorker: {
    label: 'Trabalhador estrangeiro',
    icon: 'globe',
    group: 'categorical',
    codes: ['sim', 'não'],
  },

  // Presente para ser NOMEADO na tela como recusado, não para ser usado.
  // É a mesma razão pela qual ele aparece em `rejected` no contrato: um
  // campo que some sem explicação parece esquecimento.
  personalStatus: {
    label: 'Estado civil e sexo',
    icon: 'shield',
    group: 'rejected',
    codes: ['masculino, separado', 'feminino', 'masculino, solteiro', 'masculino, casado/viúvo'],
  },
};

// Campo desconhecido não quebra a tela: vira um item genérico com o
// próprio nome. Trocar de fonte de dados (o `synthetic`, por exemplo)
// continua rendendo uma lista legível mesmo antes de alguém escrever o
// dicionário dela.
export const describeField = (field) => FIELDS[field] ?? {
  label: field, icon: 'field', group: 'numeric',
};

export const formatValue = (field, value) => {
  const { group, codes } = describeField(field);

  if (group === 'numeric') {
    return (FORMATTERS[field] ?? inteiro.format)(value);
  }

  return codes?.[value] ?? `código ${value}`;
};

// Como cada coluna numérica vira um campo do formulário. `step` é o passo
// dos controles do input — 50 DM em `creditAmount` e 1 no resto, porque
// meses, anos e contagens não têm meio.
//
// `suffix` é a unidade que fica ao lado do campo. Ela não entra no
// payload: o serviço recebe o número cru, e a unidade é só o que impede
// alguém de digitar 48 achando que são anos.
const NUMERIC_CONTROL = {
  durationMonths: { step: 1, suffix: 'meses' },
  creditAmount: { step: 50, suffix: 'DM' },
  installmentRate: { step: 1, suffix: 'de 4' },
  residenceSince: { step: 1, suffix: 'de 4' },
  age: { step: 1, suffix: 'anos' },
  existingCredits: { step: 1, suffix: '' },
  dependents: { step: 1, suffix: '' },
};

export const describeControl = (field) => NUMERIC_CONTROL[field] ?? { step: 1, suffix: '' };

export const formatPercent = (ratio, decimals = 1) =>
  `${(ratio * 100).toFixed(decimals).replace('.', ',')}%`;
