// --------------------------------------------------
// 20. O contrato de entrada
// --------------------------------------------------
// Dentro do laboratório os clientes vêm de um CSV que este projeto mesmo
// escreveu, e nenhuma validação faz falta. Um endpoint HTTP inverte isso:
// a entrada passa a vir de fora, em JSON, escrita por alguém que não leu
// o código — e cada suposição não checada vira um jeito diferente de a
// rede receber lixo e responder um número que parece legítimo.
//
// Três armadilhas concretas, todas silenciosas se ninguém olhar:
//
//   campo ausente     `undefined - min / range` é `NaN`. O `NaN` atravessa
//                     o tensor, a rede devolve `NaN`, e um `NaN >= limiar`
//                     é `false`: o cliente sai classificado como BAIXO
//                     RISCO por não ter mandado a idade.
//   número como texto `"48"` funciona por coerção em quase toda conta e
//                     falha justamente onde não deveria. Aceitar hoje é
//                     escolher depurar amanhã.
//   campo com typo    `durationMonth` sem o "s" seria ignorado, o campo
//                     certo ficaria ausente, e cai no primeiro caso.
//
// Nada disso é sofisticado. É só a diferença entre um script e um
// serviço: o script pode confiar na entrada, o serviço não pode.

const isNumber = (value) => typeof value === 'number' && Number.isFinite(value);

const describe = (value) => {
  if (value === undefined) {
    return 'ausente';
  }

  return `${JSON.stringify(value)} (${value === null ? 'null' : typeof value})`;
};

// As qualitativas chegam como ÍNDICE do código na lista da UCI, que é
// exatamente o que o CSV do laboratório guarda. A mensagem de erro
// devolve a lista inteira: quem chama não deveria precisar abrir o
// código para descobrir que `purpose = 3` é "rádio/TV".
const validateCategorical = (field, value, codes) => {
  if (!Number.isInteger(value) || value < 0 || value >= codes.length) {
    return `\`${field}\`: esperado um inteiro entre 0 e ${codes.length - 1} `
      + `(${codes.join(', ')}); recebido ${describe(value)}.`;
  }

  return null;
};

// Devolve os erros TODOS de uma vez, não o primeiro. Quem está integrando
// com o serviço corrige um payload por vez ou seis; a segunda opção é
// mais gentil e não custa nada.
const validateCustomer = (payload, schema) => {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    return {
      errors: ['O corpo precisa ser um objeto JSON com as features do cliente.'],
      customer: null,
    };
  }

  const { numeric, categorical, rejected = [] } = schema;
  const conhecidos = new Set([...numeric, ...Object.keys(categorical)]);
  const errors = [];
  const customer = {};

  numeric.forEach((field) => {
    if (!isNumber(payload[field])) {
      errors.push(`\`${field}\`: esperado um número; recebido ${describe(payload[field])}.`);

      return;
    }

    customer[field] = payload[field];
  });

  Object.entries(categorical).forEach(([field, codes]) => {
    const problema = validateCategorical(field, payload[field], codes);

    if (problema) {
      errors.push(problema);

      return;
    }

    customer[field] = payload[field];
  });

  // O atributo protegido tem mensagem própria porque o motivo da recusa
  // é próprio: não é um campo que sobrou, é um campo que o modelo nunca
  // recebeu e não vai receber agora por vir pela porta da frente.
  rejected
    .filter((field) => field in payload)
    .forEach((field) => errors.push(
      `\`${field}\` não é aceito: é o atributo protegido, e o modelo nunca `
      + 'o recebeu. A auditoria usa essa coluna; a decisão, não.',
    ));

  Object.keys(payload)
    .filter((field) => !conhecidos.has(field) && !rejected.includes(field))
    .forEach((field) => errors.push(`\`${field}\`: campo desconhecido.`));

  return { errors, customer: errors.length === 0 ? customer : null };
};

// A mesma informação que a validação usa, em formato publicável. Vira o
// `GET /schema`: quem integra descobre o contrato pelo serviço, em vez de
// por tentativa e erro contra o 400.
const describeSchema = (schema) => ({
  numeric: schema.numeric,
  categorical: Object.fromEntries(
    Object.entries(schema.categorical).map(([field, codes]) => [
      field,
      { range: [0, codes.length - 1], codes },
    ]),
  ),
  rejected: schema.rejected ?? [],
});

module.exports = {
  isNumber,
  validateCategorical,
  validateCustomer,
  describeSchema,
};
