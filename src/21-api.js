const http = require('node:http');

const { API_BODY_LIMIT, API_PORT } = require('./00-constants');
const { round } = require('./19-artifacts');
const { describeSchema, validateCustomer } = require('./20-contract');

// --------------------------------------------------
// 21. O serviço
// --------------------------------------------------
// Sem framework, sem dependência: `node:http` já faz o que este endpoint
// precisa, e o projeto inteiro se sustenta em zero pacotes além do
// TensorFlow. A camada HTTP não é a parte difícil de servir um modelo —
// a parte difícil ficou em `19-artifacts.js`, e já foi resolvida lá.
//
// O que este módulo faz de propósito é NÃO saber nada sobre modelo:
// recebe um `predict` pronto e um limiar, e nunca toca em tensor,
// scaler ou codificação. É o que torna o serviço testável sem treinar
// nada — os testes injetam uma função de pontuação de mentira.

// A probabilidade é arredondada com a MESMA precisão do limiar gravado no
// pacote (veja `19-artifacts.js`), e a classificação sai do valor já
// arredondado. Os dois lados da comparação vivem na mesma escala, então
// a resposta nunca se contradiz na fronteira.
const scoreCustomer = ({ predict, threshold, schema, metadata }) => (payload) => {
  const { errors, customer } = validateCustomer(payload, schema);

  if (errors.length > 0) {
    return {
      status: 400,
      body: { error: 'Requisição inválida.', details: errors },
    };
  }

  const riskProbability = round(predict(customer));

  return {
    status: 200,
    body: {
      riskProbability,
      classification: riskProbability >= threshold ? 'HIGH_RISK' : 'LOW_RISK',

      // O limiar viaja na resposta porque ele é uma escolha de negócio,
      // não uma propriedade do modelo: sem ele, `0.31` não diz nada.
      // Este veio da matriz de custo, medido no treino, e está gravado
      // no pacote — não é o `0.5` herdado.
      threshold,
      model: {
        source: metadata.source,
        features: metadata.featureNames.length,
        savedAt: metadata.savedAt,
      },
    },
  };
};

// `close` desliga o keep-alive nesta resposta. É o que fecha o caso do
// corpo grande demais: a requisição ficou pela metade, então a conexão
// não pode ser reaproveitada — o resto do upload que ainda está vindo
// seria lido como se fosse a próxima requisição.
const sendJson = (res, status, body, { close = false } = {}) => {
  const payload = JSON.stringify(body);

  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(payload),
    ...(close ? { connection: 'close' } : {}),
  });
  res.end(payload);
};

// Lê o corpo com teto. Sem o teto, um POST grande o bastante derruba o
// processo antes de qualquer validação rodar — a validação de esquema
// não protege contra o que nunca chega a ser um objeto.
const readJsonBody = (req, limit = API_BODY_LIMIT) => new Promise((resolve) => {
  const chunks = [];
  let size = 0;
  let excedeu = false;

  req.on('data', (chunk) => {
    if (excedeu) {
      return;
    }

    size += chunk.length;

    if (size > limit) {
      // `pause` para de acumular sem matar o socket. Chamar `destroy`
      // aqui seria mais direto e estaria ERRADO: a conexão morreria antes
      // de a resposta sair, e quem chamou veria um reset em vez do 413 —
      // ou seja, ficaria sem saber por que foi recusado. Quem fecha a
      // conexão é a resposta, logo abaixo.
      excedeu = true;
      req.pause();
      resolve({ ok: false, status: 413, error: `Corpo maior que ${limit} bytes.` });

      return;
    }

    chunks.push(chunk);
  });

  req.on('error', () => resolve({
    ok: false, status: 400, error: 'Falha ao ler o corpo da requisição.',
  }));

  req.on('end', () => {
    const raw = Buffer.concat(chunks).toString('utf8');

    if (raw.trim() === '') {
      resolve({ ok: false, status: 400, error: 'Corpo vazio.' });

      return;
    }

    try {
      resolve({ ok: true, value: JSON.parse(raw) });
    } catch (error) {
      resolve({ ok: false, status: 400, error: `JSON inválido: ${error.message}` });
    }
  });
});

// Um cliente que manda `text/plain` quase sempre está mandando JSON e
// esqueceu o cabeçalho. Recusar com 415 e dizer o que se espera custa
// menos do que deixar passar e explicar depois por que o parse falhou.
const isJsonRequest = (req) =>
  (req.headers['content-type'] ?? '').split(';')[0].trim() === 'application/json';

const createRequestListener = (routes) => async (req, res) => {
  const { pathname } = new URL(req.url, 'http://localhost');
  const route = routes[pathname];

  if (!route) {
    sendJson(res, 404, {
      error: `Rota desconhecida: ${pathname}.`,
      routes: Object.entries(routes).map(([path, handlers]) =>
        `${Object.keys(handlers).join('/')} ${path}`),
    });

    return;
  }

  const handler = route[req.method];

  if (!handler) {
    res.setHeader('allow', Object.keys(route).join(', '));
    sendJson(res, 405, {
      error: `${req.method} não é aceito em ${pathname}.`,
      allow: Object.keys(route),
    });

    return;
  }

  try {
    const { status, body } = await handler(req);

    sendJson(res, status, body, { close: status === 413 });
  } catch (error) {
    // O que vaza para quem chama é uma frase; o que vai para o log é o
    // erro inteiro. Devolver a stack seria entregar caminho de arquivo e
    // versão de biblioteca a quem só mandou um POST.
    console.error('Erro ao responder:', error);
    sendJson(res, 500, { error: 'Erro interno.' });
  }
};

// Monta as rotas a partir de um pacote já carregado. `POST /risk-score`
// é o endpoint; os outros dois existem porque um serviço sem healthcheck
// não sobe em lugar nenhum, e porque descobrir o contrato por tentativa
// e erro contra o 400 é um jeito ruim de integrar.
const createRoutes = (artifacts) => {
  const { metadata, predict, source } = artifacts;
  const schema = source.requestSchema;
  const score = scoreCustomer({
    predict,
    threshold: metadata.threshold,
    schema,
    metadata,
  });

  return {
    '/risk-score': {
      POST: async (req) => {
        if (!isJsonRequest(req)) {
          return {
            status: 415,
            body: { error: 'Envie content-type: application/json.' },
          };
        }

        const body = await readJsonBody(req);

        if (!body.ok) {
          return { status: body.status, body: { error: body.error } };
        }

        return score(body.value);
      },
    },

    '/schema': {
      GET: async () => ({
        status: 200,
        body: {
          source: metadata.source,
          encoding: metadata.encoding,
          threshold: metadata.threshold,
          request: describeSchema(schema),
        },
      }),
    },

    '/health': {
      GET: async () => ({
        status: 200,
        body: {
          status: 'ok',
          model: {
            source: metadata.source,
            features: metadata.featureNames.length,
            savedAt: metadata.savedAt,
          },
        },
      }),
    },
  };
};

const createApi = (artifacts) =>
  http.createServer(createRequestListener(createRoutes(artifacts)));

// Promisificado para que o script de entrada possa dar `await` e imprimir
// a porta REAL — com `--port=0` o sistema escolhe uma, e é assim que os
// testes sobem o serviço sem disputar porta com nada.
const listen = (server, port = API_PORT) => new Promise((resolve, reject) => {
  server.once('error', reject);
  server.listen(port, () => resolve(server.address().port));
});

module.exports = {
  scoreCustomer,
  sendJson,
  readJsonBody,
  isJsonRequest,
  createRequestListener,
  createRoutes,
  createApi,
  listen,
};
