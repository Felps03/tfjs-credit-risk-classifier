// Sobe a API de scoring sobre o modelo já treinado.
//
// Roda separado do `npm start` pelo mesmo motivo que o download da UCI
// roda: treinar e servir são operações de ciclo de vida diferentes. O
// treino acontece uma vez, demora minutos e produz um artefato; o serviço
// sobe em segundos a partir desse artefato e responde milhares de vezes.
// Misturar os dois é o jeito mais rápido de acabar com um serviço que
// retreina a cada deploy — e, portanto, que responde diferente a cada
// deploy.
//
//   npm start          treina e salva o pacote em `model/`
//   npm run serve      sobe a API lendo esse pacote
//   npm run serve -- --port=8080
const {
  createApi,
  listen,
  loadArtifacts,
  resolvePort,
} = require('..');

const main = async () => {
  const port = resolvePort(process.argv.slice(2));

  // Carrega UMA vez, na subida. Recarregar por requisição custaria
  // centenas de milissegundos e, pior, deixaria o serviço responder com
  // versões diferentes do modelo durante um deploy.
  const artifacts = await loadArtifacts();
  const { metadata } = artifacts;

  const server = createApi(artifacts);
  const bound = await listen(server, port);

  console.log(`API ouvindo em http://localhost:${bound}`);
  console.log(`  fonte:    ${metadata.source} (${metadata.featureNames.length} features)`);
  console.log(`  limiar:   ${metadata.threshold.toFixed(4)} — ${metadata.thresholdStrategy}`);
  console.log(`  treinado: ${metadata.savedAt}`);
  console.log('');
  console.log('  POST /risk-score   pontua um cliente');
  console.log('  GET  /schema       o contrato de entrada');
  console.log('  GET  /health       o pacote carregado');

  // Sem isto, um `docker stop` ou um Ctrl+C derruba a conexão em curso.
  const shutdown = () => {
    console.log('\nEncerrando...');
    server.close(() => {
      artifacts.dispose();
      process.exit(0);
    });
  };

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
};

main().catch((error) => {
  console.error(`\n${error.message}\n`);
  process.exitCode = 1;
});
