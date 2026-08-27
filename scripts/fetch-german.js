// Baixa o German Credit da UCI e converte para o schema do laboratório.
//
// Roda separado do `npm start` de propósito: treinar não deve depender de
// rede. O CSV convertido é versionado, então quem clona o repositório já
// tem os dados — este script existe para auditar a origem e para regerar
// o arquivo quando se quiser mudar o recorte de colunas.
const {
  GERMAN_SOURCE_URL,
  GERMAN_CSV_PATH,
  GERMAN_COLUMNS,
  GERMAN_PRECISION,
  parseGermanCsv,
  writeCustomersCsv,
} = require('..');

const main = async () => {
  console.log('Baixando de:', GERMAN_SOURCE_URL);

  const response = await fetch(GERMAN_SOURCE_URL);

  if (!response.ok) {
    throw new Error(`a UCI respondeu ${response.status} ${response.statusText}`);
  }

  const customers = parseGermanCsv(await response.text());
  const bad = customers.filter(({ risk }) => risk === 1).length;

  writeCustomersCsv(customers, GERMAN_CSV_PATH, {
    columns: GERMAN_COLUMNS,
    precision: GERMAN_PRECISION,
  });

  console.log('Clientes convertidos:', customers.length);
  console.log(
    `Maus pagadores: ${bad} (${((100 * bad) / customers.length).toFixed(1)}%)`,
  );
  console.log('Colunas:', GERMAN_COLUMNS.join(', '));
  console.log('Salvo em:', GERMAN_CSV_PATH);
};

main().catch((error) => {
  console.error('Falha ao baixar o dataset:', error.message);
  process.exitCode = 1;
});
