// Tabela de texto com colunas alinhadas: primeira coluna à esquerda,
// as demais à direita. Usada pela matriz e pela comparação de limiares.
const formatTable = (headers, rows) => {
  const widths = headers.map((header, column) => Math.max(
    header.length,
    ...rows.map((row) => row[column].length),
  ));

  const line = (cells) => cells
    .map((cell, column) => (column === 0
      ? cell.padEnd(widths[column])
      : cell.padStart(widths[column])))
    .join(' | ');

  return [
    line(headers),
    widths.map((width) => '-'.repeat(width)).join('-+-'),
    ...rows.map(line),
  ].join('\n');
};

module.exports = { formatTable };
