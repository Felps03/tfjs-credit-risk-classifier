const fs = require('node:fs');
const path = require('node:path');

// --------------------------------------------------
// 22. Os arquivos da página
// --------------------------------------------------
// A API já responde JSON; o que falta é entregar a página que consome
// esse JSON. Servir um punhado de arquivos estáticos é a menor coisa que
// resolve isso — e continua sendo `node:fs` e `node:path`, sem framework,
// sem bundler e sem dependência nova, pelo mesmo motivo que a API não
// tem nenhuma.
//
// O módulo NÃO vira rota. Ele entra como o último recurso do roteador de
// `21-api.js`: quando o pathname não bate com nenhuma das três rotas
// publicadas, o arquivo é tentado; se ele também não existir, o 404 de
// sempre responde. É o que mantém `/xpto` respondendo "rota desconhecida"
// em vez de "arquivo não encontrado", e o que deixa `createRoutes`
// continuar sendo exatamente as três rotas da API.
const WEB_DIR = path.join(__dirname, '..', 'web');

// Um navegador que recebe `index.html` sem `content-type` renderiza o
// HTML como texto puro, e um módulo ES servido como `text/plain` é
// recusado pelo `<script type="module">` com um erro que não diz isso.
// A tabela é curta porque o site é: cinco extensões e nada mais.
const CONTENT_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json; charset=utf-8',
};

const contentType = (filePath) =>
  CONTENT_TYPES[path.extname(filePath).toLowerCase()] ?? 'application/octet-stream';

// `/` não é um arquivo; é a intenção de abrir a página.
const withIndex = (pathname) => (pathname.endsWith('/') ? `${pathname}index.html` : pathname);

// `GET /../../etc/passwd` chega como um pathname perfeitamente válido, e
// `path.join` o resolveria para fora da pasta sem reclamar. São duas
// travas, nesta ordem:
//
//   1. `path.posix.normalize` sobre o caminho decodificado e o `.` na
//      frente: o `..` que subiria além da raiz é DESCARTADO, então
//      `/../package.json` vira `web/package.json` — que não existe, e
//      portanto vira 404. O arquivo do projeto nunca é alcançado.
//   2. a comparação de prefixo, que é a rede de segurança caso o passo
//      anterior mude. Ela inclui o separador de propósito: sem ele, uma
//      pasta vizinha chamada `web-secreta` passaria no
//      `startsWith('/.../web')`.
//
// A ordem importa: decodificar ANTES de normalizar é o que fecha o caso
// do `..` percent-encoded (`%2e%2e`), que passaria intacto por um
// normalize aplicado à URL crua.
const resolveAsset = (pathname, root = WEB_DIR) => {
  const decoded = (() => {
    try {
      return decodeURIComponent(withIndex(pathname));
    } catch {
      // `%` solto é URI inválida. Quem manda isso não está pedindo página.
      return null;
    }
  })();

  if (decoded === null || decoded.includes('\0')) {
    return null;
  }

  const resolved = path.resolve(root, `.${path.posix.normalize(decoded)}`);

  if (resolved !== root && !resolved.startsWith(root + path.sep)) {
    return null;
  }

  return resolved;
};

// Devolve o arquivo pronto para a resposta, ou `null` para "não é meu" —
// e é esse `null` que devolve a palavra ao 404 do roteador.
// O `web/package.json` existe para o Node, não para o navegador: ele diz
// que os módulos de `web/js/` são ESM, e é o que permite testá-los sem
// build. Ele não faz parte da página, então não é servido.
//
// A lista é comparada com o caminho RESOLVIDO, e não com o pathname cru.
// Comparar com o cru seria contornável pela primeira grafia diferente que
// alguém tentasse — `/../package.json` e `/%2e%2e/package.json` chegam
// aqui como texto diferente e terminam no mesmo arquivo, que é
// precisamente o que a normalização existe para garantir.
const NAO_SERVIDOS = new Set(['package.json']);

const readAsset = (pathname, root = WEB_DIR) => {
  const resolved = resolveAsset(pathname, root);

  if (resolved === null || NAO_SERVIDOS.has(path.relative(root, resolved))) {
    return null;
  }

  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isFile()) {
    return null;
  }

  return { type: contentType(resolved), body: fs.readFileSync(resolved) };
};

// `no-cache` não é "não guarde": é "guarde, mas revalide antes de usar".
// Em um laboratório que se edita e se recarrega, é a diferença entre ver
// a mudança e achar que o CSS não pegou.
const createWebHandler = (root = WEB_DIR) => (req, pathname) => {
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    return null;
  }

  const asset = readAsset(pathname, root);

  return asset === null ? null : { status: 200, headers: { 'cache-control': 'no-cache' }, ...asset };
};

module.exports = {
  WEB_DIR,
  NAO_SERVIDOS,
  CONTENT_TYPES,
  contentType,
  withIndex,
  resolveAsset,
  readAsset,
  createWebHandler,
};
