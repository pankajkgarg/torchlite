const { test, expect } = require('@playwright/test');
const fs = require('fs');
const http = require('http');
const path = require('path');

const siteRoot = path.resolve(__dirname, '../jupyterlite/_output');
let server;

const contentTypes = new Map([
  ['.css', 'text/css; charset=utf-8'],
  ['.html', 'text/html; charset=utf-8'],
  ['.ipynb', 'application/x-ipynb+json'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.json', 'application/json'],
  ['.mjs', 'text/javascript; charset=utf-8'],
  ['.png', 'image/png'],
  ['.svg', 'image/svg+xml'],
  ['.wasm', 'application/wasm'],
  ['.whl', 'application/zip'],
]);

test.beforeAll(async () => {
  if (!fs.existsSync(path.join(siteRoot, 'lab/index.html'))) {
    throw new Error('JupyterLite has not been built; see jupyterlite/README.md');
  }

  server = http.createServer((request, response) => {
    const pathname = decodeURIComponent(new URL(request.url, 'http://127.0.0.1').pathname);
    let filePath = path.resolve(siteRoot, pathname.replace(/^\/+/, ''));
    if (!filePath.startsWith(`${siteRoot}${path.sep}`)) {
      response.writeHead(403).end('Forbidden');
      return;
    }
    if (pathname.endsWith('/')) filePath = path.join(filePath, 'index.html');

    fs.readFile(filePath, (error, contents) => {
      if (error) {
        response.writeHead(error.code === 'ENOENT' ? 404 : 500).end(error.message);
        return;
      }
      response.writeHead(200, {
        'Content-Type': contentTypes.get(path.extname(filePath)) || 'application/octet-stream',
        'Cross-Origin-Embedder-Policy': 'credentialless',
        'Cross-Origin-Opener-Policy': 'same-origin',
      }).end(contents);
    });
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(8124, '127.0.0.1', resolve);
  });
});

test.afterAll(async () => {
  if (server) await new Promise(resolve => server.close(resolve));
});

const externalBrowser = process.env.PLAYWRIGHT_EXECUTABLE_PATH;
test.use(externalBrowser
  ? { launchOptions: { executablePath: externalBrowser } }
  : { channel: 'chrome' });
test.setTimeout(300_000);
test.describe.configure({ mode: 'serial' });

async function runAllCells(page, notebook) {
  await page.goto(
    `http://127.0.0.1:8124/lab/index.html?path=${notebook}`,
    { waitUntil: 'domcontentloaded' },
  );
  await expect(page.locator('.jp-Notebook')).toBeVisible({ timeout: 60_000 });
  await page.getByRole('menuitem', { name: 'Run' }).click();
  await page.getByRole('menuitem', { name: 'Run All Cells', exact: true }).click();
}

test('runs the makemore bigram notebook in a Pyodide kernel', async ({ page }) => {
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push(message.text());
  });

  await runAllCells(page, '01-makemore-bigram.ipynb');

  const lossOutput = page.getByText(/loss: 3\.7557 -> 2\.58\d{2}/);
  await expect(lossOutput).toBeVisible({ timeout: 240_000 });
  await expect(page.getByText('torchlite 0.3.0')).toBeVisible();
  await expect(page.getByText('228146 training bigrams, 27 classes')).toBeVisible();

  const relevantErrors = browserErrors.filter(
    message => !message.includes('Failed to load resource: the server responded with a status of 404'),
  );
  expect(relevantErrors, relevantErrors.join('\n')).toEqual([]);
});

test('runs the makemore MLP and BatchNorm notebook in a Pyodide kernel', async ({ page }) => {
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push(message.text());
  });

  await runAllCells(page, '02-makemore-mlp-batchnorm.ipynb');

  await expect(page.getByText(/mlp loss: 2\.7188 -> 0\.46\d{2}/)).toBeVisible({ timeout: 240_000 });
  await expect(page.getByText(/batchnorm gradients: ok; output std=0\.97\d{2}/)).toBeVisible();

  const relevantErrors = browserErrors.filter(
    message => !message.includes('Failed to load resource: the server responded with a status of 404'),
  );
  expect(relevantErrors, relevantErrors.join('\n')).toEqual([]);
});

test('runs the hierarchical WaveNet-shaped notebook in a Pyodide kernel', async ({ page }) => {
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push(message.text());
  });

  await runAllCells(page, '03-makemore-wavenet.ipynb');

  await expect(page.getByText(/wavenet loss: \d+\.\d+ -> \d+\.\d+; parameters=\d+/)).toBeVisible({
    timeout: 240_000,
  });

  const relevantErrors = browserErrors.filter(
    message => !message.includes('Failed to load resource: the server responded with a status of 404'),
  );
  expect(relevantErrors, relevantErrors.join('\n')).toEqual([]);
});

test('trains a decoder-only transformer in a Pyodide kernel', async ({ page }) => {
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push(message.text());
  });

  await runAllCells(page, '04-tiny-gpt.ipynb');

  await expect(page.getByText(/tiny-gpt loss: 2\.4008 -> 1\.29\d{2}; parameters=1112/)).toBeVisible({
    timeout: 240_000,
  });

  const relevantErrors = browserErrors.filter(
    message => !message.includes('Failed to load resource: the server responded with a status of 404'),
  );
  expect(relevantErrors, relevantErrors.join('\n')).toEqual([]);
});

test('runs source-lesson compatibility APIs in a Pyodide kernel', async ({ page }) => {
  const browserErrors = [];
  page.on('pageerror', error => browserErrors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') browserErrors.push(message.text());
  });

  await runAllCells(page, '05-source-lesson-compat.ipynb');

  await expect(page.getByText('source-lesson compatibility: ok')).toBeVisible({
    timeout: 240_000,
  });
  await expect(page.getByText('makemore lesson APIs: ok')).toBeVisible();
  await expect(page.getByText('gpt lesson APIs: ok')).toBeVisible();

  const relevantErrors = browserErrors.filter(
    message => !message.includes('Failed to load resource: the server responded with a status of 404'),
  );
  expect(relevantErrors, relevantErrors.join('\n')).toEqual([]);
});
