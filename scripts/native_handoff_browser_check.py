"""Verify browser form Origin under the application's document referrer policy.

Uses intercepted HTTPS fixtures and fake form data; never contacts Colab or Drive.
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'api'))
from app.main import security_context  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import HTMLResponse  # noqa: E402


async def main():
    async def response(request):
        return HTMLResponse('<form method="post" action="https://comfy.example.test/_studio/session"><input name="ticket" value="fake-test-ticket"></form>')

    request = Request({'type': 'http', 'method': 'GET', 'path': '/', 'headers': [], 'query_string': b''})
    document = await security_context(request, response)
    policy = document.headers['referrer-policy']
    root = Path(__file__).resolve().parents[1]
    helper = subprocess.check_output([
        'node', '--input-type=module', '-e',
        "import ts from 'typescript'; import fs from 'node:fs'; process.stdout.write(ts.transpileModule(fs.readFileSync('src/lib/native-handoff.ts','utf8'), {compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText.replace('export function','function'));",
    ], cwd=root / 'web', text=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for value, use_helper, expected in [
            ('same-origin', False, 'null'), (policy, False, 'https://studio.example.test'),
            ('same-origin', True, 'https://studio.example.test'),
            ('no-referrer', True, 'https://studio.example.test'),
        ]:
            page = await browser.new_page()
            seen = []

            async def route(r):
                if r.request.url.startswith('https://studio.example.test'):
                    await r.fulfill(content_type='text/html', headers={'Referrer-Policy': value}, body=document.body)
                else:
                    seen.append(r.request.headers.get('origin'))
                    assert r.request.method == 'POST'
                    await r.fulfill(content_type='text/html', body='fixture')

            await page.route('**/*', route)
            await page.goto('https://studio.example.test/')
            if use_helper:
                await page.evaluate(helper + '\nsubmitNativeHandoff(document, ' + json.dumps({
                    'url': 'https://comfy.example.test/_studio/session', 'ticket': 'fake-test-ticket',
                }) + ');')
            else:
                await page.evaluate('document.forms[0].submit()')
            await page.wait_for_url('https://comfy.example.test/_studio/session')
            assert seen == [expected], seen
            await page.close()
        await browser.close()
    print(json.dumps({'old_policy_origin': 'null', 'fixed_document_policy': policy,
                      'fixed_handoff_origin': 'https://studio.example.test',
                      'old_document_with_frontend_fix': True, 'real_gpu_requests': 0}))


asyncio.run(main())
