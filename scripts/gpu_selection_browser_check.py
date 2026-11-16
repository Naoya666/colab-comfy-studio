"""GPU selection regression against offline_server.py; all Colab responses are fixtures."""
import asyncio
import json
import os
from urllib.parse import parse_qs, urlsplit

from playwright.async_api import async_playwright


async def main():
    base = os.getenv('GPU_SELECTION_TEST_URL', 'http://127.0.0.1:18788')
    connected = True
    offline = False
    queries = []
    writes = []
    errors = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 390, 'height': 844})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda r: writes.append(r.url) if r.method == 'POST' and '/api/gpu/' in r.url else None)

        async def settings(route):
            response = await route.fetch()
            payload = await response.json()
            payload['data'].update(real_enabled=True, offline_mode=offline)
            await route.fulfill(response=response, json=payload)

        async def accounts(route):
            def account(ident, colab):
                return {'account_id': ident, 'alias': ident, 'enabled': True,
                        'allow_allocation': True, 'status': 'connected',
                        'authorization': {'colab': {'connected': colab}}}
            await route.fulfill(json={'data': [account('drive-only', False), account('cli-A', True),
                                               account('cli-B', connected)]})

        async def specs(route):
            query = parse_qs(urlsplit(route.request.url).query)
            ident = query.get('account_id', [None])[0]
            assert ident in {'cli-A', 'cli-B'}, query
            queries.append(ident)
            await route.fulfill(json={'data': [{'key': {'variant': 'VARIANT_GPU',
                'accelerator': 'T4' if ident == 'cli-A' else 'L4', 'shape': 'SHAPE_STANDARD'},
                'eligible': True}]})

        await page.route('**/api/gpu/settings', settings)
        await page.route('**/api/gpu/accounts', accounts)
        await page.route('**/api/gpu/compute/specs?*', specs)
        await page.goto(base)
        await page.get_by_label('访问密码').fill('mock-local-only')
        await page.get_by_role('button', name='进入工作台').click()
        await page.get_by_role('button', name='Colab GPU', exact=False).click()
        gpu = page.get_by_label('本次 GPU', exact=True)
        refresh = page.get_by_role('button', name='刷新当前账号支持的 GPU', exact=True)
        await gpu.locator('option').filter(has_text='T4').wait_for(state='attached')
        await page.get_by_text('查询账号：cli-A。', exact=False).wait_for()
        assert queries == ['cli-A'], queries
        await refresh.click()
        await refresh.wait_for(state='visible')
        await page.wait_for_function("document.querySelector('#workflow-gpu').options.length > 1")
        assert queries == ['cli-A', 'cli-A'], queries
        await page.locator('#runtime-account').select_option('cli-B')
        await gpu.locator('option').filter(has_text='L4').wait_for(state='attached')
        assert queries[-1] == 'cli-B'
        connected = False
        before = len(queries)
        await refresh.click()
        await page.get_by_text('所选账号未连接 Colab 或已停用', exact=False).wait_for()
        assert len(queries) == before
        connected = True
        await refresh.click()
        await gpu.locator('option').filter(has_text='L4').wait_for(state='attached')
        offline = True
        before = len(queries)
        await refresh.click()
        await page.get_by_text('站点处于离线测试模式，不能查询真实 Colab GPU。', exact=True).wait_for()
        assert len(queries) == before
        assert not writes, writes
        assert not errors, errors
        assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        print(json.dumps({'automatic_account': True, 'refresh_reloads_accounts': True,
                          'explicit_account': True, 'no_unauthorized_fallback': True,
                          'offline_blocked': True, 'gpu_writes': len(writes), 'page_errors': errors}))
        await page.unroute_all(behavior='wait')
        await browser.close()


asyncio.run(main())
