"""Offline mobile/desktop checks; run against scripts/offline_server.py :18788.

Exercises the real API/UI catalog and queue. No model-worker, Drive, source
API, upload session or GPU is invoked. Credentials below are local fake data.
"""
import asyncio
import json

from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 390, "height": 844})
        errors = []
        calls = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda r: calls.append((r.method, r.url)))
        await page.goto('http://127.0.0.1:18788')
        await page.get_by_label('访问密码').fill('mock-local-only')
        await page.get_by_role('button', name='进入工作台').click()
        await page.get_by_role('button', name='模型库', exact=True).click()
        await page.get_by_role('heading', name='统一模型库').wait_for()
        await page.get_by_text('添加模型 / 上传 / 导入 Drive', exact=True).click()
        await page.get_by_label('文件名（通用链接 / Drive）').fill('browser-test.safetensors')
        await page.get_by_label('模型架构', exact=True).fill('SDXL')
        await page.get_by_label('模型页面或 HTTPS 下载链接').fill('https://example.com/browser-test.safetensors')
        await page.get_by_role('button', name='解析来源并选择文件').click()
        await page.get_by_role('button', name='选择此文件并准备').click()
        await page.get_by_role('heading', name='browser-test.safetensors', exact=True).wait_for()
        await page.get_by_role('button', name='取消准备').click()
        await page.get_by_text('模型准备已取消，可重试', exact=True).wait_for()
        await page.get_by_label('搜索名称 / 文件名').fill('not-found')
        assert await page.get_by_text('暂无符合条件的模型。', exact=False).count()
        await page.get_by_label('搜索名称 / 文件名').fill('browser-test')
        await page.get_by_role('button', name='详情与引用').click()
        await page.get_by_label('显示名称', exact=True).fill('Browser SDXL')
        await page.get_by_label('标签（逗号分隔）').fill('offline, shared')
        await page.get_by_role('button', name='保存信息').click()
        await page.get_by_role('heading', name='Browser SDXL').first.wait_for()
        await page.get_by_role('button', name='关闭详情').click()
        await page.get_by_text('模型下载源凭据', exact=True).click()
        await page.get_by_label('Civitai API Key', exact=False).fill('fake-offline-key')
        await page.get_by_role('button', name='保存凭据').nth(1).click()
        await page.get_by_text('凭据已加密保存', exact=True).wait_for()
        assert 'fake-offline-key' not in await page.locator('body').inner_text()
        for width, height in [(390,844),(1280,900)]:
            await page.set_viewport_size({"width":width,"height":height})
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        await page.set_viewport_size({"width":390,"height":844})
        await page.screenshot(path='/tmp/phase3-model-library-mobile.png',full_page=True)
        assert not errors, errors
        assert not any('/api/gpu/jobs' in url and method == 'POST' for method,url in calls)
        assert not any('googleapis.com' in url or 'civitai.com' in url or 'huggingface.co' in url for _,url in calls)
        print(json.dumps({"catalog_add_search_edit":True,"queue_cancel":True,"masked_credentials":True,
                          "mobile_desktop_no_overflow":True,"real_provider_requests":0,"page_errors":errors}))
        await browser.close()


if __name__ == '__main__':
    asyncio.run(main())
