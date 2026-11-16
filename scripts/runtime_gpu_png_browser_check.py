"""Run against scripts/offline_server.py on port 18788; all GPU UI data is fake.

Checks mobile PNG extraction/import, downloads, and GPU choice persistence.
Never submits a GPU job. Screenshot is written to /tmp/studio-image-import-mobile.png.
"""

import asyncio
import io
import json

from PIL import Image, PngImagePlugin
from playwright.async_api import async_playwright


async def main():
    metadata = PngImagePlugin.PngInfo()
    metadata.add_itxt(
        "workflow",
        json.dumps(
            {
                "nodes": [{"id": 1, "type": "CustomLoader", "widgets_values": ["example.safetensors"]}],
                "links": [],
            }
        ),
        zip=True,
    )
    metadata.add_text(
        "prompt",
        json.dumps({"1": {"class_type": "CustomLoader", "inputs": {"ckpt_name": "example.safetensors"}}}),
        zip=True,
    )
    png = io.BytesIO()
    Image.new("RGB", (16, 16)).save(png, format="PNG", pnginfo=metadata)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=1)
        errors = []
        calls = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("request", lambda r: calls.append((r.method, r.url)))
        await page.goto("http://127.0.0.1:18788")
        await page.get_by_label("访问密码").fill("mock-local-only")
        await page.get_by_role("button", name="进入工作台").click()
        await page.wait_for_timeout(600)
        await page.get_by_role("button", name="打开设置").click()
        await page.get_by_role("button", name="工作流与模型").click()
        await page.get_by_text("从图片导入 ComfyUI 工作流", exact=True).click()
        before = len(calls)
        await page.locator('input[type=file][accept="image/png,.png"]').set_input_files(
            {"name": "original.png", "mimeType": "image/png", "buffer": png.getvalue()}
        )
        await page.get_by_text("workflow：已检测", exact=False).wait_for()
        assert "example.safetensors" in await page.locator("body").inner_text()
        assert not any(method == "POST" for method, url in calls[before:]), calls[before:]
        await page.get_by_role("button", name="下载 workflow.json", exact=True).click()
        await page.get_by_role("button", name="下载 prompt.json", exact=True).click()
        async with page.expect_response(
            lambda r: "/workflows/import" in r.url and r.request.method == "POST"
        ) as imported:
            await page.get_by_role("button", name="导入为待配置草稿", exact=True).click()
        response = await imported.value
        assert response.status == 201, await response.text()
        saved = (await response.json())["data"]
        assert saved["state"] == "draft" and saved["pending_configuration"]
        await page.wait_for_timeout(500)
        assert not any("/api/gpu/jobs" in url and method == "POST" for method, url in calls)
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
            "mobile horizontal overflow"
        )
        await page.screenshot(path="/tmp/studio-image-import-mobile.png", full_page=True)
        print(
            json.dumps(
                {
                    "viewport": "390x844",
                    "compressed_png_extracted": True,
                    "both_downloads_clicked": True,
                    "image_upload_requests": 0,
                    "gpu_job_creates": 0,
                    "no_horizontal_overflow": True,
                    "page_errors": errors,
                },
                ensure_ascii=False,
            )
        )
        assert not errors, errors

        # UI-only provider fixtures: no Colab allocation call is made.
        async def settings(route):
            response = await route.fetch()
            payload = await response.json()
            payload["data"]["real_enabled"] = True
            await route.fulfill(response=response, json=payload)

        async def specs(route):
            await route.fulfill(
                json={
                    "data": [
                        {
                            "key": {"variant": "VARIANT_GPU", "accelerator": g, "shape": "SHAPE_STANDARD"},
                            "eligible": True,
                        }
                        for g in ["L4", "A100"]
                    ]
                }
            )

        await page.route("**/api/gpu/settings", settings)
        await page.route("**/api/gpu/compute/specs?*", specs)
        await page.reload()
        await page.get_by_role("button", name="Colab GPU", exact=False).click()
        await page.get_by_label("本次 GPU", exact=True).wait_for()
        select = page.get_by_label("本次 GPU", exact=True)
        await select.locator("option").filter(has_text="A100").wait_for(state="attached")
        options = await select.locator("option").all_text_contents()
        assert any("L4" in v for v in options) and any("A100" in v for v in options)
        assert not any("H100" in v or "80GB" in v for v in options)
        token = await select.locator("option").filter(has_text="A100").get_attribute("value")
        await select.select_option(token)
        await page.get_by_role("button", name="立即生成", exact=True).click()
        assert "A100" in await page.get_by_role("dialog", name="确认提交任务").inner_text()
        await page.get_by_role("button", name="取消", exact=True).click()
        await page.reload()
        await page.get_by_role("button", name="Colab GPU", exact=False).click()
        await page.get_by_label("本次 GPU", exact=True).wait_for()
        assert "A100" in await page.get_by_label("本次 GPU", exact=True).input_value()
        assert await page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert not errors, errors
        print(
            "GPU UI fixtures: L4/A100 only, A100 persisted, confirmation includes GPU, no real GPU submission"
        )
        await browser.close()


asyncio.run(main())
