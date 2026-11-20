"""Run against scripts/offline_server.py on port 18788; all GPU UI data is fake.

Checks mobile PNG extraction/import, downloads, and GPU choice persistence.
Never submits a GPU job. Screenshot is written to /tmp/studio-image-import-mobile.png.
"""

import asyncio
import io
import json

from PIL import Image, PngImagePlugin
from playwright.async_api import async_playwright, expect


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
        api_settings = page.get_by_role("button", name="API 设置", exact=True)
        await expect(api_settings).to_have_class("button")
        await api_settings.click()
        await expect(page.get_by_role("heading", name="创作 API 设置")).to_be_visible()
        runtime_limit = page.get_by_label("单个任务最长运行时间（分钟）")
        await runtime_limit.fill("1501" if await runtime_limit.input_value() == "1500" else "1500")
        await page.get_by_label("任务完成后的策略").select_option("smart")
        idle_limit = page.get_by_label("空闲保留时间（分钟）")
        await idle_limit.fill("91" if await idle_limit.input_value() == "90" else "90")
        await page.get_by_role("button", name="保存运行配置").click()
        await expect(page.get_by_role("status").get_by_text("运行配置已保存")).to_be_visible()
        await page.get_by_role("button", name="返回创作").click()
        workflow_tab = page.get_by_role("navigation", name="工作区导航").get_by_role("button", name="工作流与模型")
        await workflow_tab.click()
        await expect(workflow_tab).to_have_attribute("aria-current", "page")
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

        async def dependencies(route):
            common = {"node_id": "1", "node_type": "CustomLoader", "field": "ckpt_name",
                      "model_type": "checkpoints", "architecture": "unknown", "required": True}
            await route.fulfill(json={"data": {"workflow_state": "draft", "dependencies": [
                {**common, "id": "bound", "reference": "bound-weight.safetensors", "model_id": "model-1", "status": "matched"},
                {**common, "id": "unbound", "reference": "missing-weight.safetensors", "status": "unresolved"},
            ]}})

        await page.route(f"**/workflows/{saved['id']}/model-dependencies", dependencies)
        group = page.get_by_role("button", name=saved["workflow_key"], exact=False)
        await group.click()
        version = page.locator("article").filter(has=group)
        await version.get_by_text("模型依赖", exact=True).click()
        filter_checkbox = version.get_by_role("checkbox", name="只显示未绑定的模型")
        bound = version.get_by_text("bound-weight.safetensors", exact=True)
        unbound = version.get_by_text("missing-weight.safetensors", exact=True)
        await expect(bound).to_be_visible()
        await filter_checkbox.check()
        await expect(bound).to_have_count(0)
        await expect(unbound).to_be_visible()
        await filter_checkbox.uncheck()
        await expect(bound).to_be_visible()
        delete = version.get_by_role("button", name="删除", exact=True)
        page.once("dialog", lambda dialog: dialog.dismiss())
        await delete.click()
        await expect(group).to_be_visible()
        page.once("dialog", lambda dialog: dialog.accept())
        await delete.click()
        await expect(group).to_have_count(0)
        await page.reload()
        await workflow_tab.click()
        await page.get_by_role("textbox", name="搜索工作流").fill(saved["workflow_key"])
        await expect(page.get_by_text("没有匹配的工作流", exact=True)).to_be_visible()
        await expect(group).to_have_count(0)
        print(
            json.dumps(
                {
                    "viewport": "390x844",
                    "compressed_png_extracted": True,
                    "both_downloads_clicked": True,
                    "image_upload_requests": 0,
                    "gpu_job_creates": 0,
                    "generation_api_long_timeouts_saved": True,
                    "dependency_filter_and_workflow_deletion": True,
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
            payload["data"]["offline_mode"] = False
            await route.fulfill(response=response, json=payload)

        async def accounts(route):
            await route.fulfill(json={"data": [{
                "account_id": "gpu-ui-fixture", "alias": "离线规格测试账号", "enabled": True,
                "allow_allocation": True, "status": "connected",
                "authorization": {"colab": {"connected": True}},
            }]})

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
        await page.route("**/api/gpu/accounts", accounts)
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
        await page.unroute_all(behavior='wait')
        await browser.close()


asyncio.run(main())
