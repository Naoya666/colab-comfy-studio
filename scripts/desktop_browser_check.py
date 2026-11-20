"""Verify desktop and mobile Studio flows against the local offline server.

Run scripts/offline_server.py first. Chromium can be selected with OFFLINE_CHROMIUM.
"""

import os
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

BASE = os.environ.get("OFFLINE_BROWSER_BASE", "http://127.0.0.1:18788").rstrip("/")
HANDOFF = "https://comfy.example.test/_studio/session"
PROMPT = "桌面工作区预览测试"
assert BASE.startswith("http://127.0.0.1:"), "Use the offline server"
screenshots = os.environ.get("DESKTOP_SCREENSHOTS")
if screenshots:
    Path(screenshots).mkdir(parents=True, exist_ok=True)


def screenshot(page, name):
    if screenshots:
        page.screenshot(path=Path(screenshots) / f"desktop-{name}.png")


def open_tab(page, tab):
    page.get_by_role("navigation", name="工作区导航").locator(f'button[data-tab="{tab}"]').click()


def submit(page, quantity):
    page.get_by_role("spinbutton", name="生成数量").fill(str(quantity))
    page.get_by_role("button", name="立即生成").click()
    page.get_by_role("dialog", name="确认提交任务").get_by_role("button", name="确认启动").click()


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path=os.environ.get("OFFLINE_CHROMIUM"),
        args=["--no-sandbox"],
    )
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    handoff_requests = []
    unhandled = []
    errors = []

    def route(request_route):
        request = request_route.request
        if request.url == HANDOFF:
            handoff_requests.append((request.method, request.headers.get("origin"), request.post_data))
            request_route.fulfill(content_type="text/html", body="<h1>原生编辑器测试页</h1>")
        elif request.url.startswith(BASE + "/"):
            request_route.continue_()
        else:
            unhandled.append(request.url)
            request_route.abort()

    context.route("**/*", route)
    page = context.new_page()
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(BASE)
    page.locator("input[type=password]").fill("mock-local-only")
    page.get_by_role("button", name="进入工作台").click()
    api_settings = page.get_by_role("button", name="API 设置", exact=True)
    expect(api_settings).to_have_class("button")
    assert api_settings.bounding_box()["x"] > page.get_by_role("heading", name="创作", exact=True).bounding_box()["x"]
    page.get_by_role("button", name="新建项目").click()
    project_dialog = page.get_by_role("dialog", name="新建项目")
    project_dialog.get_by_role("textbox", name="项目名称").fill("桌面布局验证")
    project_dialog.get_by_role("button", name="创建", exact=True).click()
    expect(project_dialog).not_to_be_visible()

    prompt = page.get_by_role("textbox", name="提示词", exact=True)
    prompt.fill(PROMPT)
    preview = page.get_by_role("region", name="结果预览")
    expect(preview.get_by_text("生成后在这里预览")).to_be_visible()
    generate = page.get_by_role("button", name="立即生成")
    assert preview.bounding_box()["x"] > generate.bounding_box()["x"]
    assert generate.bounding_box()["y"] < 850
    screenshot(page, "create-empty")
    advanced = page.get_by_text("高级参数", exact=False).first
    advanced.click()
    assert advanced.locator("xpath=ancestor::details").get_attribute("open") is not None
    advanced.click()

    submit(page, 2)
    expect(preview.get_by_role("status")).to_be_visible()
    second = preview.get_by_role("button", name="结果 2", exact=False)
    expect(second).to_be_visible(timeout=30_000)
    second.click()
    expect(second).to_have_attribute("aria-pressed", "true")
    assert preview.locator("img").first.evaluate("img => img.decode().then(() => img.naturalWidth > 0)")
    screenshot(page, "create-result")

    preview.get_by_role("button", name="查看任务").click()
    page.get_by_role("button", name="查看详情").first.click()
    expect(page.get_by_role("region", name="任务详情").get_by_text("执行记录")).to_be_visible()
    expect(page.get_by_role("dialog", name="任务详情")).to_have_count(0)
    screenshot(page, "jobs")

    open_tab(page, "作品")
    page.get_by_role("button", name="查看 mock-out", exact=False).first.click()
    panel = page.get_by_role("region", name="作品详情")
    expect(panel.get_by_text("生成参数")).to_be_visible()
    expect(panel.get_by_text(PROMPT)).to_be_visible()
    screenshot(page, "gallery")
    panel.get_by_role("button", name="收藏", exact=True).click()
    expect(panel.get_by_role("button", name="取消收藏")).to_be_visible()
    panel.get_by_role("button", name="用同样参数再来一次").click()
    expect(prompt).to_have_value(PROMPT)

    for theme in ["dark", "light"]:
        page.evaluate("(theme) => { document.documentElement.dataset.theme = theme; }", theme)
        for width in [360, 390, 799, 800, 960, 1024, 1100, 1440, 1920]:
            page.set_viewport_size({"width": width, "height": 900})
            for name in ["创作", "任务", "作品", "原生 ComfyUI", "工作流与模型", "模型库", "算力", "设置"]:
                if name == "设置":
                    page.get_by_role("button", name="打开设置").click()
                else:
                    open_tab(page, name)
                geometry = page.locator("main").evaluate("""main => ({
                  width: document.documentElement.scrollWidth,
                  content: main.scrollWidth,
                  area: main.clientWidth,
                  nav: getComputedStyle(document.querySelector('nav')).position,
                })""")
                assert geometry["width"] == width and geometry["content"] <= geometry["area"] + 2, (theme, name, geometry)
                assert geometry["nav"] == ("fixed" if width < 800 else "sticky"), geometry

    page.set_viewport_size({"width": 390, "height": 844})
    open_tab(page, "创作")
    expect(preview).to_have_count(0)
    submit(page, 1)
    expect(page.locator('nav button[data-tab="任务"]')).to_have_attribute("aria-current", "page")
    page.get_by_role("button", name="查看详情").first.click()
    expect(page.get_by_role("dialog", name="任务详情").get_by_text("执行记录")).to_be_visible()
    page.keyboard.press("Escape")

    page.set_viewport_size({"width": 1440, "height": 900})
    page.evaluate("document.documentElement.dataset.theme = 'dark'")
    open_tab(page, "原生 ComfyUI")
    expect(page.get_by_role("region", name="会话活动").get_by_text("当前没有原生会话")).to_be_visible()
    screenshot(page, "native")

    # Intercept the external editor and use a fake signed ticket. The browser still performs the cross-origin POST.
    def native_status(status_route):
        response = status_route.fetch()
        fixture = response.json()
        fixture["data"].update(configured=True, ready=True, can_restart=False, state="running")
        status_route.fulfill(response=response, json=fixture)

    page.route("**/api/gpu/native/status", native_status)
    page.route("**/api/gpu/native/ticket", lambda r: r.fulfill(json={
        "data": {"url": HANDOFF, "ticket": "fake-desktop-ticket"},
    }))
    entry = page.get_by_role("button", name="进入原生 ComfyUI")
    with page.expect_popup() as popup:
        entry.click()
    editor = popup.value
    expect(editor).to_have_url(HANDOFF)
    assert editor.evaluate("window.opener === null")
    assert handoff_requests and all(row == ("POST", BASE, "ticket=fake-desktop-ticket") for row in handoff_requests)
    expect(page.get_by_role("heading", name="原生 ComfyUI")).to_be_visible()
    editor.close()

    page.set_viewport_size({"width": 390, "height": 844})
    entry.click()
    expect(page).to_have_url(HANDOFF)
    assert not errors and not unhandled, (errors, unhandled)
    browser.close()

print("Desktop workspaces, mobile navigation, mock batch preview, and ComfyUI handoff passed.")
