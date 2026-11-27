"""Lossless, binary-level image metadata stripping for privacy-safe downloads.

Supports PNG, JPEG, WebP, and GIF without decoding or re-encoding image pixels:
- PNG: allowlists image/color/animation chunks and removes tEXt, zTXt, iTXt, eXIf, tIME, and private chunks.
- JPEG: removes COM, APP1 (EXIF/XMP), APP13 (IPTC/Photoshop), and other non-color APP segments while keeping JFIF, ICC_PROFILE, Adobe color transform markers, and a minimal synthetic EXIF Orientation tag when needed (values 2..8).
- WebP: removes EXIF, XMP, and unknown RIFF chunks while preserving ICCP/ALPH/ANIM/ANMF and updating/simplifying VP8X flags.
- GIF: removes Comment and non-loop Application extensions (such as XMP) while keeping frames and animation timing.
"""

from __future__ import annotations

import re
import struct
from pathlib import PurePosixPath
from urllib.parse import quote

STRIPPABLE_IMAGE_MIMES = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
})

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_KEEP_CHUNKS = frozenset({
    # Critical chunks
    b"IHDR",
    b"PLTE",
    b"IDAT",
    b"IEND",
    # Color, transparency, and physical display chunks
    b"tRNS",
    b"cHRM",
    b"gAMA",
    b"iCCP",
    b"sBIT",
    b"sRGB",
    b"bKGD",
    b"pHYs",
    b"hIST",
    b"cICP",
    b"mDCV",
    b"cLLI",
    # APNG animation chunks
    b"acTL",
    b"fcTL",
    b"fdAT",
})
_MAX_PNG_CHUNKS = 20_000

_WEBP_KEEP_CHUNKS = frozenset({
    b"VP8X",
    b"ICCP",
    b"ANIM",
    b"ANMF",
    b"ALPH",
    b"VP8 ",
    b"VP8L",
})
_WEBP_VP8X_EXIF_FLAG = 0x08
_WEBP_VP8X_XMP_FLAG = 0x04
_MAX_WEBP_CHUNKS = 20_000

_GIF_KEEP_APP_IDS = frozenset({b"NETSCAPE2.0", b"ANIMEXTS1.0"})
_MAX_GIF_BLOCKS = 50_000

_ALREADY_CLEAN_RE = re.compile(r"(?:^|[_-])clean$", re.IGNORECASE)


class ImageMetadataStripError(ValueError):
    """Raised when image bytes are malformed or cannot be safely stripped."""


def is_strippable_image_mime(mime: str | None) -> bool:
    if not isinstance(mime, str):
        return False
    return mime.strip().lower() in STRIPPABLE_IMAGE_MIMES


def clean_download_filename(filename: str | None) -> str:
    """Append a clean suffix before the file extension without duplicating separators."""
    raw = str(filename or "").strip()
    name = PurePosixPath(raw.replace("\\", "/")).name if raw else ""
    if not name or name in {".", ".."}:
        return "image_clean"

    dot = name.rfind(".")
    if dot > 0 and dot < len(name) - 1:
        stem, ext = name[:dot], name[dot:]
    else:
        stem, ext = name, ""

    if _ALREADY_CLEAN_RE.search(stem):
        return f"{stem}{ext}"
    if stem.endswith(("_", "-")):
        return f"{stem}clean{ext}"
    return f"{stem}_clean{ext}"


def content_disposition_header(filename: str, *, disposition: str = "attachment") -> str:
    kind = "inline" if disposition == "inline" else "attachment"
    safe_name = re.sub(r"[\x00-\x1f\x7f]+", "_", str(filename or "download"))
    ascii_name = safe_name.encode("ascii", "replace").decode("ascii").replace("\\", "_").replace('"', "_")
    utf8_name = quote(safe_name, safe="")
    return f'{kind}; filename="{ascii_name}"; filename*=UTF-8\'\'{utf8_name}'


def strip_image_metadata(data: bytes, mime: str) -> bytes:
    """Strip workflow, prompt, EXIF, XMP, and text metadata from supported image formats."""
    normalized = (mime or "").strip().lower()
    if normalized == "image/png":
        return _strip_png(data)
    if normalized == "image/jpeg":
        return _strip_jpeg(data)
    if normalized == "image/webp":
        return _strip_webp(data)
    if normalized == "image/gif":
        return _strip_gif(data)
    raise ImageMetadataStripError("仅支持对 PNG / JPEG / WebP / GIF 图片去除元数据")


def _strip_png(data: bytes) -> bytes:
    total = len(data)
    if total < 8 or data[:8] != _PNG_SIGNATURE:
        raise ImageMetadataStripError("PNG 文件头无效")

    out = bytearray(_PNG_SIGNATURE)
    offset = 8
    chunks = 0
    saw_ihdr = False
    saw_idat = False
    saw_iend = False

    while offset < total:
        chunks += 1
        if chunks > _MAX_PNG_CHUNKS or offset + 12 > total:
            raise ImageMetadataStripError("PNG 数据块损坏或数量超限")
        length = struct.unpack_from(">I", data, offset)[0]
        ctype = data[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > total:
            raise ImageMetadataStripError("PNG 数据块被截断")

        if chunks == 1:
            if ctype != b"IHDR" or length != 13:
                raise ImageMetadataStripError("PNG 缺少有效的 IHDR 文件头")
            width, height = struct.unpack_from(">II", data, offset + 8)
            if width == 0 or height == 0:
                raise ImageMetadataStripError("PNG 图像尺寸无效")
            saw_ihdr = True
        elif ctype == b"IHDR":
            raise ImageMetadataStripError("PNG 包含重复的 IHDR 文件头")

        if ctype == b"IDAT":
            saw_idat = True

        if ctype == b"IEND":
            if length != 0:
                raise ImageMetadataStripError("PNG IEND 数据块无效")
            out.extend(data[offset:chunk_end])
            saw_iend = True
            break

        if ctype in _PNG_KEEP_CHUNKS:
            out.extend(data[offset:chunk_end])

        offset = chunk_end

    if not (saw_ihdr and saw_idat and saw_iend):
        raise ImageMetadataStripError("PNG 文件结构不完整")
    return bytes(out)


def _extract_jpeg_exif_orientation(payload: bytes) -> int | None:
    """Return EXIF Orientation (2..8) from an APP1 Exif payload, or None if upright/absent."""
    if not payload.startswith(b"Exif\x00\x00"):
        return None
    tiff = payload[6:]
    if len(tiff) < 8:
        return None
    byte_order = tiff[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None
    magic, ifd0 = struct.unpack_from(endian + "HI", tiff, 2)
    if magic != 42 or ifd0 < 8 or ifd0 + 2 > len(tiff):
        return None
    entry_count = struct.unpack_from(endian + "H", tiff, ifd0)[0]
    for i in range(min(entry_count, 512)):
        entry_off = ifd0 + 2 + i * 12
        if entry_off + 12 > len(tiff):
            break
        tag, field_type, count = struct.unpack_from(endian + "HHI", tiff, entry_off)
        if tag == 0x0112 and field_type == 3 and count == 1:
            orientation = struct.unpack_from(endian + "H", tiff, entry_off + 8)[0]
            if 2 <= orientation <= 8:
                return orientation
            return None
    return None


def _build_minimal_orientation_exif_segment(orientation: int) -> bytes:
    """Build a minimal synthetic APP1 EXIF segment carrying only the Orientation tag."""
    payload = (
        b"Exif\x00\x00"
        b"MM"
        + struct.pack(">HI", 42, 8)
        + struct.pack(">H", 1)
        + struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0)
        + struct.pack(">I", 0)
    )
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def _strip_jpeg(data: bytes) -> bytes:
    total = len(data)
    if total < 4 or data[:2] != b"\xff\xd8":
        raise ImageMetadataStripError("JPEG 文件头无效")

    out = bytearray(b"\xff\xd8")
    offset = 2
    saw_sos = False
    saw_eoi = False
    emitted_orientation = False

    while offset < total:
        if data[offset] != 0xFF:
            raise ImageMetadataStripError("JPEG 段标记结构无效")
        while offset < total and data[offset] == 0xFF:
            offset += 1
        if offset >= total:
            raise ImageMetadataStripError("JPEG 文件在标记处截断")

        marker = data[offset]
        offset += 1

        if marker == 0xD9:  # EOI
            out.extend(b"\xff\xd9")
            saw_eoi = True
            break
        if marker == 0xD8:  # Unexpected SOI
            raise ImageMetadataStripError("JPEG 包含异常的 SOI 标记")
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # TEM / RST0..RST7
            out.extend((0xFF, marker))
            continue

        if offset + 2 > total:
            raise ImageMetadataStripError("JPEG 段长度字段被截断")
        seg_len = struct.unpack_from(">H", data, offset)[0]
        seg_end = offset + seg_len
        if seg_len < 2 or seg_end > total:
            raise ImageMetadataStripError("JPEG 段数据被截断")

        if marker == 0xDA:  # SOS (Start of Scan)
            saw_sos = True
            out.extend((0xFF, marker))
            out.extend(data[offset:seg_end])
            # Scan entropy-coded bytes until the next real marker (0xFF followed by non-0x00 and non-RST).
            scan = seg_end
            next_marker_pos = -1
            while scan < total:
                ff_pos = data.find(b"\xff", scan)
                if ff_pos == -1:
                    raise ImageMetadataStripError("JPEG 图像扫描数据缺少结束标记")
                cursor = ff_pos + 1
                while cursor < total and data[cursor] == 0xFF:
                    cursor += 1
                if cursor >= total:
                    raise ImageMetadataStripError("JPEG 图像扫描数据被截断")
                next_byte = data[cursor]
                if next_byte == 0x00 or 0xD0 <= next_byte <= 0xD7:
                    scan = cursor + 1
                    continue
                next_marker_pos = ff_pos
                break
            if next_marker_pos < 0:
                raise ImageMetadataStripError("JPEG 图像扫描数据未正常结束")
            out.extend(data[seg_end:next_marker_pos])
            offset = next_marker_pos
            continue

        payload = data[offset + 2 : seg_end]
        keep = True
        if marker == 0xFE:  # COM comment
            keep = False
        elif 0xE0 <= marker <= 0xEF:  # APP0..APP15
            if marker == 0xE0:
                keep = payload.startswith(b"JFIF\x00")
            elif marker == 0xE1:
                keep = False
                if not emitted_orientation:
                    orientation = _extract_jpeg_exif_orientation(payload)
                    if orientation is not None:
                        out.extend(_build_minimal_orientation_exif_segment(orientation))
                        emitted_orientation = True
            elif marker == 0xE2:
                keep = payload.startswith(b"ICC_PROFILE\x00")
            elif marker == 0xEE:
                keep = payload.startswith(b"Adobe")
            else:
                keep = False

        if keep:
            out.extend((0xFF, marker))
            out.extend(data[offset:seg_end])

        offset = seg_end

    if not (saw_sos and saw_eoi):
        raise ImageMetadataStripError("JPEG 文件结构不完整")
    return bytes(out)


def _strip_webp(data: bytes) -> bytes:
    total = len(data)
    if total < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ImageMetadataStripError("WebP 文件头无效")

    riff_size = struct.unpack_from("<I", data, 4)[0]
    riff_end = 8 + riff_size
    if riff_size < 4 or riff_end > total:
        raise ImageMetadataStripError("WebP RIFF 长度无效或被截断")

    kept: list[tuple[bytes, bytes]] = []
    offset = 12
    chunks = 0
    while offset < riff_end:
        chunks += 1
        if chunks > _MAX_WEBP_CHUNKS or offset + 8 > riff_end:
            raise ImageMetadataStripError("WebP 数据块损坏或数量超限")
        fourcc = data[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]
        payload_start = offset + 8
        payload_end = payload_start + chunk_size
        if payload_end > riff_end:
            raise ImageMetadataStripError("WebP 数据块被截断")
        padded_end = payload_end + (chunk_size & 1)
        if padded_end > riff_end:
            padded_end = payload_end

        if fourcc in _WEBP_KEEP_CHUNKS:
            kept.append((fourcc, data[payload_start:payload_end]))

        offset = padded_end

    has_image_data = any(cc in {b"VP8 ", b"VP8L", b"ANMF"} for cc, _ in kept)
    if not has_image_data:
        raise ImageMetadataStripError("WebP 缺少图像数据块")

    rebuilt_chunks: list[tuple[bytes, bytes]] = []
    other_chunks = [entry for entry in kept if entry[0] != b"VP8X"]
    for fourcc, payload in kept:
        if fourcc == b"VP8X":
            if len(payload) != 10:
                raise ImageMetadataStripError("WebP VP8X 扩展头长度无效")
            flags = payload[0] & ~(_WEBP_VP8X_EXIF_FLAG | _WEBP_VP8X_XMP_FLAG)
            # If no extended features remain and we have a single simple VP8/VP8L payload,
            # emit a standard simple WebP without an empty VP8X header.
            if flags == 0 and len(other_chunks) == 1 and other_chunks[0][0] in {b"VP8 ", b"VP8L"}:
                continue
            rebuilt_chunks.append((fourcc, bytes((flags,)) + payload[1:]))
        else:
            rebuilt_chunks.append((fourcc, payload))

    body = bytearray()
    for fourcc, payload in rebuilt_chunks:
        body.extend(fourcc)
        body.extend(struct.pack("<I", len(payload)))
        body.extend(payload)
        if len(payload) & 1:
            body.append(0)

    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + bytes(body)


def _skip_gif_sub_blocks(data: bytes, offset: int) -> int:
    total = len(data)
    while True:
        if offset >= total:
            raise ImageMetadataStripError("GIF 子块被截断")
        sub_len = data[offset]
        offset += 1
        if sub_len == 0:
            return offset
        if offset + sub_len > total:
            raise ImageMetadataStripError("GIF 子块数据被截断")
        offset += sub_len


def _strip_gif(data: bytes) -> bytes:
    total = len(data)
    if total < 14 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        raise ImageMetadataStripError("GIF 文件头无效")

    packed = data[10]
    offset = 13
    if packed & 0x80:
        gct_size = 3 * (1 << ((packed & 0x07) + 1))
        if offset + gct_size > total:
            raise ImageMetadataStripError("GIF 全局色表被截断")
        offset += gct_size

    out = bytearray(data[:offset])
    saw_image = False
    saw_trailer = False
    blocks = 0

    while offset < total:
        blocks += 1
        if blocks > _MAX_GIF_BLOCKS:
            raise ImageMetadataStripError("GIF 数据块数量超限")
        introducer = data[offset]

        if introducer == 0x3B:  # Trailer
            out.append(0x3B)
            saw_trailer = True
            break

        if introducer == 0x2C:  # Image Descriptor
            if offset + 11 > total:
                raise ImageMetadataStripError("GIF 图像块被截断")
            block_start = offset
            img_packed = data[offset + 9]
            cursor = offset + 10
            if img_packed & 0x80:
                lct_size = 3 * (1 << ((img_packed & 0x07) + 1))
                cursor += lct_size
                if cursor + 1 > total:
                    raise ImageMetadataStripError("GIF 局部色表被截断")
            cursor += 1  # LZW minimum code size
            block_end = _skip_gif_sub_blocks(data, cursor)
            out.extend(data[block_start:block_end])
            saw_image = True
            offset = block_end
            continue

        if introducer == 0x21:  # Extension
            if offset + 2 > total:
                raise ImageMetadataStripError("GIF 扩展块被截断")
            block_start = offset
            label = data[offset + 1]
            if label == 0xF9:  # Graphic Control Extension (delay / transparency)
                block_end = _skip_gif_sub_blocks(data, offset + 2)
                out.extend(data[block_start:block_end])
                offset = block_end
                continue
            if label == 0xFF:  # Application Extension
                if offset + 3 > total:
                    raise ImageMetadataStripError("GIF 应用扩展块被截断")
                app_len = data[offset + 2]
                app_end = offset + 3 + app_len
                if app_end > total:
                    raise ImageMetadataStripError("GIF 应用扩展块头被截断")
                app_id = data[offset + 3 : app_end]
                block_end = _skip_gif_sub_blocks(data, app_end)
                if app_id in _GIF_KEEP_APP_IDS:
                    out.extend(data[block_start:block_end])
                offset = block_end
                continue
            # Strip Comment (0xFE), Plain Text (0x01), and any unknown extension blocks.
            offset = _skip_gif_sub_blocks(data, offset + 2)
            continue

        raise ImageMetadataStripError("GIF 块结构无效")

    if not (saw_image and saw_trailer):
        raise ImageMetadataStripError("GIF 文件结构不完整")
    return bytes(out)
