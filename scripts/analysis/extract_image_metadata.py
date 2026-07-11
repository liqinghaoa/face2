"""Extract file, image, EXIF, GPS, ICC, and XMP metadata from an image directory.

The script writes a JSON intermediate consumed by the spreadsheet builder. It does
not modify source images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import os
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}

ORIENTATION_TEXT = {
    1: "正常（0°）",
    2: "水平镜像",
    3: "旋转180°",
    4: "垂直镜像",
    5: "水平镜像后逆时针旋转90°",
    6: "顺时针旋转90°",
    7: "水平镜像后顺时针旋转90°",
    8: "逆时针旋转90°",
}

MAIN_HEADERS = [
    "序号", "ID", "文件名", "相对路径", "绝对路径", "扩展名",
    "文件大小(字节)", "文件大小(MB)", "文件创建时间", "文件修改时间", "SHA256",
    "图像格式", "MIME类型", "宽度(px)", "高度(px)", "长宽比", "总像素(MP)",
    "色彩模式", "通道数", "位深(每通道)", "是否动画", "帧数",
    "EXIF存在", "EXIF字节数", "ICC存在", "ICC字节数", "XMP存在", "XMP字节数",
    "DPI_X", "DPI_Y", "JFIF版本", "JFIF单位", "JFIF密度X", "JFIF密度Y",
    "方向值", "方向解释", "图像描述", "厂商", "相机型号", "相机序列号",
    "镜头厂商", "镜头型号", "镜头序列号", "软件", "作者", "版权",
    "文件内修改时间", "原始拍摄时间", "数字化时间", "时区偏移",
    "曝光时间(s)", "光圈F值", "曝光程序", "ISO", "快门速度值(APEX)",
    "光圈值(APEX)", "亮度值(APEX)", "曝光补偿(EV)", "最大光圈值(APEX)",
    "测光模式", "光源", "闪光灯", "焦距(mm)", "35mm等效焦距(mm)",
    "数字变焦比", "白平衡", "曝光模式", "场景拍摄类型", "增益控制",
    "对比度", "饱和度", "锐度", "色彩空间", "EXIF像素宽度", "EXIF像素高度",
    "GPS纬度", "GPS经度", "GPS海拔(m)", "GPS日期", "GPS时间",
    "用户注释", "Windows标题", "Windows主题", "Windows备注", "Windows关键字",
    "Windows作者", "提取状态", "错误信息",
]


def safe_number(value: Any) -> float | int | None:
    if value is None:
        return None
    try:
        result = float(value)
        if not math.isfinite(result):
            return None
        return int(result) if result.is_integer() else result
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def decode_bytes(value: bytes) -> str:
    payload = value
    encodings = ["utf-8", "utf-16le", "utf-16be", "gb18030", "latin-1"]
    if payload.startswith(b"ASCII\x00\x00\x00"):
        payload = payload[8:]
        encodings = ["ascii", "utf-8", "latin-1"]
    elif payload.startswith(b"UNICODE\x00"):
        payload = payload[8:]
        encodings = ["utf-16le", "utf-16be", "utf-8"]
    for encoding in encodings:
        try:
            decoded = payload.decode(encoding).rstrip("\x00").strip()
            if not decoded:
                return ""
            printable = sum(char.isprintable() or char in "\t\r\n" for char in decoded) / len(decoded)
            if printable >= 0.9:
                return decoded
        except UnicodeDecodeError:
            continue
    return payload.hex()


def to_json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return decode_bytes(value)
    if isinstance(value, (tuple, list)):
        return [to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    numeric = safe_number(value)
    return numeric if numeric is not None else str(value)


def display_value(value: Any) -> str:
    converted = to_json_value(value)
    if isinstance(converted, (dict, list)):
        text = json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
    else:
        text = "" if converted is None else str(converted)
    return text[:32760]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exif_groups(exif: Any) -> list[tuple[str, dict[int, Any], dict[int, str]]]:
    groups: list[tuple[str, dict[int, Any], dict[int, str]]] = [("IFD0", dict(exif.items()), ExifTags.TAGS)]
    ifd_enum = getattr(ExifTags, "IFD", None)
    if ifd_enum is None:
        return groups
    definitions = [
        ("ExifIFD", getattr(ifd_enum, "Exif", None), ExifTags.TAGS),
        ("GPSIFD", getattr(ifd_enum, "GPSInfo", None), ExifTags.GPSTAGS),
        ("InteropIFD", getattr(ifd_enum, "Interop", None), ExifTags.TAGS),
    ]
    for group_name, ifd_id, tag_names in definitions:
        if ifd_id is None:
            continue
        try:
            values = exif.get_ifd(ifd_id)
        except Exception:
            values = {}
        if values:
            groups.append((group_name, dict(values), tag_names))
    return groups


def flatten_dict(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(flatten_dict(child, item))
        return rows
    if isinstance(value, (list, tuple)):
        return [(prefix, value)]
    return [(prefix, value)]


def parse_xmp(value: Any) -> list[tuple[str, str]]:
    if not value:
        return []
    try:
        payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        root = ET.fromstring(payload)
    except Exception:
        return [("raw_packet", display_value(value))]
    rows: list[tuple[str, str]] = []
    for element in root.iter():
        tag = element.tag.split("}", 1)[-1]
        text = (element.text or "").strip()
        if text:
            rows.append((tag, text))
        for attr, attr_value in element.attrib.items():
            attr_name = attr.split("}", 1)[-1]
            rows.append((f"{tag}.@{attr_name}", attr_value))
    return rows


def payload_size(value: Any) -> int:
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return 0


def dms_to_decimal(value: Any, ref: Any) -> float | None:
    try:
        parts = [float(item) for item in value]
        decimal = parts[0] + parts[1] / 60 + parts[2] / 3600
        if str(ref).upper() in {"S", "W"}:
            decimal = -decimal
        return round(decimal, 8)
    except (TypeError, ValueError, ZeroDivisionError, IndexError):
        return None


def bits_per_channel(mode: str) -> int | None:
    if mode == "1":
        return 1
    if ";16" in mode:
        return 16
    if mode in {"I", "F"}:
        return 32
    if mode:
        return 8
    return None


def get_first(values: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in values and values[name] not in (None, "", b""):
            return to_json_value(values[name])
    return None


def iso_value(values: dict[str, Any]) -> Any:
    value = get_first(values, "PhotographicSensitivity", "ISOSpeedRatings", "ISOSpeed")
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def extract_one(path: Path, root: Path, index: int) -> tuple[dict[str, Any], list[list[Any]]]:
    stat = path.stat()
    row = {header: None for header in MAIN_HEADERS}
    row.update({
        "序号": index,
        "ID": path.stem,
        "文件名": path.name,
        "相对路径": str(path.relative_to(root)),
        "绝对路径": str(path.resolve()),
        "扩展名": path.suffix.lower(),
        "文件大小(字节)": stat.st_size,
        "文件大小(MB)": round(stat.st_size / 1024 / 1024, 4),
        "文件创建时间": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "文件修改时间": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "SHA256": sha256_file(path),
        "提取状态": "成功",
        "错误信息": "",
    })
    details: list[list[Any]] = []
    try:
        with Image.open(path) as image:
            width, height = image.size
            info = dict(image.info)
            exif = image.getexif()
            exif_blob = info.get("exif", b"")
            icc_blob = info.get("icc_profile", b"")
            xmp_blob = info.get("xmp") or info.get("XML:com.adobe.xmp") or b""
            dpi = info.get("dpi") or (None, None)
            jfif_density = info.get("jfif_density") or (None, None)
            jfif_version = info.get("jfif_version")
            mime_type = Image.MIME.get(image.format) or mimetypes.guess_type(path.name)[0]
            row.update({
                "图像格式": image.format,
                "MIME类型": mime_type,
                "宽度(px)": width,
                "高度(px)": height,
                "长宽比": round(width / height, 6) if height else None,
                "总像素(MP)": round(width * height / 1_000_000, 4),
                "色彩模式": image.mode,
                "通道数": len(image.getbands()),
                "位深(每通道)": bits_per_channel(image.mode),
                "是否动画": "是" if getattr(image, "is_animated", False) else "否",
                "帧数": getattr(image, "n_frames", 1),
                "EXIF存在": "是" if exif else "否",
                "EXIF字节数": len(exif_blob) if isinstance(exif_blob, bytes) else 0,
                "ICC存在": "是" if icc_blob else "否",
                "ICC字节数": len(icc_blob) if isinstance(icc_blob, bytes) else 0,
                "XMP存在": "是" if xmp_blob else "否",
                "XMP字节数": payload_size(xmp_blob),
                "DPI_X": safe_number(dpi[0]) if len(dpi) > 0 else None,
                "DPI_Y": safe_number(dpi[1]) if len(dpi) > 1 else None,
                "JFIF版本": ".".join(str(part) for part in jfif_version) if jfif_version else None,
                "JFIF单位": info.get("jfif_unit"),
                "JFIF密度X": safe_number(jfif_density[0]) if len(jfif_density) > 0 else None,
                "JFIF密度Y": safe_number(jfif_density[1]) if len(jfif_density) > 1 else None,
            })

            values: dict[str, Any] = {}
            seen: set[tuple[str, int, str, str]] = set()
            gps_values: dict[str, Any] = {}
            for group_name, group, name_map in exif_groups(exif):
                for tag_id, value in group.items():
                    tag_name = name_map.get(tag_id, f"Tag_{tag_id}")
                    if tag_name in {"ExifOffset", "GPSInfo", "InteropOffset"}:
                        continue
                    values.setdefault(tag_name, value)
                    if group_name == "GPSIFD":
                        gps_values[tag_name] = value
                    shown = display_value(value)
                    key = (group_name, int(tag_id), tag_name, shown)
                    if key not in seen:
                        details.append([path.stem, path.name, group_name, int(tag_id), tag_name, shown])
                        seen.add(key)

            for key, value in info.items():
                if key in {"exif", "icc_profile", "xmp", "XML:com.adobe.xmp"}:
                    continue
                details.append([path.stem, path.name, "PillowInfo", "", str(key), display_value(value)])

            for name, value in parse_xmp(xmp_blob):
                details.append([path.stem, path.name, "XMP", "", name, display_value(value)])

            orientation = safe_number(get_first(values, "Orientation"))
            gps_lat = dms_to_decimal(gps_values.get("GPSLatitude"), gps_values.get("GPSLatitudeRef"))
            gps_lon = dms_to_decimal(gps_values.get("GPSLongitude"), gps_values.get("GPSLongitudeRef"))
            gps_alt = safe_number(gps_values.get("GPSAltitude"))
            if gps_alt is not None and safe_number(gps_values.get("GPSAltitudeRef")) == 1:
                gps_alt = -gps_alt
            gps_time = gps_values.get("GPSTimeStamp")
            if gps_time:
                try:
                    gps_time = ":".join(f"{int(float(item)):02d}" for item in gps_time)
                except Exception:
                    gps_time = display_value(gps_time)

            field_map = {
                "方向值": orientation,
                "方向解释": ORIENTATION_TEXT.get(int(orientation), "未知") if orientation is not None else None,
                "图像描述": get_first(values, "ImageDescription"),
                "厂商": get_first(values, "Make"),
                "相机型号": get_first(values, "Model"),
                "相机序列号": get_first(values, "BodySerialNumber", "CameraSerialNumber"),
                "镜头厂商": get_first(values, "LensMake"),
                "镜头型号": get_first(values, "LensModel"),
                "镜头序列号": get_first(values, "LensSerialNumber"),
                "软件": get_first(values, "Software"),
                "作者": get_first(values, "Artist"),
                "版权": get_first(values, "Copyright"),
                "文件内修改时间": get_first(values, "DateTime"),
                "原始拍摄时间": get_first(values, "DateTimeOriginal"),
                "数字化时间": get_first(values, "DateTimeDigitized"),
                "时区偏移": get_first(values, "OffsetTimeOriginal", "OffsetTime"),
                "曝光时间(s)": safe_number(get_first(values, "ExposureTime")),
                "光圈F值": safe_number(get_first(values, "FNumber")),
                "曝光程序": get_first(values, "ExposureProgram"),
                "ISO": iso_value(values),
                "快门速度值(APEX)": safe_number(get_first(values, "ShutterSpeedValue")),
                "光圈值(APEX)": safe_number(get_first(values, "ApertureValue")),
                "亮度值(APEX)": safe_number(get_first(values, "BrightnessValue")),
                "曝光补偿(EV)": safe_number(get_first(values, "ExposureBiasValue")),
                "最大光圈值(APEX)": safe_number(get_first(values, "MaxApertureValue")),
                "测光模式": get_first(values, "MeteringMode"),
                "光源": get_first(values, "LightSource"),
                "闪光灯": get_first(values, "Flash"),
                "焦距(mm)": safe_number(get_first(values, "FocalLength")),
                "35mm等效焦距(mm)": safe_number(get_first(values, "FocalLengthIn35mmFilm")),
                "数字变焦比": safe_number(get_first(values, "DigitalZoomRatio")),
                "白平衡": get_first(values, "WhiteBalance"),
                "曝光模式": get_first(values, "ExposureMode"),
                "场景拍摄类型": get_first(values, "SceneCaptureType"),
                "增益控制": get_first(values, "GainControl"),
                "对比度": get_first(values, "Contrast"),
                "饱和度": get_first(values, "Saturation"),
                "锐度": get_first(values, "Sharpness"),
                "色彩空间": get_first(values, "ColorSpace"),
                "EXIF像素宽度": safe_number(get_first(values, "ExifImageWidth", "PixelXDimension")),
                "EXIF像素高度": safe_number(get_first(values, "ExifImageHeight", "PixelYDimension")),
                "GPS纬度": gps_lat,
                "GPS经度": gps_lon,
                "GPS海拔(m)": gps_alt,
                "GPS日期": get_first(gps_values, "GPSDateStamp"),
                "GPS时间": gps_time,
                "用户注释": get_first(values, "UserComment"),
                "Windows标题": get_first(values, "XPTitle"),
                "Windows主题": get_first(values, "XPSubject"),
                "Windows备注": get_first(values, "XPComment"),
                "Windows关键字": get_first(values, "XPKeywords"),
                "Windows作者": get_first(values, "XPAuthor"),
            }
            row.update(field_map)
    except Exception as exc:
        row["提取状态"] = "失败"
        row["错误信息"] = f"{type(exc).__name__}: {exc}"[:1000]
    return row, details


def field_definitions() -> list[list[str]]:
    definitions = {
        "SHA256": ("文件校验", "文件内容的 SHA-256 摘要，可用于确认文件是否完全相同。", "十六进制文本"),
        "长宽比": ("图像属性", "宽度除以高度。", "无量纲"),
        "总像素(MP)": ("图像属性", "宽度×高度，以百万像素表示。", "MP"),
        "色彩模式": ("图像属性", "Pillow 解码后的像素模式，如 RGB、RGBA、L。", "文本"),
        "位深(每通道)": ("图像属性", "每个颜色通道的估计位数；常规 JPEG 通常为 8。", "bit"),
        "EXIF存在": ("元数据容器", "图片是否包含可解析的 EXIF 标签。", "是/否"),
        "ICC存在": ("元数据容器", "图片是否嵌入 ICC 色彩配置文件。", "是/否"),
        "XMP存在": ("元数据容器", "图片是否包含 XMP 扩展元数据。", "是/否"),
        "DPI_X": ("分辨率", "水平方向打印分辨率；不改变实际像素数。", "dpi"),
        "DPI_Y": ("分辨率", "垂直方向打印分辨率；不改变实际像素数。", "dpi"),
        "JFIF单位": ("JPEG/JFIF", "JFIF 密度单位：0=无单位，1=dpi，2=dpcm。", "编码值"),
        "方向值": ("EXIF", "EXIF Orientation，表示查看时需要进行的旋转或镜像。", "1–8"),
        "文件内修改时间": ("EXIF", "EXIF DateTime，设备写入的图像文件修改时间，可能没有时区。", "日期时间文本"),
        "原始拍摄时间": ("EXIF", "EXIF DateTimeOriginal，通常表示快门拍摄时刻。", "日期时间文本"),
        "数字化时间": ("EXIF", "EXIF DateTimeDigitized，图像被数字化或生成的时间。", "日期时间文本"),
        "曝光时间(s)": ("拍摄参数", "感光元件曝光持续时间。", "秒"),
        "光圈F值": ("拍摄参数", "镜头焦距与有效孔径之比；数值越小，光圈通常越大。", "F-number"),
        "ISO": ("拍摄参数", "相机感光度设置。", "ISO"),
        "快门速度值(APEX)": ("拍摄参数", "APEX 快门速度值，通常满足曝光时间约为 2^(-Tv) 秒。", "APEX"),
        "光圈值(APEX)": ("拍摄参数", "APEX 光圈值，通常满足 F 值约为 2^(Av/2)。", "APEX"),
        "亮度值(APEX)": ("拍摄参数", "相机记录的场景亮度 APEX 值。", "APEX"),
        "曝光补偿(EV)": ("拍摄参数", "相对自动测光结果的人为曝光补偿。", "EV"),
        "测光模式": ("拍摄参数", "相机计算曝光时采用的测光方式编码。", "EXIF编码"),
        "闪光灯": ("拍摄参数", "闪光灯是否触发及相关状态的 EXIF 位掩码值。", "EXIF编码"),
        "焦距(mm)": ("拍摄参数", "拍摄时镜头实际焦距。", "mm"),
        "35mm等效焦距(mm)": ("拍摄参数", "换算到 35mm 全画幅视角后的等效焦距。", "mm"),
        "白平衡": ("拍摄参数", "白平衡模式编码，常见 0=自动、1=手动。", "EXIF编码"),
        "曝光模式": ("拍摄参数", "曝光控制模式编码，常见 0=自动、1=手动、2=包围曝光。", "EXIF编码"),
        "场景拍摄类型": ("拍摄参数", "场景模式编码，常见 0=标准、1=风景、2=人像、3=夜景。", "EXIF编码"),
        "色彩空间": ("EXIF", "EXIF 色彩空间编码，常见 1=sRGB、65535=未校准。", "EXIF编码"),
        "GPS纬度": ("GPS", "由 GPS 度分秒与南北纬参考转换得到的十进制度。", "度"),
        "GPS经度": ("GPS", "由 GPS 度分秒与东西经参考转换得到的十进制度。", "度"),
        "GPS海拔(m)": ("GPS", "相对海平面的海拔；依据 GPSAltitudeRef 修正正负。", "m"),
        "提取状态": ("质量控制", "图片能否被成功打开并完成元数据解析。", "成功/失败"),
        "错误信息": ("质量控制", "提取失败时记录异常类型和消息。", "文本"),
    }
    rows: list[list[str]] = []
    for header in MAIN_HEADERS:
        category, explanation, unit = definitions.get(header, ("基础字段", f"逐图片记录的“{header}”字段。", "文本或数值"))
        rows.append([header, category, explanation, unit])
    rows.extend([
        ["EXIF原始明细.元数据组", "原始明细", "标签所属容器或 IFD：IFD0、ExifIFD、GPSIFD、InteropIFD、PillowInfo、XMP。", "文本"],
        ["EXIF原始明细.标签ID", "原始明细", "EXIF 数字标签 ID；非 EXIF 容器可能为空。", "整数"],
        ["EXIF原始明细.标签名", "原始明细", "标准标签名或解析器返回的属性名。", "文本"],
        ["EXIF原始明细.标签值", "原始明细", "原始标签值的可读文本表示；长值最多保留 32760 个字符。", "文本"],
    ])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    root = args.image_dir.resolve()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path.relative_to(root)).lower(),
    )
    rows: list[list[Any]] = []
    raw_details: list[list[Any]] = []
    for index, path in enumerate(files, start=1):
        row, details = extract_one(path, root, index)
        rows.append([row[header] for header in MAIN_HEADERS])
        raw_details.extend(details)

    positions = {name: idx for idx, name in enumerate(MAIN_HEADERS)}
    summary = {
        "image_dir": str(root),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(rows),
        "successful": sum(row[positions["提取状态"]] == "成功" for row in rows),
        "failed": sum(row[positions["提取状态"]] == "失败" for row in rows),
        "exif_files": sum(row[positions["EXIF存在"]] == "是" for row in rows),
        "icc_files": sum(row[positions["ICC存在"]] == "是" for row in rows),
        "xmp_files": sum(row[positions["XMP存在"]] == "是" for row in rows),
        "gps_files": sum(row[positions["GPS纬度"]] is not None and row[positions["GPS经度"]] is not None for row in rows),
        "extensions": dict(Counter(row[positions["扩展名"]] for row in rows)),
        "raw_detail_count": len(raw_details),
    }
    payload = {
        "summary": summary,
        "main_headers": MAIN_HEADERS,
        "main_rows": rows,
        "raw_headers": ["ID", "文件名", "元数据组", "标签ID", "标签名", "标签值"],
        "raw_rows": raw_details,
        "definition_headers": ["字段名", "类别", "含义", "单位/取值"],
        "definition_rows": field_definitions(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
