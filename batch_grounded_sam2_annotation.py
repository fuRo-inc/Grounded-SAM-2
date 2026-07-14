#!/usr/bin/env python3
# coding: utf-8

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from PIL import Image, UnidentifiedImageError
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from supervision.draw.color import ColorPalette
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from utils.supervision_utils import CUSTOM_COLOR_MAP


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# Grounding DINOの出力ラベルを最終ラベルへ統合する
CLASS_MAP = {
    "person": "person",
    "people": "person",
    "pedestrian": "person",

    "car": "car",
    "truck": "car",
    "bus": "car",
    "motorcycle": "car",
    "motorbike": "car",
    "bicycle": "car",
    "bike": "car",
    "vehicle": "car",
}

# CVATなどで利用する最終クラスID
FINAL_CLASS_IDS = {
    "person": 0,
    "car": 1, # "vehicle": 1,
}

def dewarp_cylindrical(
    out_w: int,
    cx_fish: float,
    cy_fish: float,
    hfov_deg: float = 180.0,
    vfov_deg: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    hfov = np.radians(hfov_deg)
    vfov = np.radians(vfov_deg)

    f_cyl = out_w / hfov
    out_h = int(
        np.round(
            2.0 * f_cyl * np.tan(vfov / 2.0)
        )
    )

    out_shape = (out_h, out_w)

    cx = out_w / 2.0
    cy = out_h / 2.0

    u = np.arange(
        out_w,
        dtype=np.float32,
    ) + 0.5

    v = np.arange(
        out_h,
        dtype=np.float32,
    ) + 0.5

    uu, vv = np.meshgrid(u, v)

    yaw = (uu - cx) / f_cyl
    t = (vv - cy) / f_cyl

    r_h = 1.0 / np.sqrt(1.0 + t * t)

    x = r_h * np.sin(yaw)
    y = -t * r_h
    z = r_h * np.cos(yaw)

    theta = np.arccos(
        np.clip(z, -1.0, 1.0)
    )

    angle = np.arctan2(y, x)

    f_fish = (
        2.0
        * min(cx_fish, cy_fish)
        / np.pi
    )

    radius = f_fish * theta

    map_x = cx_fish + radius * np.cos(angle)
    map_y = cy_fish - radius * np.sin(angle)

    return (
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        out_shape,
    )

class FixedCylindricalDecoder:
    def __init__(
        self,
        fisheye_width: int = 2192,
        fisheye_height: int = 2192,
        cylindrical_width: int = 2192,
        horizontal_fov: float = 180.0,
        vertical_fov: float = 81.65,
        output_width: int = 1280,
        output_height: int = 704,
    ) -> None:
        self.fisheye_width = fisheye_width
        self.fisheye_height = fisheye_height

        self.cylindrical_width = cylindrical_width
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov

        self.output_width = output_width
        self.output_height = output_height

        self.map_x, self.map_y, self.cylindrical_shape = (
            dewarp_cylindrical(
                out_w=self.cylindrical_width,
                cx_fish=self.fisheye_width / 2.0,
                cy_fish=self.fisheye_height / 2.0,
                hfov_deg=self.horizontal_fov,
                vfov_deg=self.vertical_fov,
            )
        )

        self.expected_input_size = (
            self.fisheye_width,
            self.fisheye_height,
        )

    def decode(
        self,
        image_rgb: np.ndarray,
    ) -> np.ndarray:
        input_height, input_width = image_rgb.shape[:2]

        if (
            input_width != self.fisheye_width
            or input_height != self.fisheye_height
        ):
            raise ValueError(
                "入力画像サイズが想定と異なります。"
                f" expected="
                f"{self.fisheye_width}x{self.fisheye_height},"
                f" actual={input_width}x{input_height}"
            )

        cylindrical_image = cv2.remap(
            image_rgb,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        resized_image = cv2.resize(
            cylindrical_image,
            (
                self.output_width,
                self.output_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        if resized_image.shape[:2] != (
            self.output_height,
            self.output_width,
        ):
            raise RuntimeError(
                "円筒画像の出力サイズが想定と異なります。"
                f" actual={resized_image.shape}"
            )

        return resized_image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grounding DINOとSAM 2を使用して人物・車両を一括検出し、"
            "車種ラベルをcarへ統合します。"
        )
    )

    parser.add_argument(
        "--dewarp-fisheye",
        action="store_true",
        help=(
            "2192x2192の魚眼画像を円筒展開し、"
            "1280x704へリサイズしてから推論します。"
        ),
    )

    parser.add_argument(
        "--save-decoded-images",
        action="store_true",
        help="円筒展開後の画像も保存します。",
    )

    parser.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face上のGrounding DINOモデルID",
    )

    parser.add_argument(
        "--text-prompt",
        default="person. car. truck. bus. motorcycle. bicycle.",
        help=(
            "Grounding DINOへ渡すプロンプト。"
            "小文字かつ各クラスをピリオドで区切ることを推奨します。"
        ),
    )

    parser.add_argument(
        "--img-path",
        type=Path,
        default=Path("../dataset_output/selected"),
        help="入力画像または入力画像ディレクトリ",
    )

    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=Path("./checkpoints/sam2.1_hiera_large.pt"),
        help="SAM 2のチェックポイント",
    )

    parser.add_argument(
        "--sam2-model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM 2のモデル設定ファイル",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/person_car"),
        help="出力ディレクトリ",
    )

    # parser.add_argument(
    #     "--box-threshold",
    #     type=float,
    #     default=0.25,
    #     help="Grounding DINOのBBox検出しきい値",
    # )

    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.20,
        help="personに適用するGrounding DINOスコアしきい値",
    )

    parser.add_argument(
        "--car-threshold",
        type=float,
        default=0.30,
        help="car系クラスに適用するGrounding DINOスコアしきい値",
    )

    parser.add_argument(
        "--overlap-iou-threshold",
        type=float,
        default=0.70,
        help=(
            "同じ最終クラスのBBox同士がこのIoU以上で重なる場合、"
            "スコアの低い方を削除する"
        ),
    )

    parser.add_argument(
        "--text-threshold",
        type=float,
        default=0.20,
        help="Grounding DINOのテキスト一致しきい値",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="入力ディレクトリ配下を再帰的に検索する",
    )

    parser.add_argument(
        "--no-dump-json",
        action="store_true",
        help="JSONを保存しない",
    )

    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="BBoxおよびマスクの可視化画像を保存しない",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="CUDAが利用可能でもCPUを使用する",
    )

    parser.add_argument(
        "--keep-unknown-classes",
        action="store_true",
        help=(
            "CLASS_MAPに存在しないラベルも、その名前のまま保存する。"
            "通常は指定しないことを推奨します。"
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    # if not 0.0 <= args.box_threshold <= 1.0:
    #     raise ValueError(
    #         "--box-thresholdは0.0から1.0の範囲にしてください。"
    #     )

    if not 0.0 <= args.person_threshold <= 1.0:
        raise ValueError(
            "--person-thresholdは0.0から1.0の範囲にしてください。"
        )
    
    if not 0.0 <= args.car_threshold <= 1.0:
        raise ValueError(
            "--car-thresholdは0.0から1.0の範囲にしてください。"
        )
    
    if not 0.0 <= args.overlap_iou_threshold <= 1.0:
        raise ValueError(
            "--overlap-iou-thresholdは0.0から1.0の範囲にしてください。"
        )
    
    if not 0.0 <= args.text_threshold <= 1.0:
        raise ValueError(
            "--text-thresholdは0.0から1.0の範囲にしてください。"
        )

    if not args.img_path.exists():
        raise FileNotFoundError(
            f"入力パスが見つかりません: {args.img_path}"
        )

    if not args.sam2_checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM 2チェックポイントが見つかりません: "
            f"{args.sam2_checkpoint}"
        )


def collect_image_paths(
    input_path: Path,
    recursive: bool,
) -> list[Path]:
    input_path = input_path.expanduser().resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"対応していない画像形式です: {input_path}"
            )
        return [input_path]

    iterator = (
        input_path.rglob("*")
        if recursive
        else input_path.glob("*")
    )

    image_paths = sorted(
        path.resolve()
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"画像が見つかりませんでした: {input_path}"
        )

    return image_paths


def normalize_class_name(class_name: str) -> str:
    """
    Grounding DINOから返るラベル表記を正規化する。

    例:
      "Car"       -> "car"
      "truck."    -> "truck"
      "a person"  -> "person"
    """

    normalized = class_name.strip().lower()
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"^(a|an|the)\s+", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)

    return normalized


def map_class_name(
    original_class_name: str,
    keep_unknown_classes: bool,
) -> str | None:
    normalized = normalize_class_name(original_class_name)

    mapped_class_name = CLASS_MAP.get(normalized)

    if mapped_class_name is not None:
        return mapped_class_name

    if keep_unknown_classes:
        return normalized

    return None


def single_mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """
    2次元バイナリマスクをCOCO RLE形式へ変換する。
    """

    binary_mask = np.asarray(mask, dtype=np.uint8)

    rle = mask_util.encode(
        np.asfortranarray(binary_mask[:, :, None])
    )[0]

    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def make_output_stem(
    image_path: Path,
    input_root: Path,
) -> str:
    """
    サブディレクトリに同名画像が存在しても衝突しにくい名前を作る。
    """

    if input_root.is_dir():
        try:
            relative_path = image_path.relative_to(input_root)
        except ValueError:
            relative_path = Path(image_path.name)
    else:
        relative_path = Path(image_path.name)

    name_without_suffix = relative_path.with_suffix("").as_posix()

    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "__",
        name_without_suffix,
    )

    return safe_name


def create_autocast_context(device: str):
    """
    CUDA時のみbfloat16 autocastを使用する。

    GPUによってbfloat16が適さない場合はfloat16へフォールバックする。
    """

    if device != "cuda":
        return nullcontext()

    if torch.cuda.is_bf16_supported():
        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        )

    return torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    )


def setup_cuda_options(device: str) -> None:
    if device != "cuda":
        return

    major = torch.cuda.get_device_properties(0).major

    if major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def build_models(
    grounding_model_id: str,
    sam2_model_config: str,
    sam2_checkpoint: Path,
    device: str,
):
    print("[INFO] SAM 2モデルをロードします。")

    sam2_model = build_sam2(
        sam2_model_config,
        str(sam2_checkpoint),
        device=device,
    )

    sam2_predictor = SAM2ImagePredictor(sam2_model)

    print(
        f"[INFO] Grounding DINOをロードします: "
        f"{grounding_model_id}"
    )

    processor = AutoProcessor.from_pretrained(
        grounding_model_id
    )

    grounding_model = (
        AutoModelForZeroShotObjectDetection
        .from_pretrained(grounding_model_id)
        .to(device)
    )

    grounding_model.eval()

    return sam2_predictor, processor, grounding_model

def compute_iou(
    box_a: np.ndarray,
    box_b: np.ndarray,
) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection_area = intersection_width * intersection_height

    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(
        0.0,
        float(box_a[3] - box_a[1]),
    )

    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(
        0.0,
        float(box_b[3] - box_b[1]),
    )

    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area

def class_aware_nms(
    boxes: np.ndarray,
    scores: list[float],
    mapped_names: list[str],
    original_names: list[str],
    iou_threshold: float,
) -> tuple[
    np.ndarray,
    list[float],
    list[str],
    list[str],
]:
    """
    同じ最終クラス内でのみNMSを行う。

    person同士、car同士は重複除去するが、
    personとcarの重なりは削除しない。
    """

    if len(boxes) == 0:
        return boxes, scores, original_names, mapped_names

    kept_indices: list[int] = []

    for class_name in sorted(set(mapped_names)):
        class_indices = [
            index
            for index, name in enumerate(mapped_names)
            if name == class_name
        ]

        # スコアの高い順
        class_indices.sort(
            key=lambda index: scores[index],
            reverse=True,
        )

        while class_indices:
            best_index = class_indices.pop(0)
            kept_indices.append(best_index)

            remaining_indices: list[int] = []

            for candidate_index in class_indices:
                iou = compute_iou(
                    boxes[best_index],
                    boxes[candidate_index],
                )

                if iou < iou_threshold:
                    remaining_indices.append(
                        candidate_index
                    )

            class_indices = remaining_indices

    kept_indices.sort()

    return (
        boxes[kept_indices],
        [scores[index] for index in kept_indices],
        [original_names[index] for index in kept_indices],
        [mapped_names[index] for index in kept_indices],
    )


def filter_and_map_detections(
    boxes: np.ndarray,
    scores: list[float],
    class_names: list[str],
    keep_unknown_classes: bool,
    person_threshold: float,
    car_threshold: float,
    overlap_iou_threshold: float,
) -> tuple[
    np.ndarray,
    list[float],
    list[str],
    list[str],
]:
    filtered_boxes: list[np.ndarray] = []
    filtered_scores: list[float] = []
    original_names: list[str] = []
    mapped_names: list[str] = []

    for box, score, original_name in zip(
        boxes,
        scores,
        class_names,
    ):
        mapped_name = map_class_name(
            original_name,
            keep_unknown_classes=keep_unknown_classes,
        )

        if mapped_name is None:
            continue

        score = float(score)

        if mapped_name == "person" and score < person_threshold:
            continue

        if mapped_name == "car" and score < car_threshold:
            continue

        filtered_boxes.append(box)
        filtered_scores.append(score)
        original_names.append(
            normalize_class_name(original_name)
        )
        mapped_names.append(mapped_name)

    if not filtered_boxes:
        return (
            np.empty((0, 4), dtype=np.float32),
            [],
            [],
            [],
        )

    filtered_boxes_array = np.asarray(
        filtered_boxes,
        dtype=np.float32,
    )

    return class_aware_nms(
        boxes=filtered_boxes_array,
        scores=filtered_scores,
        original_names=original_names,
        mapped_names=mapped_names,
        iou_threshold=overlap_iou_threshold,
    )

def save_visualizations(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    mapped_class_names: list[str],
    original_class_names: list[str],
    grounding_scores: list[float],
    output_stem: str,
    box_output_dir: Path,
    mask_output_dir: Path,
) -> None:
    if len(boxes) == 0:
        cv2.imwrite(
            str(box_output_dir / f"{output_stem}.jpg"),
            image_bgr,
        )
        cv2.imwrite(
            str(mask_output_dir / f"{output_stem}.jpg"),
            image_bgr,
        )
        return

    class_ids = np.array(
        [
            FINAL_CLASS_IDS.get(class_name, index + 2)
            for index, class_name
            in enumerate(mapped_class_names)
        ],
        dtype=np.int32,
    )

    detections = sv.Detections(
        xyxy=boxes,
        mask=masks.astype(bool),
        class_id=class_ids,
    )

    labels = [
        (
            f"{mapped_name} "
            f"({original_name}) "
            f"{score:.2f}"
        )
        for mapped_name, original_name, score in zip(
            mapped_class_names,
            original_class_names,
            grounding_scores,
        )
    ]

    color_palette = ColorPalette.from_hex(
        CUSTOM_COLOR_MAP
    )

    box_annotator = sv.BoxAnnotator(
        color=color_palette
    )

    label_annotator = sv.LabelAnnotator(
        color=color_palette
    )

    mask_annotator = sv.MaskAnnotator(
        color=color_palette
    )

    box_frame = box_annotator.annotate(
        scene=image_bgr.copy(),
        detections=detections,
    )

    box_frame = label_annotator.annotate(
        scene=box_frame,
        detections=detections,
        labels=labels,
    )

    cv2.imwrite(
        str(box_output_dir / f"{output_stem}.jpg"),
        box_frame,
    )

    mask_frame = mask_annotator.annotate(
        scene=box_frame.copy(),
        detections=detections,
    )

    cv2.imwrite(
        str(mask_output_dir / f"{output_stem}.jpg"),
        mask_frame,
    )


def process_one_image(
    image_path: Path,
    input_root: Path,
    text_prompt: str,
    box_threshold: float,
    text_threshold: float,
    person_threshold: float,
    car_threshold: float,
    overlap_iou_threshold: float,
    keep_unknown_classes: bool,
    device: str,
    sam2_predictor: SAM2ImagePredictor,
    processor,
    grounding_model,
    output_dir: Path,
    dump_json: bool,
    save_visualization: bool,
    fisheye_decoder: FixedCylindricalDecoder | None,
    save_decoded_images: bool,
) -> dict[str, Any]:
    output_stem = make_output_stem(
        image_path=image_path,
        input_root=input_root,
    )

    try:
        with Image.open(image_path) as pil_image:
            original_image_rgb = np.array(
                pil_image.convert("RGB")
            )
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(
            f"画像を読み込めません: {exc}"
        ) from exc

    if fisheye_decoder is not None:
        image_array = fisheye_decoder.decode(
            original_image_rgb
        )
    else:
        image_array = original_image_rgb

    image_height, image_width = image_array.shape[:2]

    image_rgb = Image.fromarray(image_array)

    if save_decoded_images:
        decoded_bgr = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR,
        )

        decoded_path = (
            output_dir
            / "decoded"
            / f"{output_stem}.jpg"
        )

        success = cv2.imwrite(
            str(decoded_path),
            decoded_bgr,
        )

        if not success:
            raise RuntimeError(
                f"補正後画像を保存できません: {decoded_path}"
            )

    image_bgr = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2BGR,
    )

    sam2_predictor.set_image(image_array)

    inputs = processor(
        images=image_rgb,
        text=text_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        with create_autocast_context(device):
            grounding_outputs = grounding_model(**inputs)

    grounding_results = (
        processor.post_process_grounded_object_detection(
            grounding_outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(image_height, image_width)],
        )
    )

    raw_result = grounding_results[0]

    raw_boxes = (
        raw_result["boxes"]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    raw_scores = (
        raw_result["scores"]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )

    raw_class_names = list(raw_result["labels"])

    (
        input_boxes,
        grounding_scores,
        original_class_names,
        mapped_class_names,
    ) = filter_and_map_detections(
        boxes=raw_boxes,
        scores=raw_scores,
        class_names=raw_class_names,
        keep_unknown_classes=keep_unknown_classes,
        person_threshold=person_threshold,
        car_threshold=car_threshold,
        overlap_iou_threshold=overlap_iou_threshold,
    )

    if len(input_boxes) > 0:
        with torch.inference_mode():
            with create_autocast_context(device):
                masks, sam_scores, _ = sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes,
                    multimask_output=False,
                )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        masks = np.asarray(masks, dtype=bool)

        sam_scores_list = (
            np.asarray(sam_scores)
            .reshape(-1)
            .astype(float)
            .tolist()
        )
    else:
        masks = np.empty(
            (0, image_height, image_width),
            dtype=bool,
        )
        sam_scores_list = []

    if save_visualization:
        save_visualizations(
            image_bgr=image_bgr,
            boxes=input_boxes,
            masks=masks,
            mapped_class_names=mapped_class_names,
            original_class_names=original_class_names,
            grounding_scores=grounding_scores,
            output_stem=output_stem,
            box_output_dir=output_dir / "bbox",
            mask_output_dir=output_dir / "mask",
        )

    annotations: list[dict[str, Any]] = []

    for index, (
        mapped_class_name,
        original_class_name,
        box,
        mask,
        grounding_score,
        sam_score,
    ) in enumerate(
        zip(
            mapped_class_names,
            original_class_names,
            input_boxes,
            masks,
            grounding_scores,
            sam_scores_list,
        )
    ):
        x_min, y_min, x_max, y_max = [
            float(value)
            for value in box
        ]

        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)

        mask_rle = single_mask_to_rle(mask)

        annotations.append(
            {
                "id": index,
                "class_name": mapped_class_name,
                "original_class_name": original_class_name,
                "class_id": FINAL_CLASS_IDS.get(
                    mapped_class_name,
                    -1,
                ),
                "bbox": [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ],
                "bbox_xywh": [
                    x_min,
                    y_min,
                    width,
                    height,
                ],
                "segmentation": mask_rle,
                "grounding_score": float(
                    grounding_score
                ),
                "sam_score": float(sam_score),
            }
        )

    image_result: dict[str, Any] = {
        "image_path": str(image_path),
        "file_name": image_path.name,
        "output_stem": output_stem,

        "dewarped": fisheye_decoder is not None,
        "dewarp_mode": (
            "cylindrical_fixed_2192_to_1280x704"
            if fisheye_decoder is not None
            else "none"
        ),
        "decoded_img_width": image_width,
        "decoded_img_height": image_height,

        "text_prompt": text_prompt,
        "candidate_box_threshold": box_threshold,
        "person_threshold": person_threshold,
        "car_threshold": car_threshold,
        "overlap_iou_threshold": overlap_iou_threshold,
        "text_threshold": text_threshold,
        "box_format": "xyxy",
        "img_width": image_width,
        "img_height": image_height,
        "num_annotations": len(annotations),
        "annotations": annotations,
    }

    if dump_json:
        json_output_path = (
            output_dir
            / "json"
            / f"{output_stem}.json"
        )

        with json_output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                image_result,
                file,
                ensure_ascii=False,
                indent=2,
            )

    return image_result


def prepare_output_directories(
    output_dir: Path,
    dump_json: bool,
    save_visualization: bool,
    save_decoded_images: bool,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dump_json:
        (output_dir / "json").mkdir(
            parents=True,
            exist_ok=True,
        )

    if save_visualization:
        (output_dir / "bbox").mkdir(
            parents=True,
            exist_ok=True,
        )

        (output_dir / "mask").mkdir(
            parents=True,
            exist_ok=True,
        )

    if save_decoded_images:
        (output_dir / "decoded").mkdir(
            parents=True,
            exist_ok=True,
        )

def main() -> None:
    args = parse_args()
    validate_args(args)

    input_path = args.img_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sam2_checkpoint = (
        args.sam2_checkpoint
        .expanduser()
        .resolve()
    )

    dump_json = not args.no_dump_json
    save_visualization = not args.no_visualization

    device = (
        "cuda"
        if torch.cuda.is_available()
        and not args.force_cpu
        else "cpu"
    )

    setup_cuda_options(device)

    image_paths = collect_image_paths(
        input_path=input_path,
        recursive=args.recursive,
    )

    prepare_output_directories(
        output_dir=output_dir,
        dump_json=dump_json,
        save_visualization=save_visualization,
        save_decoded_images=args.save_decoded_images,
    )

    print(f"[INFO] 入力画像数: {len(image_paths)}")
    print(f"[INFO] 使用デバイス: {device}")
    print(f"[INFO] プロンプト: {args.text_prompt}")
    print(
        f"[INFO] person threshold: "
        f"{args.person_threshold}"
    )
    print(
        f"[INFO] car threshold: "
        f"{args.car_threshold}"
    )
    print(
        f"[INFO] text threshold: "
        f"{args.text_threshold}"
    )
    print(f"[INFO] 出力先: {output_dir}")

    fisheye_decoder: FixedCylindricalDecoder | None = None

    if args.dewarp_fisheye:
        fisheye_decoder = FixedCylindricalDecoder(
            fisheye_width=2192,
            fisheye_height=2192,
            cylindrical_width=2192,
            horizontal_fov=180.0,
            vertical_fov=81.65,
            output_width=1280,
            output_height=704,
        )

        print(
            "[INFO] 魚眼円筒展開を有効化しました: "
            "2192x2192 -> cylindrical -> 1280x704"
        )
    else:
        print("[INFO] 魚眼円筒展開は無効です。")

    if device == "cuda":
        print(
            f"[INFO] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    (
        sam2_predictor,
        processor,
        grounding_model,
    ) = build_models(
        grounding_model_id=args.grounding_model,
        sam2_model_config=args.sam2_model_config,
        sam2_checkpoint=sam2_checkpoint,
        device=device,
    )

    all_results: list[dict[str, Any]] = []
    failed_images: list[dict[str, str]] = []

    total_persons = 0
    total_cars = 0

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        print(
            f"[INFO] [{index}/{len(image_paths)}] "
            f"{image_path.name}"
        )

        try:
            image_result = process_one_image(
                image_path=image_path,
                input_root=input_path,
                text_prompt=args.text_prompt,

                box_threshold=min(
                    args.person_threshold,
                    args.car_threshold,
                ),

                text_threshold=args.text_threshold,
                person_threshold=args.person_threshold,
                car_threshold=args.car_threshold,
                overlap_iou_threshold=args.overlap_iou_threshold,

                keep_unknown_classes=args.keep_unknown_classes,
                device=device,
                sam2_predictor=sam2_predictor,
                processor=processor,
                grounding_model=grounding_model,
                output_dir=output_dir,
                dump_json=dump_json,
                save_visualization=save_visualization,
                fisheye_decoder=fisheye_decoder,
                save_decoded_images=args.save_decoded_images,
            )

            all_results.append(image_result)

            person_count = sum(
                annotation["class_name"] == "person"
                for annotation
                in image_result["annotations"]
            )

            car_count = sum(
                annotation["class_name"] == "car"
                for annotation
                in image_result["annotations"]
            )

            total_persons += person_count
            total_cars += car_count

            print(
                f"[INFO] 検出結果: "
                f"person={person_count}, "
                f"car={car_count}"
            )

        except Exception as exc:
            print(
                f"[WARN] 画像をスキップします: "
                f"{image_path}: {exc}",
                file=sys.stderr,
            )

            failed_images.append(
                {
                    "image_path": str(image_path),
                    "error": str(exc),
                }
            )

        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "grounding_model": args.grounding_model,
        "text_prompt": args.text_prompt,
        "class_map": CLASS_MAP,
        "final_class_ids": FINAL_CLASS_IDS,
        # "box_threshold": args.box_threshold,
        "candidate_box_threshold": min(
            args.person_threshold,
            args.car_threshold,),
        "person_threshold": args.person_threshold,
        "car_threshold": args.car_threshold,
        "overlap_iou_threshold": args.overlap_iou_threshold,
        "text_threshold": args.text_threshold,
        "num_input_images": len(image_paths),
        "num_success_images": len(all_results),
        "num_failed_images": len(failed_images),
        "num_person_annotations": total_persons,
        "num_car_annotations": total_cars,
        "failed_images": failed_images,
        "images": all_results,
    }

    if dump_json:
        combined_json_path = (
            output_dir
            / "all_results.json"
        )

        with combined_json_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

    print()
    print("========== 処理結果 ==========")
    print(f"入力画像数       : {len(image_paths)}")
    print(f"成功画像数       : {len(all_results)}")
    print(f"失敗画像数       : {len(failed_images)}")
    print(f"person検出数     : {total_persons}")
    print(f"car検出数        : {total_cars}")
    print(f"出力先           : {output_dir}")
    print("==============================")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\n[INFO] ユーザーによって中断されました。",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as exc:
        print(
            f"[ERROR] {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
