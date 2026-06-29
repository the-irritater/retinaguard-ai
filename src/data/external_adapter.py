"""
RetinaGuard AI - External Dataset Adapter
-
Purpose: Standarise external retinal datasets (e.g. APTOS, Messidor-2, EyePACS)
to match the input format, preprocessing pipeline, and labels expected by the
trained RetinaGuard AI models.

Provides dataset conversion interfaces to evaluate the frozen model on external
data without changing the trained model's internals.

Usage:
    from src.data.external_adapter import ExternalDatasetAdapter
    adapter = ExternalDatasetAdapter(dataset_name='aptos')
    metadata_df = adapter.adapt_metadata(external_csv_path)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger("retinaguard.adapter")


class ExternalDatasetAdapter:
    """Standardises third-party retinal dataset annotations for cross-dataset validation."""

    def __init__(self, dataset_name: str = "aptos") -> None:
        """Initialise the adapter.

        Args:
            dataset_name: Name of the external dataset (e.g., 'aptos', 'messidor2').
        """
        self.dataset_name = dataset_name.lower()

    def adapt_metadata(
        self,
        csv_path: str | Path,
        image_dir: str | Path,
    ) -> pd.DataFrame:
        """Read external metadata CSV and maps it to RetinaGuard standard format.

        Standard columns output:
            - image_id: Unique string identifier
            - full_path: Absolute path to image
            - dr_grade: Integer 0-4
            - binary_label: Integer 0 (grade < 2) or 1 (grade >= 2)
            - match_status: 'matched'
            - partition: 'external'
            - dataset_name: Name of the external dataset

        Args:
            csv_path: Path to external dataset CSV file.
            image_dir: Path to directory containing external images.

        Returns:
            Standardised DataFrame ready for IDRiDDataset loading.
        """
        csv_path = Path(csv_path)
        image_dir = Path(image_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"External CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} records from external dataset: {self.dataset_name}")

        standard_rows = []

        if self.dataset_name == "aptos":
            # APTOS 2019 format: columns are 'id_code' (image name) and 'diagnosis' (0-4 grade)
            id_col = "id_code"
            grade_col = "diagnosis"
            
            for idx, row in df.iterrows():
                image_id = str(row[id_col]).strip()
                dr_grade = int(row[grade_col])
                
                # Check extension (APTOS usually uses .png)
                img_path = image_dir / f"{image_id}.png"
                if not img_path.exists():
                    # Fallback to .jpg
                    img_path = image_dir / f"{image_id}.jpg"
                
                if img_path.exists():
                    standard_rows.append({
                        "image_id": image_id,
                        "stem": image_id,
                        "filename": img_path.name,
                        "full_path": str(img_path),
                        "partition": "external",
                        "dr_grade": dr_grade,
                        "binary_label": 1 if dr_grade >= 2 else 0,
                        "match_status": "matched",
                        "readable": True,
                        "dataset_name": "aptos_2019",
                    })

        elif self.dataset_name in ("messidor2", "messidor_2"):
            # Messidor-2 format: columns 'image' and 'adjudicated_dr_grade' (or similar)
            id_col = "image"
            grade_col = "adjudicated_dr_grade"
            
            for idx, row in df.iterrows():
                filename = str(row[id_col]).strip()
                dr_grade = int(row[grade_col])
                img_path = image_dir / filename
                image_id = img_path.stem

                if img_path.exists():
                    standard_rows.append({
                        "image_id": image_id,
                        "stem": image_id,
                        "filename": filename,
                        "full_path": str(img_path),
                        "partition": "external",
                        "dr_grade": dr_grade,
                        "binary_label": 1 if dr_grade >= 2 else 0,
                        "match_status": "matched",
                        "readable": True,
                        "dataset_name": "messidor2",
                    })
        else:
            # Generic/custom fallback
            # Expects columns: 'image_id' and 'dr_grade'
            id_col = "image_id" if "image_id" in df.columns else df.columns[0]
            grade_col = "dr_grade" if "dr_grade" in df.columns else df.columns[1]
            logger.warning(
                f"Using generic adapter. Mapping columns: '{id_col}' -> image_id, "
                f"'{grade_col}' -> dr_grade"
            )

            for idx, row in df.iterrows():
                image_id = str(row[id_col]).strip()
                dr_grade = int(row[grade_col])
                
                # Search for image file with common extensions
                img_path = None
                for ext in [".png", ".jpg", ".jpeg"]:
                    p = image_dir / f"{image_id}{ext}"
                    if p.exists():
                        img_path = p
                        break
                
                if img_path is not None:
                    standard_rows.append({
                        "image_id": image_id,
                        "stem": image_id,
                        "filename": img_path.name,
                        "full_path": str(img_path),
                        "partition": "external",
                        "dr_grade": dr_grade,
                        "binary_label": 1 if dr_grade >= 2 else 0,
                        "match_status": "matched",
                        "readable": True,
                        "dataset_name": self.dataset_name,
                    })

        adapted_df = pd.DataFrame(standard_rows)
        logger.info(
            f"Successfully adapted {len(adapted_df)} images from external dataset "
            f"'{self.dataset_name}'."
        )
        return adapted_df


# -
# Cross-Dataset Performance Delta Report Utility
# -
def compute_performance_delta(
    internal_metrics: Dict[str, float],
    external_metrics: Dict[str, float],
) -> Dict[str, float]:
    """Calculate performance change metrics between internal test and external dataset.

    Args:
        internal_metrics: Dictionary of test metrics from internal evaluation.
        external_metrics: Dictionary of test metrics from external validation.

    Returns:
        Dictionary containing delta (external - internal) for each metric.
    """
    deltas = {}
    for k in internal_metrics.keys():
        if k in external_metrics and isinstance(internal_metrics[k], (int, float)):
            deltas[f"{k}_delta"] = float(external_metrics[k] - internal_metrics[k])
    return deltas
