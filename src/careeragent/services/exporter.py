from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Optional

from careeragent.services.health_service import get_artifacts_root


class CareerDossierExporter:
    """
    Description: L9 exporter that bundles reports into a single zip for one-click download.
    Layer: L9
    Input: artifacts/reports folder + final PDF path
    Output: Zip file stored under artifacts/exports/
    """

    def __init__(self, *, artifacts_root: Optional[Path] = None) -> None:
        """
        Description: Initialize exporter.
        Layer: L0
        Input: artifacts_root optional
        Output: CareerDossierExporter
        """
        self._root = artifacts_root or get_artifacts_root()
        self._exports = self._root / "exports"
        self._exports.mkdir(parents=True, exist_ok=True)

    def bundle_reports(self, *, run_id: str, final_pdf_path: Optional[str] = None) -> Dict[str, str]:
        """
        Description: Zip artifacts/reports/<run_id>/ plus optional final PDF into one dossier.
        Layer: L9
        Input: run_id + final_pdf_path
        Output: dict with zip path
        """
        reports_dir = self._root / "reports" / run_id
        zip_path = self._exports / f"career_dossier_{run_id}.zip"

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            if reports_dir.exists():
                for p in reports_dir.rglob("*"):
                    if p.is_file():
                        z.write(p, arcname=str(Path("reports") / run_id / p.relative_to(reports_dir)))
            if final_pdf_path:
                fp = Path(final_pdf_path)
                if fp.exists() and fp.is_file():
                    z.write(fp, arcname=str(Path("final") / fp.name))

        return {"zip": str(zip_path)}
