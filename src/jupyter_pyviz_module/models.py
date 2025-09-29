
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import os

@dataclass
class DatasetModel:

    df: pd.DataFrame
    kind: str  # "msr", "itm", "scn", etc.

    @classmethod
    def from_csv(cls, path: str) -> "DatasetModel":
        df = pd.read_csv(path)

        # Guess dataset kind from filename
        filename = os.path.basename(path).lower()
        if "_msr" in filename:
            required = ["IRAS", "true_IRAS"]
            kind = "msr"
        elif "_itm" in filename:
            required = ["item_IRAS", "item_DR_10cm_uSv_h"]  # example required cols
            kind = "itm"
        elif "_scn" in filename:
            required = ["machine", "position", "beam_p_s"]  # example required cols
            kind = "scn"
        else:
            raise ValueError(f"Could not infer dataset type from {filename}")

        # Validate required columns
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {kind}: {missing}")

        return cls(df=df, kind=kind)
