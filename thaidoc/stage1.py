"""Stage 1 — coarse visual *family* classifier.

Per the spec/plan the production design uses EfficientNet-V2-B0 / MobileNet-V3
(see docs/DESIGN.md §7). For this CPU-only, no-real-data PoC the ONLY backend
implemented is ``sklearn`` logistic regression over down-scaled grayscale pixel
features: it trains in seconds on CPU and needs no model download, preserving
the "runs anywhere" guarantee. The torchvision EfficientNet/MobileNet backend is
a documented production upgrade, NOT yet implemented here.

The family probabilities this stage emits are the legitimate target for the
temperature-scaling calibration demo (real classifier outputs — though on
RENDERER-BOUNDED data).

Persistence uses ``np.savez`` (plain arrays), deliberately NOT pickle, so a
tampered model artifact cannot execute code on load (avoids the pickle RCE
trust boundary called out in security review).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DATA_DIR, SEED

MODEL_PATH = DATA_DIR / "stage1_model.npz"
_FEAT_SIZE = (48, 48)


def _features(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L").resize(_FEAT_SIZE)
    arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
    return arr


@dataclass
class Stage1Result:
    topk: list[tuple[str, float]]   # [(family, prob), ...] sorted desc

    @property
    def best_family(self) -> str:
        return self.topk[0][0]

    @property
    def best_prob(self) -> float:
        return self.topk[0][1]


class Stage1Classifier:
    def __init__(self, backend: str = "sklearn"):
        self.backend = backend
        self._clf = None
        self._classes: list[str] = []

    # --- training ---
    def train(self, manifest_rows: list[dict]) -> dict:
        from .synth import image_path
        from sklearn.linear_model import LogisticRegression

        train = [r for r in manifest_rows if r["split"] == "train"]
        X = np.stack([_features(image_path(r["filename"])) for r in train])
        y = [r["family"] for r in train]
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(X, y)
        self._clf = clf
        self._classes = list(clf.classes_)
        return {"n_train": len(train), "classes": self._classes}

    def save(self, path: Path = MODEL_PATH) -> None:
        # Persist plain arrays (NOT pickle) so a tampered artifact cannot run
        # code on load. The estimator is rebuilt deterministically from coefs.
        if self._clf is None:
            raise RuntimeError("Nothing to save; train first.")
        np.savez(path, coef=self._clf.coef_, intercept=self._clf.intercept_,
                 classes=np.asarray(self._classes, dtype="U64"))

    def load(self, path: Path = MODEL_PATH) -> "Stage1Classifier":
        from sklearn.linear_model import LogisticRegression
        d = np.load(path, allow_pickle=False)  # plain arrays only -> no RCE
        clf = LogisticRegression()
        clf.coef_ = d["coef"]
        clf.intercept_ = d["intercept"]
        clf.classes_ = d["classes"]
        self._clf = clf
        self._classes = [str(c) for c in d["classes"]]
        return self

    # --- inference ---
    def predict(self, path: Path, k: int = 3) -> Stage1Result:
        if self._clf is None:
            raise RuntimeError("Stage1Classifier not trained/loaded.")
        probs = self._clf.predict_proba(_features(path).reshape(1, -1))[0]
        order = np.argsort(probs, kind="stable")[::-1][:k]
        return Stage1Result(topk=[(self._classes[i], float(probs[i])) for i in order])

    def predict_logits(self, path: Path) -> tuple[np.ndarray, list[str]]:
        """Return raw decision_function logits (for temperature scaling)."""
        if self._clf is None:
            raise RuntimeError("Stage1Classifier not trained/loaded.")
        logits = self._clf.decision_function(_features(path).reshape(1, -1))[0]
        logits = np.atleast_1d(logits)
        # Binary LogisticRegression returns a scalar decision; expand to the
        # two-column form [-d, +d] so it aligns with the 2-element class list.
        if logits.shape[0] == 1 and len(self._classes) == 2:
            d = float(logits[0])
            logits = np.array([-d, d])
        return logits, self._classes


def evaluate_family_accuracy(clf: Stage1Classifier, rows: list[dict]) -> dict:
    from .synth import image_path
    top1 = top3 = 0
    for r in rows:
        res = clf.predict(image_path(r["filename"]), k=3)
        fams = [f for f, _ in res.topk]
        if fams[0] == r["family"]:
            top1 += 1
        if r["family"] in fams:
            top3 += 1
    n = len(rows)
    return {"n": n, "top1": top1 / n if n else 0.0, "top3": top3 / n if n else 0.0}


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    from .synth import load_manifest

    rows = load_manifest()
    clf = Stage1Classifier()
    info = clf.train(rows)
    clf.save()
    test = [r for r in rows if r["split"] == "test"]
    acc = evaluate_family_accuracy(clf, test)
    print(f"Stage-1 trained on {info['n_train']} imgs, {len(info['classes'])} families.")
    print(f"Family accuracy on test (RENDERER-BOUNDED, not predictive): "
          f"top1={acc['top1']:.3f} top3={acc['top3']:.3f} (n={acc['n']})")
