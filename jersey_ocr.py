"""Jersey-number reading with PARSeq on torso crops of tracked players."""

import cv2
import numpy as np
import torch

DIGITS = set("0123456789")


class JerseyReader:
    def __init__(self):
        self.model = torch.hub.load("baudm/parseq", "parseq", pretrained=True,
                                    trust_repo=True).eval()
        self.h, self.w = self.model.hparams.img_size  # 32 x 128
        self.tokenizer = self.model.tokenizer

    @staticmethod
    def torso_crop(frame, box):
        """Chest/back region of a person box, where the number is printed."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx1, cx2 = int(x1 + 0.12 * w), int(x2 - 0.12 * w)
        cy1, cy2 = int(y1 + 0.14 * h), int(y1 + 0.50 * h)
        H, W = frame.shape[:2]
        cx1, cy1, cx2, cy2 = max(cx1, 0), max(cy1, 0), min(cx2, W), min(cy2, H)
        if cx2 - cx1 < 8 or cy2 - cy1 < 8:
            return None
        return frame[cy1:cy2, cx1:cx2]

    def _prep(self, crop):
        # PARSeq was trained on crops stretched to its fixed input size; padding hurts it.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_CUBIC)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return (t - 0.5) / 0.5

    @torch.inference_mode()
    def read(self, crops):
        """Return [(number_str or None, confidence)] for a list of BGR crops."""
        if not crops:
            return []
        batch = torch.stack([self._prep(c) for c in crops])
        probs = self.model(batch).softmax(-1)
        labels, confs = self.tokenizer.decode(probs)
        out = []
        for text, conf in zip(labels, confs):
            # conf holds per-character probabilities (incl. EOS); their product is the sequence score
            score = float(conf.prod())
            if 1 <= len(text) <= 2 and set(text) <= DIGITS and not (len(text) == 2 and text[0] == "0"):
                out.append((text, score))
            else:
                out.append((None, score))
        return out
