"""Minimal end-to-end demo.

    python examples/demo.py path/to/page.jpg [--no-visual] [--device cpu]
"""
import argparse

from eka_pii_redaction import ImagePIIRedactor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--hf-repo", default="ekacare/pii-redactors",
                    help="HF repo id or a local dir with image/layoutlmv3/ + image/yolo/best.pt")
    ap.add_argument("--no-visual", action="store_true", help="text PII only")
    ap.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--categories", default="",
                    help="comma-separated categories to detect (default: all)")
    ap.add_argument("--mode", default="solid", choices=["solid", "blur", "pixelate"])
    ap.add_argument("--out", default="redacted.png")
    args = ap.parse_args()

    redactor = ImagePIIRedactor(
        args.hf_repo,
        detect_visual=not args.no_visual,
        device=args.device,
        categories=[s for s in args.categories.split(",") if s] or None,
    )

    entities = redactor.detect(args.image)
    print(f"Detected {len(entities)} PII entities:")
    for e in entities:
        txt = f' "{e.text}"' if e.text else ""
        print(f"  [{e.kind:6}] {e.category:30} {e.l1:16} bbox={e.bbox} "
              f"score={e.score}{txt}")

    redactor.redact(args.image, entities, mode=args.mode).save(args.out)
    print(f"\nRedacted image saved to {args.out}")


if __name__ == "__main__":
    main()
