Drop up to 3 sample document images here (no real PII — synthetic or
already-redaction-safe) with these exact filenames so the Image tab's
"try a sample" thumbnails pick them up:

- sample-1.jpg
- sample-2.jpg
- sample-3.jpg

Missing files simply hide that thumbnail (`ImageTab.tsx`'s `onError` handler) —
you don't need all three.
