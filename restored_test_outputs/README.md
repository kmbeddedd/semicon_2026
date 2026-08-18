# Restored test outputs

This directory contains the final predictions produced by `weights/best_model.pt`
for all 400 inputs from `Test_NoisyLR.zip`.

- `000000.npy` through `000399.npy` are the authoritative submission arrays.
  Every file is a finite `float32` array with shape `256 x 256` and values in
  the physical `[0, 1]` range.
- `000000_restored.png` through `000399_restored.png` are 8-bit visual previews.
  They are provided for convenient inspection and should not replace the NPY
  arrays during numerical scoring.

The outputs were generated with 8-fold D4 test-time augmentation using:

```bash
python eval.py \
  --input_dir data/test/NoisyLR \
  --output_dir restored_test_outputs \
  --weights weights/best_model.pt \
  --scale 2 \
  --batch_size 8
```

The filenames preserve a one-to-one mapping with the input NPY filenames.
