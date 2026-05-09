# Ghibli Pix2Pix-Zero Accuracy Evaluation

This report summarizes the accuracy and structural consistency of the current Ghibli style transfer model.

## Structural Consistency (Input vs Output)

These metrics measure how well the original photo's layout was preserved during the style transfer.

| Image       | MSE (Lower is better) | PSNR (Higher is better) |
| :---------- | :-------------------- | :---------------------- |
| city        | 2093.32               | 14.92 dB                |
| city2       | 2673.41               | 13.86 dB                |
| forest      | 2348.95               | 14.42 dB                |
| mountain    | 2483.77               | 14.18 dB                |
| road        | 1531.29               | 16.28 dB                |
| street      | 3780.22               | 12.36 dB                |
| **AVERAGE** | **2485.16**           | **14.34 dB**            |

### Metric Definitions

- **MSE (Mean Squared Error)**: The average squared difference between the original and edited pixels. 2485 is a standard range for style transfer, indicating a significant but controlled texture change.
- **PSNR (Peak Signal-to-Noise Ratio)**: A ratio of the maximum possible power of a signal and the power of corrupting noise. 14.34 dB suggests a clear style shift while keeping the original composition recognizable.
