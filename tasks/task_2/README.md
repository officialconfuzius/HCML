Task 2 Results: Evasion Attack via Frequency Filtering

This section evaluates the effectiveness of our proposed evasion attack—a low-pass frequency filter with a radius of 30—against the AEMAD morphing attack detector. The attack was applied to six different morphing datasets, and the performance was measured using the Equal Error Rate (EER). A higher EER indicates a more successful evasion attack, as it signifies a higher error rate for the detector.

### Results Table

The following table compares the EER of the detector on the original (baseline) datasets versus the post-processed datasets.

+------------------+------------------+-------------------+---------------+--------------------------+
| Morphing Dataset | Baseline EER (%) | Processed EER (%) | Absolute Gain | Relative Improvement (%) |
|------------------|------------------|-------------------|---------------|--------------------------|
| FaceMorpher      |       0.30       |        6.60       |    +6.30      |         +2100%           |
| OpenCV           |       8.33       |       20.53       |    +12.20     |          +146%           |
| MIPGAN I         |       0.90       |       15.00       |    +14.10     |         +1567%           |
| MIPGAN II        |       0.40       |       11.91       |    +11.51     |         +2878%           |
| Webmorph         |      12.40       |       21.60       |    +9.20      |           +74%           |
| MorDIFF          |       7.90       |       35.50       |    +27.60     |          +349%           |
+------------------+------------------+-------------------+---------------+--------------------------+


<img width="706" height="559" alt="image" src="https://github.com/user-attachments/assets/67555cfd-d567-4206-9138-2fa1301cce35" />

### Interpretation of Results

Universal Success: The frequency filtering attack was successful against every single morphing dataset. In all cases, the detector's error rate (EER) increased after the processing, proving the method is robust and generalizable.

Massive Impact on High-Fidelity Morphs: The attack was most devastating, in relative terms, on the datasets that the detector was originally best at identifying. For MIPGAN-II and FaceMorpher, where the baseline error was less than 1%, the attack increased the detector's error rate by an incredible +2878% and +2100% respectively. This suggests these high-quality morphs contain subtle, high-frequency artifacts that, once removed, make them exceptionally difficult for the detector to flag.

Most Vulnerable Dataset: The MorDIFF dataset showed the highest final error rate (35.50%) and the largest absolute increase (+27.6 points). This means that while the baseline MorDIFF images were moderately detectable, the processing made them extremely evasive.

Conclusion: the experiment conclusively demonstrates that frequency domain filtering is a highly effective evasion technique against reconstruction-based morphing attack detectors like AEMAD. By removing the subtle high-frequency evidence of manipulation, we can significantly degrade the detector's performance across a wide variety of morphing methods.
