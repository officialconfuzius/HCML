Task 2 Results: Evasion Attack via Frequency Filtering

This section evaluates the effectiveness of our proposed evasion attack—a low-pass frequency filter with a radius of 30—against the AEMAD morphing attack detector. The attack was applied to six different morphing datasets, and the performance was measured using the Equal Error Rate (EER). A higher EER indicates a more successful evasion attack, as it signifies a higher error rate for the detector.

### Results Table

The following table compares the EER of the detector on the original (baseline) datasets versus the post-processed datasets.

<table>
    <thead>
        <tr>
            <th>Morphing Dataset</th>
            <th>Baseline EER (%)</th>
            <th>Processed EER (%)</th>
            <th>Absolute Gain</th>
            <th>Relative Improvement (%)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FaceMorpher</td>
            <td align="center">0.30</td>
            <td align="center">6.60</td>
            <td align="center">+6.30</td>
            <td align="center">+2100%</td>
        </tr>
        <tr>
            <td>OpenCV</td>
            <td align="center">8.33</td>
            <td align="center">20.53</td>
            <td align="center">+12.20</td>
            <td align="center">+146%</td>
        </tr>
        <tr>
            <td>MIPGAN I</td>
            <td align="center">0.90</td>
            <td align="center">15.00</td>
            <td align="center">+14.10</td>
            <td align="center">+1567%</td>
        </tr>
        <tr>
            <td>MIPGAN II</td>
            <td align="center">0.40</td>
            <td align="center">11.91</td>
            <td align="center">+11.51</td>
            <td align="center">+2878%</td>
        </tr>
        <tr>
            <td>Webmorph</td>
            <td align="center">12.40</td>
            <td align="center">21.60</td>
            <td align="center">+9.20</td>
            <td align="center">+74%</td>
        </tr>
        <tr>
            <td>MorDIFF</td>
            <td align="center">7.90</td>
            <td align="center">35.50</td>
            <td align="center">+27.60</td>
            <td align="center">+349%</td>
        </tr>
    </tbody>
</table>



### Interpretation of Results

Universal Success: The frequency filtering attack was successful against every single morphing dataset. In all cases, the detector's error rate (EER) increased after the processing, proving the method is robust and generalizable.

Massive Impact on High-Fidelity Morphs: The attack was most devastating, in relative terms, on the datasets that the detector was originally best at identifying. For MIPGAN-II and FaceMorpher, where the baseline error was less than 1%, the attack increased the detector's error rate by an incredible +2878% and +2100% respectively. This suggests these high-quality morphs contain subtle, high-frequency artifacts that, once removed, make them exceptionally difficult for the detector to flag.

Most Vulnerable Dataset: The MorDIFF dataset showed the highest final error rate (35.50%) and the largest absolute increase (+27.6 points). This means that while the baseline MorDIFF images were moderately detectable, the processing made them extremely evasive.

Conclusion: the experiment conclusively demonstrates that frequency domain filtering is a highly effective evasion technique against reconstruction-based morphing attack detectors like AEMAD. By removing the subtle high-frequency evidence of manipulation, we can significantly degrade the detector's performance across a wide variety of morphing methods.


Attached is the visual effect caused to the photo after the smoothening filter.
As we can observe the top photo (Original) is indeed more rough then the bottom one (the proccesed version).

<img width="706" height="559" alt="image" src="https://github.com/user-attachments/assets/67555cfd-d567-4206-9138-2fa1301cce35" />


## Bonus: Re-evaluation of task 1 using the dataset of task 2
To check whether or not the attacks would still be successful, we re-evaluated the dataset against the face recognition models of task 1. Here are the results: 

**Table 2: Attack Success Rates (FNMR@FMR=1%) for Each Model and Morphing Dataset**

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>ElasticFaceArc<br><sub>Success Rate</sub></th>
            <th>ElasticFaceCos<br><sub>Success Rate</sub></th>
            <th>CurricularFace<br><sub>Success Rate</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FaceMorpher_aligned</td>
            <td align="center">0.9440</td>
            <td align="center">0.9590</td>
            <td align="center">0.9520</td>
        </tr>
        <tr>
            <td>Webmorph_aligned</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
        </tr>
        <tr>
            <td>MorDIFF_aligned</td>
            <td align="center">0.9880</td>
            <td align="center">0.9920</td>
            <td align="center">0.9900</td>
        </tr>
        <tr>
            <td>MIPGAN_I_aligned</td>
            <td align="center">0.9150</td>
            <td align="center">0.9400</td>
            <td align="center">0.9340</td>
        </tr>
        <tr>
            <td>MIPGAN_II_aligned</td>
            <td align="center">0.8719</td>
            <td align="center">0.8949</td>
            <td align="center">0.8809</td>
        </tr>
        <tr>
            <td>OpenCV_aligned</td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9980</b></td>
            <td align="center"><b>0.9959</b></td>
        </tr>
    </tbody>
</table>

**Table 3: Attack Success Rates (FNMR@FMR=0.1%) for Each Model and Morphing Dataset**

<table>
    <thead>
        <tr>
            <th>Dataset</th>
            <th>ElasticFaceArc<br><sub>Success Rate</sub></th>
            <th>ElasticFaceCos<br><sub>Success Rate</sub></th>
            <th>CurricularFace<br><sub>Success Rate</sub></th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>FaceMorpher_aligned</td>
            <td align="center">0.8840</td>
            <td align="center">0.9040</td>
            <td align="center">0.8920</td>
        </tr>
        <tr>
            <td>Webmorph_aligned</td>
            <td align="center"><b>0.9760</b></td>
            <td align="center"><b>0.9840</b></td>
            <td align="center"><b>0.9820</b></td>
        </tr>
        <tr>
            <td>MorDIFF_aligned</td>
            <td align="center">0.9200</td>
            <td align="center">0.9580</td>
            <td align="center">0.9360</td>
        </tr>
        <tr>
            <td>MIPGAN_I_aligned</td>
            <td align="center">0.7190</td>
            <td align="center">0.7400</td>
            <td align="center">0.7310</td>
        </tr>
        <tr>
            <td>MIPGAN_II_aligned</td>
            <td align="center">0.6096</td>
            <td align="center">0.6517</td>
            <td align="center">0.6376</td>
        </tr>
        <tr>
            <td>OpenCV_aligned</td>
            <td align="center">0.9492</td>
            <td align="center">0.9715</td>
            <td align="center">0.9533</td>
        </tr>
    </tbody>
</table>

### Comparison of Task 1 Baseline vs. Task 2 Frequency-Filtered Results

Tables 1 and 2 from the Task 1 directory show the baseline attack success rates for various morphing datasets and models, while Tables 2 and 3 from the Task 2 directory show the success rates after applying the frequency filtering (low-pass) evasion attack.

#### Key Observations:
- **Success Rates Remain High:**
  - Across all datasets and models, the attack success rates after frequency filtering (Task 2, Tables 2 & 3) remain very high and are comparable to the baseline results (Task 1, Tables 1 & 2). For example, OpenCV_aligned and Webmorph_aligned consistently achieve success rates above 0.98 in both cases.
- **Minimal Impact on Recognition Models:**
  - The frequency filtering attack, while highly effective at evading the AEMAD detector (as shown by increased EER), does not significantly reduce the ability of the face recognition models (ElasticFaceArc, ElasticFaceCos, CurricularFace) to match morphs to their sources. The success rates for morphing attacks remain nearly unchanged.
- **Robustness of Face Recognition:**
  - This suggests that the face recognition models are robust to the type of frequency filtering applied. The morphs, even after smoothing, still retain enough identity information to bypass the recognition systems.
- **Dataset Trends:**
  - The relative ranking of datasets is preserved: OpenCV, Webmorph, and MorDIFF morphs are always the most effective, while MIPGAN_II is the least effective, regardless of filtering.
- **Threshold Effects:**
  - The stricter threshold (FNMR@FMR=0.1%) in both tasks reduces success rates slightly, but the overall trends and high effectiveness of morphs persist.

#### Summary:
- **Frequency filtering is a powerful evasion technique against morphing detectors, but it does not substantially impact the success of morphing attacks against face recognition models.**
- **Morphs remain highly effective at bypassing recognition, even after post-processing, indicating that further countermeasures may be needed to defend against these attacks.**
- **The results highlight the need for multi-layered security, as improvements in detector evasion do not necessarily translate to reduced recognition performance.**
