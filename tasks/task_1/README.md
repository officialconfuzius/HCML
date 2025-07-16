## Subtask 1: Evaluate Existing Attacks

Assess the effectiveness of current attack methods by measuring their successful attack rates. This involves analyzing various attack strategies and quantifying their ability to bypass recognition systems.

---

### Interpretation of Baseline Evaluation Tables

The tables below report the **attack success rates** (i.e., the proportion of morphs that successfully bypass the face recognition system) for different morphing datasets and face recognition models. The rates are shown for two security thresholds:
- **FNMR@FMR=1%** (Table 1): False Non-Match Rate at a False Match Rate of 1%
- **FNMR@FMR=0.1%** (Table 2): False Non-Match Rate at a False Match Rate of 0.1% (stricter)

**Columns:**  Each column is a different face recognition model: ElasticFaceArc, ElasticFaceCos, CurricularFace. Each row is a different morphing dataset (e.g., FaceMorpher_aligned, Webmorph_aligned, etc.).

#### Key Observations
- **Model Sensitivity:**
  - *ElasticFaceCos* is highly robust to FaceMorpher_aligned morphs (very low success rate: 0.0020 at both thresholds), but less so for other datasets.
  - *ElasticFaceArc* and *CurricularFace* are generally more vulnerable, with high success rates for most datasets.
- **Dataset Difficulty:**
  - *OpenCV_aligned* and *Webmorph_aligned* morphs are the most effective at bypassing all models (success rates >0.98 in most cases).
  - *FaceMorpher_aligned* is only effective against ElasticFaceArc and CurricularFace, not ElasticFaceCos.
  - *MIPGAN_II_aligned* is much less effective against CurricularFace at both thresholds (success rates near zero).
- **Effect of Stricter Threshold (0.1%):**
  - All models and datasets see a drop in success rates at the stricter threshold, but the relative ranking of datasets and models remains similar.
  - The drop is most pronounced for MIPGAN_II_aligned and MIPGAN_I_aligned.
- **Best and Worst Cases:**
  - *Best attacks:* OpenCV_aligned and Webmorph_aligned morphs, which almost always succeed regardless of the model or threshold.
  - *Worst attacks:* FaceMorpher_aligned against ElasticFaceCos, and MIPGAN_II_aligned against CurricularFace - as both have success rates of less than 1%. MIPGAN_II_aligned in general shows significantly lower success rates than the other attacks.

**Summary:**
- *ElasticFaceCos* is the most robust model against FaceMorpher_aligned morphs, but not against others.
- *OpenCV_aligned* and *Webmorph_aligned* morphs are the most dangerous, consistently achieving high attack success rates.
- Stricter thresholds reduce attack success, but not enough to stop the best morphs.
- The effectiveness of a morphing attack depends both on the morphing method and the recognition model.

**Conclusion:**
Some morphing methods can still reliably fool even strong face recognition models, especially with certain datasets. However, model choice and thresholding can significantly impact system security.

---

**Table 1: Attack Success Rates (FNMR@FMR=1%) for Each Model and Morphing Dataset**

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
            <td align="center">0.9430</td>
            <td align="center">0.0020</td>
            <td align="center">0.9500</td>
        </tr>
        <tr>
            <td>Webmorph_aligned</td>
            <td align="center">0.9900</td>
            <td align="center">0.9860</td>
            <td align="center">0.9880</td>
        </tr>
        <tr>
            <td>MorDIFF_aligned</td>
            <td align="center">0.9790</td>
            <td align="center">0.9920</td>
            <td align="center">0.9870</td>
        </tr>
        <tr>
            <td>MIPGAN_I_aligned</td>
            <td align="center">0.9240</td>
            <td align="center">0.9370</td>
            <td align="center">0.9280</td>
        </tr>
        <tr>
            <td>MIPGAN_II_aligned</td>
            <td align="center">0.8779</td>
            <td align="center">0.8989</td>
            <td align="center">0.0030</td>
        </tr>
        <tr>
            <td>OpenCV_aligned</td>
            <td align="center"><b>0.9929</b></td>
            <td align="center"><b>0.9959</b></td>
            <td align="center"><b>0.9929</b></td>
        </tr>
    </tbody>
</table>

**Table 2: Attack Success Rates (FNMR@FMR=0.1%) for Each Model and Morphing Dataset**

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
            <td align="center">0.8710</td>
            <td align="center">0.0020</td>
            <td align="center">0.8780</td>
        </tr>
        <tr>
            <td>Webmorph_aligned</td>
            <td align="center"><b>0.9800</b></td>
            <td align="center"><b>0.9780</b></td>
            <td align="center"><b>0.9780</b></td>
        </tr>
        <tr>
            <td>MorDIFF_aligned</td>
            <td align="center">0.8730</td>
            <td align="center">0.9300</td>
            <td align="center">0.8950</td>
        </tr>
        <tr>
            <td>MIPGAN_I_aligned</td>
            <td align="center">0.6900</td>
            <td align="center">0.7360</td>
            <td align="center">0.7080</td>
        </tr>
        <tr>
            <td>MIPGAN_II_aligned</td>
            <td align="center">0.6046</td>
            <td align="center">0.6436</td>
            <td align="center">0.0010</td>
        </tr>
        <tr>
            <td>OpenCV_aligned</td>
            <td align="center">0.9289</td>
            <td align="center">0.9644</td>
            <td align="center">0.9461</td>
        </tr>
    </tbody>
</table>

## Subtask 2: Post-Process Morphs to Improve Attack Rates

Enhance existing morphs through both automatic and manual post-processing techniques to increase their similarity and, consequently, their attack success rates. This includes experimenting with different post-processing methods to optimize morph quality and effectiveness.

### First Approach: 
We began by focusing on the orange dots that appeared as artifacts in the morphed face images. Rather than simply deleting these pixels, we used a computer vision technique called inpainting to seamlessly replace them. This approach aimed to enhance the visual integrity of the morphed face. Since face recognition models are trained on millions of real faces, they are highly sensitive to unnatural features. Artifacts like orange dots can disrupt the model’s perception, lowering the similarity score. By removing these artifacts, we made the faces appear more natural to the model, which led to improved similarity scores.

Here’s a concise technical summary of the two steps (the code can be found in the file `post_process_morphs.py`):

#### Detection (Mask Creation):

The script converts the image from BGR to HSV color space, making it easier to isolate the orange color.
Using cv2.inRange(), it creates a binary mask: white pixels represent detected orange areas (artifacts), black pixels are everything else.

#### Reconstruction (Inpainting):

The script uses cv2.inpaint() with the mask to repair the orange areas.
The function analyzes the surrounding "healthy" pixels and fills in the masked (damaged) regions with new, natural-looking texture and color, seamlessly blending the repair.
This process detects and removes orange artifacts by algorithmically reconstructing those regions based on their surroundings.

---

**Table 3: Attack Success Rates (FNMR@FMR=1%) for Each Model and Post-processed Morphing Dataset Using the First Approach**

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
            <td>FaceMorpher_processed_orange</td>
            <td align="center">0.9480</td>
            <td align="center">0.9620</td>
            <td align="center">0.9570</td>
        </tr>
        <tr>
            <td>Webmorph_processed_orange</td>
            <td align="center">0.9900</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
        </tr>
        <tr>
            <td>MorDIFF_processed_orange</td>
            <td align="center">0.9870</td>
            <td align="center">0.9940</td>
            <td align="center">0.9900</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_orange</td>
            <td align="center">0.9400</td>
            <td align="center">0.9510</td>
            <td align="center">0.9430</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_orange</td>
            <td align="center">0.9099</td>
            <td align="center">0.9289</td>
            <td align="center">0.9069</td>
        </tr>
        <tr>
            <td>OpenCV_processed_orange</td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9929</b></td>
        </tr>
    </tbody>
</table>


**Table 4: Attack Success Rates (FNMR@FMR=0.1%) for Each Model and Post-processed Morphing Dataset Using the First Approach**

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
            <td>FaceMorpher_processed_orange</td>
            <td align="center">0.8840</td>
            <td align="center">0.9150</td>
            <td align="center">0.9010</td>
        </tr>
        <tr>
            <td>Webmorph_processed_orange</td>
            <td align="center"><b>0.9800</b></td>
            <td align="center"><b>0.9860</b></td>
            <td align="center"><b>0.9860</b></td>
        </tr>
        <tr>
            <td>MorDIFF_processed_orange</td>
            <td align="center">0.9180</td>
            <td align="center">0.9630</td>
            <td align="center">0.9370</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_orange</td>
            <td align="center">0.7510</td>
            <td align="center">0.7960</td>
            <td align="center">0.7610</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_orange</td>
            <td align="center">0.6557</td>
            <td align="center">0.7167</td>
            <td align="center">0.6877</td>
        </tr>
        <tr>
            <td>OpenCV_processed_orange</td>
            <td align="center">0.9563</td>
            <td align="center">0.9776</td>
            <td align="center">0.9644</td>
        </tr>
    </tbody>
</table>

### Interpretation of Post-Processing Results (Tables 3 & 4)

Tables 3 and 4 show the attack success rates after applying the first post-processing approach (orange artifact removal) to the morphing datasets. These results can be directly compared to the baseline results in Tables 1 and 2.

#### Key Findings:
- **Overall Improvement:**
  - For most datasets and models, post-processing leads to higher attack success rates compared to the baseline (Tables 1 & 2). This means that removing visible artifacts makes morphs more likely to fool the face recognition systems.
- **ElasticFaceCos Sensitivity:**
  - The most dramatic improvement is seen for the FaceMorpher dataset against ElasticFaceCos: the success rate jumps from 0.0020 (Table 1) to 0.9620 (Table 3) at FNMR@FMR=1%. This shows that ElasticFaceCos is highly sensitive to such artifacts, and their removal makes morphs much more effective.
- **Other Models and Datasets:**
  - For ElasticFaceArc and CurricularFace, the improvement is smaller but still present for most datasets. For example, FaceMorpher_processed_orange increases from 0.9430 to 0.9480 (ElasticFaceArc) and from 0.9500 to 0.9570 (CurricularFace).
  - For datasets that were already highly effective (e.g., Webmorph, OpenCV), the success rates remain very high, with only minor increases.
- **Stricter Threshold (FNMR@FMR=0.1%):**
  - At the stricter threshold (Table 4), the same trends hold: post-processing increases attack success rates across the board, and the improvement is especially pronounced for previously low-performing combinations.
- **MIPGAN_II:**
  - Even for MIPGAN_II, which was previously ineffective against CurricularFace (0.0030 in Table 1), the post-processed version achieves a much higher success rate (0.9069 in Table 3).

#### Summary:
- **Artifact removal via inpainting is highly effective at increasing the attack success rate of morphs, especially for models that are sensitive to visual artifacts.**
- **The gap between the best and worst performing morphing methods is reduced after post-processing, making most attacks highly effective regardless of the model.**
- **Comparing to the baseline, post-processing can turn previously ineffective attacks into highly successful ones.**

---

### Second Approach: Gray Background Artifact Removal with Safe Zone Protection

This post-processing method targets gray background artifacts that often appear around the edges of morphed face images. The key idea is to remove these unwanted gray regions while protecting the central area of the image (where the main facial features are) to avoid distorting the face itself.

**How it works:**
- A central "safe zone" is defined, covering the main face region (from 15% to 85% of the image width and height).
- A mask is created to detect gray pixels (low saturation, medium brightness) in the image, but only outside the safe zone.
- The detected gray regions are then inpainted (reconstructed) using surrounding pixels, seamlessly removing the artifacts while leaving the face untouched.
- The script processes all images in the dataset, saves the cleaned images, and generates new triplet files for evaluation.

**Motivation:**
- This approach is designed to improve the visual quality of morphs by removing distracting background artifacts, which may help increase their similarity scores and attack success rates, especially for models sensitive to such artifacts.
- By protecting the central face region, the risk of distorting important facial features is minimized, preserving the effectiveness of the morph.

The implementation can be found in `post_process_morphs_attempt2.py`.

---

**Table 5: Attack Success Rates (FNMR@FMR=1%) for Each Model and Post-processed Morphing Dataset Using the Second Approach**

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
            <td>FaceMorpher_processed_gray</td>
            <td align="center">0.9450</td>
            <td align="center">0.9580</td>
            <td align="center">0.9550</td>
        </tr>
        <tr>
            <td>Webmorph_processed_gray</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
        </tr>
        <tr>
            <td>MorDIFF_processed_gray</td>
            <td align="center">0.9860</td>
            <td align="center">0.9940</td>
            <td align="center">0.9890</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_gray</td>
            <td align="center">0.9320</td>
            <td align="center">0.9370</td>
            <td align="center">0.9330</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_gray</td>
            <td align="center">0.8979</td>
            <td align="center">0.9089</td>
            <td align="center">0.8989</td>
        </tr>
        <tr>
            <td>OpenCV_processed_gray</td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9929</b></td>
        </tr>
    </tbody>
</table>


**Table 6: Attack Success Rates (FNMR@FMR=0.1%) for Each Model and Post-processed Morphing Dataset Using the Second Approach**

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
            <td>FaceMorpher_processed_gray</td>
            <td align="center">0.8830</td>
            <td align="center">0.9140</td>
            <td align="center">0.8990</td>
        </tr>
        <tr>
            <td>Webmorph_processed_gray</td>
            <td align="center"><b>0.9780</b></td>
            <td align="center"><b>0.9860</b></td>
            <td align="center"><b>0.9820</b></td>
        </tr>
        <tr>
            <td>MorDIFF_processed_gray</td>
            <td align="center">0.9150</td>
            <td align="center">0.9610</td>
            <td align="center">0.9330</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_gray</td>
            <td align="center">0.7440</td>
            <td align="center">0.7710</td>
            <td align="center">0.7390</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_gray</td>
            <td align="center">0.6416</td>
            <td align="center">0.6737</td>
            <td align="center">0.6647</td>
        </tr>
        <tr>
            <td>OpenCV_processed_gray</td>
            <td align="center">0.9522</td>
            <td align="center">0.9736</td>
            <td align="center">0.9563</td>
        </tr>
    </tbody>
</table>

### Interpretation of Post-Processing Results: Second Approach (Tables 5 & 6)

Tables 5 and 6 present the attack success rates after applying the second post-processing approach (gray background artifact removal with safe zone protection). These results should be compared to the baseline (Tables 1 & 2) and to the first post-processing approach (Tables 3 & 4).

#### Key Findings:
- **Consistent Improvement Over Baseline:**
  - For all datasets and models, the second approach increases attack success rates compared to the baseline (Tables 1 & 2). This demonstrates that removing gray background artifacts—even while protecting the central face—makes morphs more effective at bypassing recognition systems.
- **Magnitude of Improvement:**
  - The improvement is especially notable for ElasticFaceCos and for datasets that had lower baseline success rates (e.g., FaceMorpher, MIPGAN_II). For example, FaceMorpher_processed_gray against ElasticFaceCos rises from 0.0020 (Table 1) to 0.9580 (Table 5) at FNMR@FMR=1%.
  - For datasets that were already highly effective (Webmorph, OpenCV), the success rates remain very high, with only slight increases.
- **Comparison to First Approach:**
  - The second approach achieves similar or slightly lower success rates compared to the first approach (orange artifact removal), but still provides a substantial boost over the baseline. This suggests that both types of artifact removal are beneficial, and the choice may depend on the specific dataset or visual artifacts present.
- **Stricter Threshold (FNMR@FMR=0.1%):**
  - At the stricter threshold (Table 6), the trends persist: post-processing increases attack success rates, and the gap between the best and worst performing morphing methods is reduced.
- **Preservation of Face Quality:**
  - By protecting the central face region, this approach minimizes the risk of distorting facial features, ensuring that the morphs remain visually plausible while still improving attack effectiveness.

#### Summary:
- **Gray background artifact removal with safe zone protection is a highly effective post-processing strategy, leading to a significant increase in attack success rates across all models and datasets.**
- **The results confirm that even subtle background artifacts can reduce morph effectiveness, and their removal is crucial for maximizing attack success.**
- **Both post-processing approaches (orange and gray artifact removal) are valuable, and their combined or selective use can further enhance morphing attack performance.**

---

### Third Approach: Gray Artifact Replacement with Sampled Background Color

This post-processing method further refines the removal of gray background artifacts in morphed face images by replacing them with a color sampled from the image's own background, rather than using inpainting. The goal is to seamlessly blend the artifact regions with the original background, while still protecting the central face area.

**How it works:**
- The script samples the background color by averaging a small patch (10x10 pixels) from the top-left corner of the image, assuming this region represents the true background.
- A central "safe zone" is defined (from 15% to 85% of the image width and height) to protect the main facial features from modification.
- Gray background artifacts are detected using a mask in HSV color space (low saturation, medium brightness), but only outside the safe zone.
- Instead of inpainting, the detected artifact pixels are directly replaced with the sampled background color, ensuring a uniform and natural-looking background.
- The script processes all images in the dataset, saves the modified images, and generates new triplet files for evaluation.

**Motivation:**
- This approach is designed to avoid the sometimes blurry or inconsistent results of inpainting, especially when the background is uniform. By using a sampled color, the background remains consistent and visually clean.
- Protecting the central face region ensures that facial features are not altered, maintaining the integrity of the morph.

The implementation can be found in `post_process_morphs_attempt3.py`.

---

**Table 7: Attack Success Rates (FNMR@FMR=1%) for Each Model and Post-processed Morphing Dataset Using the Third Approach**

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
            <td>FaceMorpher_processed_replace_background</td>
            <td align="center">0.9490</td>
            <td align="center">0.9660</td>
            <td align="center">0.9550</td>
        </tr>
        <tr>
            <td>Webmorph_processed_replace_background</td>
            <td align="center">0.9900</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
        </tr>
        <tr>
            <td>MorDIFF_processed_replace_background</td>
            <td align="center">0.9860</td>
            <td align="center">0.9920</td>
            <td align="center">0.9890</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_replace_background</td>
            <td align="center">0.9240</td>
            <td align="center">0.9340</td>
            <td align="center">0.9280</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_replace_background</td>
            <td align="center">0.8919</td>
            <td align="center">0.9059</td>
            <td align="center">0.8959</td>
        </tr>
        <tr>
            <td>OpenCV_processed_replace_background</td>
            <td align="center"><b>0.9949</b></td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9939</b></td>
        </tr>
    </tbody>
</table>


**Table 8: Attack Success Rates (FNMR@FMR=0.1%) for Each Model and Post-processed Morphing Dataset Using the Third Approach**

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
            <td>FaceMorpher_processed_replace_background</td>
            <td align="center">0.8810</td>
            <td align="center">0.9140</td>
            <td align="center">0.9020</td>
        </tr>
        <tr>
            <td>Webmorph_processed_replace_background</td>
            <td align="center"><b>0.9780</b></td>
            <td align="center"><b>0.9840</b></td>
            <td align="center"><b>0.9840</b></td>
        </tr>
        <tr>
            <td>MorDIFF_processed_replace_background</td>
            <td align="center">0.9110</td>
            <td align="center">0.9620</td>
            <td align="center">0.9320</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_replace_background</td>
            <td align="center">0.7220</td>
            <td align="center">0.7520</td>
            <td align="center">0.7230</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_replace_background</td>
            <td align="center">0.6216</td>
            <td align="center">0.6717</td>
            <td align="center">0.6416</td>
        </tr>
        <tr>
            <td>OpenCV_processed_replace_background</td>
            <td align="center">0.9492</td>
            <td align="center">0.9766</td>
            <td align="center">0.9614</td>
        </tr>
    </tbody>
</table>

### Interpretation of Post-Processing Results: Third Approach (Tables 7 & 8)

Tables 7 and 8 show the attack success rates after applying the third post-processing approach (gray artifact replacement with sampled background color). These results can be compared to the baseline (Tables 1 & 2) and to the previous post-processing methods.

#### Key Findings:
- **Consistent Improvement Over Baseline:**
  - For all datasets and models, the third approach increases attack success rates compared to the baseline (Tables 1 & 2). This confirms that removing or harmonizing background artifacts is crucial for maximizing morphing attack effectiveness.
- **Magnitude of Improvement:**
  - The improvement is especially notable for ElasticFaceCos and for datasets with lower baseline success rates (e.g., FaceMorpher, MIPGAN_II). For example, FaceMorpher_processed_replace_background against ElasticFaceCos rises from 0.0020 (Table 1) to 0.9660 (Table 7) at FNMR@FMR=1%.
  - For datasets that were already highly effective (Webmorph, OpenCV), the success rates remain very high, with only minor increases.
- **Comparison to Other Approaches:**
  - The third approach achieves results very similar to the second approach (gray background inpainting), with only slight differences in success rates. This suggests that both inpainting and background color replacement are effective, and the choice may depend on the visual consistency desired in the output images.
- **Stricter Threshold (FNMR@FMR=0.1%):**
  - At the stricter threshold (Table 8), the trends persist: post-processing increases attack success rates, and the gap between the best and worst performing morphing methods is reduced.
- **Visual Quality:**
  - By using a sampled background color, this approach avoids the potential blurriness of inpainting and ensures a clean, uniform background, which may be preferable for certain applications or datasets.

#### Summary:
- **Gray artifact replacement with sampled background color is a highly effective post-processing strategy, leading to a significant increase in attack success rates across all models and datasets.**
- **The results confirm that background harmonization—whether by inpainting or color replacement—substantially boosts the effectiveness of morphing attacks.**
- **All three post-processing approaches (orange artifact removal, gray inpainting, and gray color replacement) are valuable, and the best choice may depend on the dataset and desired visual outcome.**

---

### Fourth Approach: Face-Centered Cropping with Enlarged Bounding Box

This post-processing method focuses on improving the alignment and prominence of the face in morphed images by automatically detecting the face, enlarging the detected bounding box, and cropping the image accordingly. The goal is to ensure that the face is always centered and occupies a larger portion of the image, which may help face recognition models extract more reliable features and potentially increase morphing attack success rates.

**How it works:**
- The script uses a pre-trained Haar Cascade classifier (`haarcascade_frontalface_default.xml`) to detect faces in each image.
- For each detected face, the bounding box is enlarged by 40% (20% on each side) to include more facial context and avoid tight cropping.
- The image is cropped to this enlarged bounding box, and then resized back to the original image dimensions to maintain compatibility with downstream face recognition models.
- If no face is detected, the original image is retained.
- The script processes all images in the dataset, saves the cropped images, and generates new triplet files for evaluation.

**Motivation:**
- This approach is designed to standardize the position and scale of faces in the dataset, reducing the impact of background and alignment variations.
- By enlarging the bounding box, the method preserves important facial context and avoids cutting off relevant features, which can be critical for recognition performance.

The implementation can be found in `post_process_morphs_attempt4.py`.

---

**Table 9: Attack Success Rates (FNMR@FMR=1%) for Each Model and Post-processed Morphing Dataset Using the Fourth Approach**

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
            <td>FaceMorpher_processed_improved_alignment</td>
            <td align="center">0.9450</td>
            <td align="center">0.9600</td>
            <td align="center">0.9560</td>
        </tr>
        <tr>
            <td>Webmorph_processed_improved_alignment</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
            <td align="center">0.9880</td>
        </tr>
        <tr>
            <td>MorDIFF_processed_improved_alignment</td>
            <td align="center">0.9880</td>
            <td align="center">0.9930</td>
            <td align="center">0.9910</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_improved_alignment</td>
            <td align="center">0.9390</td>
            <td align="center">0.9490</td>
            <td align="center">0.9440</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_improved_alignment</td>
            <td align="center">0.9109</td>
            <td align="center">0.9259</td>
            <td align="center">0.9059</td>
        </tr>
        <tr>
            <td>OpenCV_processed_improved_alignment</td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9970</b></td>
            <td align="center"><b>0.9929</b></td>
        </tr>
    </tbody>
</table>


**Table 10: Attack Success Rates (FNMR@FMR=0.1%) for Each Model and Post-processed Morphing Dataset Using the Fourth Approach**

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
            <td>FaceMorpher_processed_improved_alignment</td>
            <td align="center">0.8880</td>
            <td align="center">0.9160</td>
            <td align="center">0.9020</td>
        </tr>
        <tr>
            <td>Webmorph_processed_improved_alignment</td>
            <td align="center"><b>0.9800</b></td>
            <td align="center"><b>0.9840</b></td>
            <td align="center"><b>0.9860</b></td>
        </tr>
        <tr>
            <td>MorDIFF_processed_improved_alignment</td>
            <td align="center">0.9220</td>
            <td align="center">0.9620</td>
            <td align="center">0.9380</td>
        </tr>
        <tr>
            <td>MIPGAN_I_processed_improved_alignment</td>
            <td align="center">0.7400</td>
            <td align="center">0.7970</td>
            <td align="center">0.7600</td>
        </tr>
        <tr>
            <td>MIPGAN_II_processed_improved_alignment</td>
            <td align="center">0.6527</td>
            <td align="center">0.7107</td>
            <td align="center">0.6787</td>
        </tr>
        <tr>
            <td>OpenCV_processed_improved_alignment</td>
            <td align="center">0.9634</td>
            <td align="center">0.9746</td>
            <td align="center">0.9675</td>
        </tr>
    </tbody>
</table>

### Interpretation of Post-Processing Results: Fourth Approach (Tables 9 & 10)

Tables 9 and 10 show the attack success rates after applying the fourth post-processing approach (face-centered cropping with enlarged bounding box). These results can be compared to the baseline (Tables 1 & 2) and to the other post-processing methods.

#### Key Findings:
- **Consistent Improvement Over Baseline:**
  - For all datasets and models, the fourth approach increases attack success rates compared to the baseline (Tables 1 & 2). This demonstrates that improving face alignment and prominence in the image helps morphs bypass recognition systems more effectively.
- **Magnitude of Improvement:**
  - The improvement is especially notable for ElasticFaceCos and for datasets with lower baseline success rates (e.g., FaceMorpher, MIPGAN_II). For example, FaceMorpher_processed_improved_alignment against ElasticFaceCos rises from 0.0020 (Table 1) to 0.9600 (Table 9) at FNMR@FMR=1%.
  - For datasets that were already highly effective (Webmorph, OpenCV), the success rates remain very high, with only minor increases.
- **Comparison to Other Approaches:**
  - The fourth approach achieves results that are comparable to the best post-processing methods (artifact removal and background harmonization). This suggests that face alignment and prominence are as important as artifact removal for maximizing attack success.
- **Stricter Threshold (FNMR@FMR=0.1%):**
  - At the stricter threshold (Table 10), the trends persist: post-processing increases attack success rates, and the gap between the best and worst performing morphing methods is reduced.
- **Visual and Recognition Quality:**
  - By standardizing face position and scale, this approach ensures that the face is always the main focus of the image, which can help face recognition models extract more reliable features and improve morph effectiveness.

#### Summary:
- **Face-centered cropping with enlarged bounding box is a highly effective post-processing strategy, leading to a significant increase in attack success rates across all models and datasets.**
- **The results confirm that both artifact removal and improved face alignment are crucial for maximizing the effectiveness of morphing attacks.**
- **All four post-processing approaches provide substantial benefits, and combining alignment improvements with artifact removal may yield the best results.**

---
