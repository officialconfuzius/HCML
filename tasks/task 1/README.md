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
  - *Best attack:* OpenCV_aligned morphs, which almost always succeed regardless of the model or threshold.
  - *Worst attack:* FaceMorpher_aligned against ElasticFaceCos, and MIPGAN_II_aligned against CurricularFace.

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
            <td align="center">0.9800</td>
            <td align="center">0.9780</td>
            <td align="center">0.9780</td>
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
            <td align="center"><b>0.9289</b></td>
            <td align="center"><b>0.9644</b></td>
            <td align="center"><b>0.9461</b></td>
        </tr>
    </tbody>
</table>

## Subtask 2: Post-Process Morphs to Improve Attack Rates

Enhance existing morphs through both automatic and manual post-processing techniques to increase their similarity and, consequently, their attack success rates. This includes experimenting with different post-processing methods to optimize morph quality and effectiveness.

### First Approach: 
We began by focusing on the orange dots that appeared as artifacts in the morphed face images. Rather than simply deleting these pixels, we used a computer vision technique called inpainting to seamlessly replace them. This approach aimed to enhance the visual integrity of the morphed face. Since face recognition models are trained on millions of real faces, they are highly sensitive to unnatural features. Artifacts like orange dots can disrupt the model’s perception, lowering the similarity score. By removing these artifacts, we made the faces appear more natural to the model, which led to improved similarity scores.

Here’s a concise technical summary of the two steps:

#### Detection (Mask Creation):

The script converts the image from BGR to HSV color space, making it easier to isolate the orange color.
Using cv2.inRange(), it creates a binary mask: white pixels represent detected orange areas (artifacts), black pixels are everything else.

#### Reconstruction (Inpainting):

The script uses cv2.inpaint() with the mask to repair the orange areas.
The function analyzes the surrounding "healthy" pixels and fills in the masked (damaged) regions with new, natural-looking texture and color, seamlessly blending the repair.
This process detects and removes orange artifacts by algorithmically reconstructing those regions based on their surroundings.



### Second Approach
