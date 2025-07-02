## Subtask 1: Evaluate Existing Attacks

Assess the effectiveness of current attack methods by measuring their successful attack rates. This involves analyzing various attack strategies and quantifying their ability to bypass recognition systems.

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
            <td align="center">0.9470</td>
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
            <td align="center">0.9099</td>
            <td align="center">0.8989</td>
            <td align="center">0.0030</td>
        </tr>
        <tr>
            <td>OpenCV_aligned</td>
            <td align="center">0.9929</td>
            <td align="center">0.9959</td>
            <td align="center">0.9929</td>
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
