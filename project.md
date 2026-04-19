# Student Project: Deep Learning Pipeline for Strawberry Harvesting
**AIDA 2158A: Neural Networks and Deep Learning** **Instructor:** Dr. M. Tufail  
**Institution:** Red Deer Polytechnic

---

## 1 Project Description
This project focuses on developing a complete deep learning pipeline for strawberry harvesting using RGB images, with emphasis on both fruit detection and stem understanding for future robotic grasp planning. 

### Key Workflow:
1.  **Detection:** Students will train and validate a **YOLOv11 segmentation model** to detect strawberry instances from an existing annotated dataset.
2.  **Target Selection:** The model identifies the **largest visible ripe strawberry** in each image as the target fruit.
3.  **ROI Generation:** A Region of Interest (ROI) is automatically cropped around the detected strawberry and its surrounding biologically relevant pixels.
4.  **Local Learning:** This smaller ROI is used for subsequent stem-region learning, allowing the segmentation model to focus on the local area rather than the full image.
5.  **Annotation:** Students will manually annotate the first 100 cropped ROI images using the **Digital Sreeni annotation tool** (integrated with SAM).
    * **Scope:** Segment the visible **crown–stem–peduncle** structure associated with the target strawberry (including the peduncle, calyx/leaf crown, and a small upper portion of the strawberry body).
    * **Exclusions:** Exclude unrelated fruits, stems, and leaves; do not guess hidden/ambiguous stem regions.
6.  **U-Net Training:** Manually generated masks will train a **U-Net model** for crown–stem–peduncle segmentation.
7.  **Orientation:** Stem orientation is extracted using geometric analysis such as **Principal Component Analysis (PCA)** or skeleton-based estimation.

The goal is to support robotic decisions: target confirmation, graspable stem localization, gripper alignment, and cutting direction.

---

## 2 Project Motivation
Autonomous fruit harvesting requires reliable perception of:
* Ripe strawberry localization
* Crown–stem–peduncle localization
* Stem orientation estimation
* Grasp point generation

---

## 3 Project Objectives
The objective is to detect:
* The largest visible red or partially ripe strawberry.
* Its biologically connected crown–stem region.
* A graspable local stem segment.

---

## 4 Available Dataset
Students will use a strawberry segmentation dataset containing:
* RGB strawberry images.
* Segmentation masks for strawberries.
* YOLOv11 segmentation labels.

**Link:** [Google Drive Dataset](https://drive.google.com/drive/folders/1wksHY4mml3ux2EpuG28tLfofGxwElWyX?usp=sharing)

---

## 5 Full Pipeline
1.  **YOLOv11-seg:** Detects strawberry instances.
2.  **Selection:** Largest red strawberry is selected automatically.
3.  **Crop:** Segmentation-based ROI crop is generated.
4.  **Annotation:** ROI manually annotated using SAM-assisted Digital Sreeni.
5.  **U-Net:** Learns crown–stem–peduncle masks inside ROI.
6.  **PCA:** Computes stem orientation.
7.  **Robotics:** Grasp direction is estimated.

---

## 6 Important Annotation Rule
For images containing multiple strawberries:
* **Ignore** all non-target strawberries.
* **Ignore** green fruits unless they belong to the target ROI.
* **Annotate** only the crown–stem–peduncle region of the selected target fruit.
* **Do not guess** hidden stem regions.

---

## 7 Example Annotation
**Figure 1: Target annotation example.**
The largest red strawberry is selected as the harvest target. Students should annotate the visible crown–stem–peduncle structure, including the peduncle segment, connected calyx, and a small upper portion of the strawberry body.

---

## 8 Peduncle Annotation Tool
Students must annotate the first 100 ROI images using:
**Digital Sreeni Annotation Tool** GitHub: [digitalsreeni-image-annotator](https://github.com/sreenidigital/digitalsreeni-image-annotator)

---

## 9 How to Annotate Using SAM in Digital Sreeni
1.  Load cropped ROI image.
2.  Select **SAM-assisted segmentation mode**.
3.  Draw bounding box around visible crown–stem–peduncle region.
4.  Run SAM segmentation.
5.  Refine mask manually if needed.
6.  Save binary mask (**White** = target region; **Black** = background).

---

## 10-13 Project Modules & Deadlines

| Module | Task | Deadline |
| :--- | :--- | :--- |
| **Module 1** | **YOLOv11-seg:** Train (yolo11s-seg.pt), validate, and write ROI crop code. | Week 1 |
| **Module 2** | **Manual Annotation:** Complete first 100 ROI masks. | Week 1 |
| **Module 3** | **U-Net Training:** Train model on cropped ROI masks. | Week 2 |
| **Module 4** | **Stem Angle Extraction:** Calculate angles using PCA. | Weeks 3–4 |

---

## 14 Submission of Results
Students must submit:
* Training curves and validation plots.
* Hyperparameter table.
* Annotation progress and ROI examples.
* U-Net outputs.

---

## 15 Evaluation Rubric

| Component | Marks |
| :--- | :--- |
| YOLOv11 training and validation | 20 |
| ROI generation quality | 15 |
| Annotation quality | 20 |
| U-Net training | 20 |
| Stem angle extraction | 15 |
| Final report and presentation | 10 |

---

## 16 Expected Learning Outcomes
* Deep learning segmentation (YOLO & U-Net).
* ROI generation and SAM-assisted annotation.
* Semantic segmentation techniques.
* Geometric reasoning for robotics applications.