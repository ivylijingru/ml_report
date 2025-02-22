# Project Proposal

## Video Preview
[Video YouTube Link](https://youtu.be/_46bmIG7S_A)

## Introduction/Background
Singing Voice Transcription (SVT) converts singing audio into musical notes by detecting onset, pitch (MIDI), and offset, with applications in music education, composition, and retrieval. Traditional models rely on supervised learning, requiring costly, large labeled datasets. Recent advancements in self-supervised learning, such as MERT [[1]](#ref1), have improved transcription accuracy by learning musical representations from large-scale unlabeled data. This project fine-tunes MERT on the manually validated MIR-ST500 dataset [[2]](#ref2), which contains 500 pop songs and 162,438 annotated notes, aiming to enhance transcription accuracy while reducing dependency on labeled data.  

## Problem Definition
SVT remains challenging due to variations in vocal performance, background noise, and the need for large, costly labeled datasets, as many supervised models struggle with generalization. To address this, we propose fine-tuning MERT to leverage pre-trained musical representations, improving feature extraction and transcription accuracy. This approach reduces the reliance on large annotated datasets, making the solution more scalable and efficient.

## Methods
We finetune MERT for SVT following the methodology outlined in **Toward Leveraging Pre-Trained Self-Supervised Frontends** [[3]](#ref3). This involves extracting MERT embeddings, training a lightweight classifier, and evaluating performance using standard SVT metrics such as COnPOff, COnP, and COn. Additionally, we apply K-means clustering to visualize the learned representations and assess how well MERT organizes singing voice features.

### Data Preprocessing Methods
- **Audio Resampling**: All audio files are standardized to 16kHz using `librosa.resample()` to ensure uniformity in feature extraction.
- **Audio Random Slicing**: Each song is segmented into 5-second non-overlapping clips using `torchaudio.transform.Vad()`, allowing the model to focus on smaller audio chunks and improving generalization.
- **Embedding Extraction**: Fine-tuned MERT embeddings are extracted, providing a high-dimensional feature space that captures musical elements such as pitch and rhythm.

### Machine Learning Methods
- **Transformer-Based Model (MERT)**: MERT serves as a feature extractor, using its pretrained musical representations to identify pitch and onset patterns.
- **Linear Classifier**: A fully connected layer implemented with `torch.nn.Linear` is fine-tuned on labeled onset, pitch, and offset data, predicting note sequences from the extracted embeddings. We will first freeze the parameters of the pretrained MERT model and make them learnable for the linear classifier. After several epochs, we will unfreeze the Transformer encoders and fine-tune them.
- **Clustering Algorithm**: `sklearn.cluster.KMeans` is used to visualize MERT embeddings and explore clustering patterns in singing voice features, helping analyze how well the model distinguishes different musical elements.

## (Potential) Results and Discussion

### Quantitative Metrics
To evaluate the performance of our fine-tuned MERT model for singing voice transcription, we will use the following standard SVT metrics:

- **COnPOff (Correct Onset, Pitch, and Offset)**: Measures strict note accuracy by requiring correct onset, pitch, and offset predictions. This metric ensures that the transcribed notes align precisely with ground truth.
- **COnP (Correct Onset and Pitch)**: Evaluates whether the model correctly predicts note onset and pitch, allowing some flexibility with offset detection.
- **COn (Correct Onset)**: Focuses only on onset detection accuracy, assessing how well the model identifies when a note starts.

These metrics are computed using the `mir_eval` library with tolerances of 50ms for onset and offset and 50 cents for pitch.

### Project Goals
- **Achieve High Transcription Accuracy** – Improve COnPOff, COnP, and COn scores compared to baseline models such as EfficientNet-b0 and JDCnote.
- **Reduce Dependence on Labeled Data** – Fine-tune MERT efficiently using a smaller labeled dataset, demonstrating the effectiveness of self-supervised learning for SVT.

### Expected Results
- Fine-tuned MERT is expected to outperform fully supervised models in COnP and COn metrics, as self-supervised embeddings enhance feature extraction.
- Using K-means clustering for embedding visualization may reveal distinct note-grouping patterns, validating MERT's learned musical representations.

## References

<a id="ref1"></a>  
[1] Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Xiao, C. Lin, A. Ragni, E. Benetos, N. Gyenge, R. Dannenberg, R. Liu, W. Chen, G. Xia, Y. Shi, W. Huang, Z. Wang, Y. Guo, and J. Fu, “MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training,” *arXiv (Cornell University)*, May 2023, doi: [https://doi.org/10.48550/arxiv.2306.00107](https://doi.org/10.48550/arxiv.2306.00107).

<a id="ref2"></a>  
[2] J. Y. Wang and J. S. R. Jang, "On the Preparation and Validation of a Large-Scale Dataset of Singing Transcription," *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Toronto, ON, Canada, 2021, pp. 276-280, doi: [10.1109/ICASSP39728.2021.9414601](https://doi.org/10.1109/ICASSP39728.2021.9414601).

<a id="ref3"></a>  
[3] Y. Yamamoto, "Toward Leveraging Pre-Trained Self-Supervised Frontends for Automatic Singing Voice Understanding Tasks: Three Case Studies," *2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, Taipei, Taiwan, 2023, pp. 1745-1752, doi: [10.1109/APSIPAASC58517.2023.10317286](https://doi.org/10.1109/APSIPAASC58517.2023.10317286).

# Proposal Contributions

| **Name**   | **Proposal Contributions** |
|------------|---------------------------|
| Jingru Li    | Designed the proposal idea and developted report webpage              |
| Andrew Wang   | Created Slides and filmed video               |
| Anish Arora        | Created Slides                         |
| Youhan Li        | Drafted and refined the proposal    |
| Xinni Li        | Refined the proposal and developted github page |

# Gantt Chart

<!-- [Access the Gantt Graph here](./GanttChart_Group34.xlsx) -->

| TASK TITLE                               | TASK OWNER                          | START DATE | DUE DATE | DURATION |
|------------------------------------------|-------------------------------------|------------|----------|----------|
| Project Team Composition                 | All                                 | 1/17/24    | 2/2/24   | 15       |
| **Project Proposal**                      |                                     |            |          |          |
| Introduction & Background                | Jingru Li & Youhan Li               | 2/2/24     | 2/21/24  | 19       |
| Problem Definition                       | Jingru Li, Youhan Li & Xinni Li     | 2/2/24     | 2/21/24  | 19       |
| Methods                                  | Jingru Li & Youhan Li               | 2/2/24     | 2/21/24  | 19       |
| Potential Dataset                        | Jingru Li & Youhan Li               | 2/2/24     | 2/21/24  | 19       |
| Potential Results & Discussion           | Jingru Li & Youhan Li               | 2/2/24     | 2/21/24  | 19       |
| Video Creation & Recording               | Andrew Wang & Anish Arora           | 2/10/24    | 2/21/24  | 11       |
| GitHub Page                              | Jingru Li & Xinni Li                | 2/10/24    | 2/21/24  | 11       |
| **Midterm Report**                        |                                     |            |          |          |
| Audio Resampling (16kHz)                 | Anish Arora                         | 2/22/24    | 2/24/24  | 2        |
| Audio Slicing (5s clips)                 | Andrew Wang                         | 2/25/24    | 2/27/24  | 2        |
| MERT Embedding Extraction                | Jingru Li                           | 2/27/24    | 2/28/24  | 1        |
| Frozen MERT + Linear Classifier          | Youhan Li                           | 3/2/24     | 3/8/24   | 6        |
| Unfrozen MERT Encoder Tuning             | Xinni Li                            | 3/9/24     | 3/15/24  | 6        |
| K-means Clustering                       | Jingru Li & Youhan Li               | 3/16/24    | 3/24/24  | 8        |
| Model Evaluation (COnPOff/COnP/COn)      | Xinni Li                            | 3/25/24    | 3/27/24  | 2        |
| Results Visualization                    | Jingru Li & Youhan Li               | 3/28/24    | 3/30/24  | 2        |
| Report Consolidation                     | Anish Arora & Andrew Wang           | 3/28/24    | 3/31/24  | 3        |
| **Final Report**                          |                                     |            |          |          |
| Baseline Comparison (EfficientNet-b0/JDCnote) | Jingru Li                     | 4/1/24     | 4/5/24   | 4        |
| Labeled Data Efficiency Analysis         | Youhan Li                           | 4/6/24     | 4/10/24  | 4        |
| Clustering Pattern Validation            | Xinni Li                            | 4/11/24    | 4/14/24  | 3        |
| Model Integration & Testing              | Jingru Li & Youhan Li               | 4/15/24    | 4/17/24  | 2        |
| Demo Video & GitHub Update               | Anish Arora & Andrew Wang           | 4/17/24    | 4/22/24  | 5        |
| Final Report                             | All                                 | 4/18/24    | 4/23/24  | 5        |
