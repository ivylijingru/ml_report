# Final Report

## Introduction/Background
Singing Voice Transcription (SVT) converts singing audio into musical notes by detecting onset, pitch (MIDI), and offset, with applications in music education, composition, and retrieval. Traditional models rely on supervised learning, requiring costly, large labeled datasets. Recent advancements in self-supervised learning, such as MERT [[1]](#ref1), have improved transcription accuracy by learning musical representations from large-scale unlabeled data. This project fine-tunes MERT on the manually validated MIR-ST500 dataset [[2]](#ref2), which contains 500 pop songs and 162,438 annotated notes, aiming to enhance transcription accuracy while reducing dependency on labeled data.  

## Problem Definition
SVT remains challenging due to variations in vocal performance, background noise, and the need for large, costly labeled datasets, as many supervised models struggle with generalization. To address this, we propose fine-tuning MERT to leverage pre-trained musical representations, improving feature extraction and transcription accuracy. This approach reduces the reliance on large annotated datasets, making the solution more scalable and efficient.

## Methods
We finetune MERT for SVT following the methodology outlined in **Toward Leveraging Pre-Trained Self-Supervised Frontends** [[3]](#ref3). This involves extracting MERT embeddings, training a lightweight classifier, and evaluating performance using standard SVT metrics such as COnPOff, COnP, and COn. Additionally, we apply K-means clustering to visualize the learned representations and assess how well MERT organizes singing voice features.

### Data Preprocessing Methods (Supervised)
- **Vocal Source Separation**: We started by applying a source separation model to the original MIR-ST500 audio tracks. This isolates the vocal stem from background accompaniment, allowing us to focus purely on singing voice features for downstream analysis. We applied a source separation model called [Demucs](https://github.com/facebookresearch/demucs).
- **Audio Resampling**: All audio files are standardized to 16kHz using `librosa.resample()` to ensure uniformity in feature extraction.
- **Audio Random Slicing**: Each song is segmented into 5-second non-overlapping clips using `torchaudio.transform.Vad()`, allowing the model to focus on smaller audio chunks and improving generalization.
- **Embedding Extraction**: Fine-tuned MERT embeddings are extracted, providing a high-dimensional feature space that captures musical elements such as pitch and rhythm.

### Machine Learning Methods (Supervised)
- **Transformer-Based Model (MERT)**: MERT serves as a feature extractor, using its pretrained musical representations to identify pitch and onset patterns.
- **Linear Classifier**: A fully connected layer implemented with `torch.nn.Linear` is fine-tuned on labeled onset, pitch, and offset data, predicting note sequences from the extracted embeddings. We will first freeze the parameters of the pretrained MERT model and make them learnable for the linear classifier. After several epochs, we will unfreeze the Transformer encoders and fine-tune them.



## Data Preprocessing (Unsupervised)

### 1. Vocal Source Separation  
We started by applying a source separation model to the original MIR-ST500 audio tracks.  
This isolates the vocal stem from background accompaniment, allowing us to focus purely on singing voice features for downstream analysis.

### 2. Converting Ground Truth to Pitch Class Labels  
The original label format in `mirst500-train.json` is as follows:
"1": [[17.839583, 18.095866, 58.0], [18.129167, 18.527083, 61.0], ...]
Each item contains:
- **start_time (s)** – when the note begins  
- **end_time (s)** – when the note ends  
- **MIDI_pitch** – the pitch number (e.g., 60 = C4)

We convert each MIDI pitch to its **pitch class** by computing `MIDI % 12`, resulting in 12 categories:  
**C, C#, D, D#, E, F, F#, G, G#, A, A#, B**

The converted labels (without octave information) are saved in the file:  
_**MIR_ST500_pitchclass.json**_

This transformation simplifies the problem and helps us focus on pitch categories rather than exact pitches, which is more meaningful for unsupervised clustering tasks.


### 3. Selecting a Balanced Clip for Analysis

After pitch class conversion, we analyzed all clips in the dataset and selected one with:
- Diverse pitch classes
- A relatively balanced distribution across them

We chose:  
_**Clip ID: 172**_

This clip is used as our test example for representation extraction and clustering analysis.


### 4. Audio Loading and Resampling

We load the vocal track for clip 172 using `soundfile.read()`, then resample it to 16kHz using `torchaudio.transforms.Resample()`  
This ensures compatibility with MERT's expected input format.


### 5. Extracting MERT Embeddings

The full audio is passed into the MERT model to extract frame-level embeddings.

- **Output shape:** `(T, 768)` where `T` is the number of time frames (50Hz frame rate)
- **Saved as:**  
  _**clip_172_full_repr.npy**_

These embeddings capture musical features like pitch, onset, and timbre in a high-dimensional latent space.


### 6. Aligning Embeddings with Pitch Class Labels

Using the time-aligned annotations in _**MIR_ST500_pitchclass.json**_, we:
- Convert time intervals to frame indices (based on 50Hz)
- Map each frame in the MERT embedding to its corresponding pitch class


### 7. Sampling Embeddings per Pitch Class

To construct a clean and balanced dataset:
- We randomly sample 100 frames per pitch class
- Each pitch class maps to a NumPy array of shape `(100, 768)`
- Saved as:
  _**pitchclass_samples.pkl**_

This file will be used in the next phase: unsupervised clustering, dimensionality reduction, visualization, and evaluation.

---

## Next Steps: Unsupervised Learning

### 1. Dimensionality Reduction
We’ll reduce the 768-dim vectors to 2D (e.g., using t-SNE, PCA, or UMAP) so we can visualize the data.  
This will help us see whether embeddings from the same pitch class naturally group together.

### 2. Clustering
We applied t-SNE to reduce the original 768-dimensional MERT embeddings down to 2D for visualization.
The goal is to check whether the learned embeddings can be automatically grouped into 12 clusters, aligning with the 12 pitch classes.
In the 2D plot (left), we can observe visibly clustered regions corresponding to certain pitch classes, suggesting that MERT embeddings encode pitch-related structure to some degree.

### 3. Visualization
We visualize the reduced 2D embeddings, coloring the points by:
Left: Embeddings colored by their true pitch class label
Right: Embeddings colored by their predicted cluster assignment from KMeans

![tsne_cluster_plot](https://raw.githubusercontent.com/ivylijingru/ml_report/main/tsne_cluster.png)

From the visual comparison:
Several pitch classes (e.g., A, B) exhibit well-defined, isolated clusters.
Some pitch classes (e.g., C#, A#) appear more entangled or split across multiple clusters.
Certain clusters align well with pitch semantics, though overlaps and misgrouping are present.

### 4. Evaluation
We’ll compute clustering quality metrics to find how well the clustering aligns with actual pitch classes.
The ARI of 0.44 indicates moderate agreement between predicted clusters and ground truth pitch classes.
The NMI of 0.63 suggests that about 63% of the mutual information between embeddings and pitch class labels is preserved.

If MERT embeddings cluster well by pitch class, it means the model has already captured pitch-related information without supervision.  
This supports the idea that self-supervised models like MERT reduce the need for large labeled datasets, and still produce musically meaningful features.


## Finetuned Model Evaluation (SVT Metrics)

We evaluated the finetuned MERT model for singing voice transcription. The results are summarized below (from `svt_mert_finetune_debug_smaller_bs32_weight_498_res_reproduce.txt`):

| Metric   | Precision | Recall   | F1-score |
|----------|-----------|----------|----------|
| COnPOff  | 0.479789  | 0.491418 | 0.484970 |
| COnP     | 0.702347  | 0.720901 | 0.710672 |
| COn      | 0.758610  | 0.777980 | 0.767256 |

_**gt note num:** 31311.0_  
_**tr note num:** 32015.0_

These results reflect standard SVT metrics used in the community:
- **COnPOff**: correct onset, pitch, and offset  
- **COnP**: correct onset and pitch  
- **COn**: correct onset

## Post-hoc Analysis of Self-Supervised Embeddings via Dimensionality Reduction & Clustering
Although the training of MERT and its variants is self-supervised, we perform a post-hoc analysis using pitch class labels to examine whether the learned embeddings organize musically meaningful structures. Specifically, we reduce the 768-dimensional MERT embeddings to 2D using t-SNE and apply KMeans clustering (k=12). These clusters are then compared to pitch class labels (only for evaluation, not for training). The t-SNE visualizations reveal that the quality of clustering in the latent space correlates strongly with the overall performance of the models in the downstream SVT task.

### Original MERT (No Finetuning)
![MERT](https://raw.githubusercontent.com/ivylijingru/ml_report/main/MERT%20%2B%20Linear%20head_tsne_clusters_Self_SL.png)
</div>
The original MERT model, without any fine-tuning, exhibits relatively entangled and overlapping clusters. This indicates that while self-supervised pretraining captures some pitch-relevant information, the separation of pitch classes in the embedding space is limited.

### MERT + LoRA
![MERT_LORA](https://raw.githubusercontent.com/ivylijingru/ml_report/main/lora_tsne_clusters_Self_SL.png)
</div>
MERT + LoRA shows moderately improved clustering. The boundaries between pitch classes are clearer compared to the base MERT, although certain clusters still remain mixed. This matches its slight improvement in COn and COnP metrics.


### MERT + Linear Head (Finetuned)
![MERT_FINETUNE](https://raw.githubusercontent.com/ivylijingru/ml_report/main/MERT%20%2B%20Linear%20head%20%2B%20Finetune_tsne_clusters_Self_SL.png)
The MERT + Linear Head (Finetuned) model demonstrates the most distinct and compact clustering structure. Each pitch class forms a well-defined group, indicating that fine-tuning with even a simple linear head leads to more pitch-sensitive embeddings. This visually aligns with its superior performance across all metrics, especially in the challenging COnPOff task.

These visual results further support the numerical findings: fine-tuning enhances not only task-specific accuracy but also the intrinsic structure of the learned representation space, making it more semantically organized by pitch.

## Large Files & Model Checkpoints

The following files are hosted on Google Drive due to their size:

- Audio files of clip 172: Mixture.mp3 and Vocals.wav
- `clip_172_full_repr.npy` — full-frame MERT representation of clip 172  
- Finetuned MERT model checkpoint and config files

**Access them here:**  
[Google Drive - MERT Finetuned Files](https://drive.google.com/drive/folders/1X1SSe9KMDvmhg-VaSxrW1zLPEeCnTdU-?usp=sharing)

# Visualization
Our results compared to all Singing Voice Transcrition models in **Toward Leveraging Pre-Trained Self-Supervised Frontends** [[3]]
![output-2](https://raw.githubusercontent.com/ivylijingru/ml_report/main/output-2.png)

Our results compared to just MERT
![output](https://raw.githubusercontent.com/ivylijingru/ml_report/main/output.png)

Comparison between MERT + Linear head, MERT + LoRA, MERT
![comparison](https://raw.githubusercontent.com/ivylijingru/ml_report/main/Comparison.png)

As we can see in the image above the models all performed quite well based on F1 score, recall, and precision on COn. They then performed slightly worse on COnP and quite poorly on COnPOff. Further, we observed that the MERT + Linear head significantly outperformed both MERT and MERT + LoRA on all 9 possible fields with the outperformance especially notable in the most difficult task of COnPOff. Finally, MERT + LoRA outperformed MERT on both COn and COnP but was worse on COnPOff. Note that we can compare across recall, F1 and precision freely in this situation since all 3 of those had the same relative ordering of the models meaning that none of the algorithms were biased toward one type of prediction. 


# Results & Discussion
The results more or less match the F1-score's that were produced in **Toward Leveraging Pre-Trained Self-Supervised Frontends** [[3]](#ref3). Our F1-score for COnPOff of 0.484970 outperforms the reference paper by around 1.8% while the F1-score's for COnP and COn slightly underperformed the paper's scores by around 0.5% and 1.5% respectively. Additionally, the MERT fine tuning from the paper outperformed all other models (both SSL and conventional) on COnP and our fine-tuning similarly outperformed all other methods. Since these numbers are all around the same our fine-tuning is a very accurate replication of the paper. One of the choices that we made that could have helped these results would be the fact that we didn't care about octave the way that was done in the original paper. Since the metrics were only focused on pitch and timing we wouldn't be predicting a larger target vector than we needed. The next steps that we intend to take based on this is continue to focus on the clustering and unsupervised learning as well as trying out a different method to continue to improve on these results.
We compared three models on note boundary prediction (COnPOff, COnP, COn). The first model was MERT + Linear head (finetuned) which achieved the best overall performance, especially on the most challenging COnPOff metric. Next,	MERT + LoRA (finetuned) came second, with relatively strong recall. Finally, original MERT (no finetuning) performed the worst, showing that task-specific adaptation is necessary. All models performed best on COn, suggesting that detecting note onsets is easier than predicting full note boundaries. We conclude that finetuning significantly improves MERT’s performance, and the simple linear head outperforms LoRA in this task.

## References

<a id="ref1"></a>  
[1] Y. Li, R. Yuan, G. Zhang, Y. Ma, X. Chen, H. Yin, C. Xiao, C. Lin, A. Ragni, E. Benetos, N. Gyenge, R. Dannenberg, R. Liu, W. Chen, G. Xia, Y. Shi, W. Huang, Z. Wang, Y. Guo, and J. Fu, “MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training,” *arXiv (Cornell University)*, May 2023, doi: [https://doi.org/10.48550/arxiv.2306.00107](https://doi.org/10.48550/arxiv.2306.00107).

<a id="ref2"></a>  
[2] J. Y. Wang and J. S. R. Jang, "On the Preparation and Validation of a Large-Scale Dataset of Singing Transcription," *ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, Toronto, ON, Canada, 2021, pp. 276-280, doi: [10.1109/ICASSP39728.2021.9414601](https://doi.org/10.1109/ICASSP39728.2021.9414601).

<a id="ref3"></a>  
[3] Y. Yamamoto, "Toward Leveraging Pre-Trained Self-Supervised Frontends for Automatic Singing Voice Understanding Tasks: Three Case Studies," *2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)*, Taipei, Taiwan, 2023, pp. 1745-1752, doi: [10.1109/APSIPAASC58517.2023.10317286](https://doi.org/10.1109/APSIPAASC58517.2023.10317286).

# Final Checkpoint Contributions

| **Name**   | **Final Checkpoint Contributions** |
|------------|---------------------------|
| Jingru Li    | |
| Andrew Wang   |Created slides and gave presentation|
| Anish Arora        |Prepared report Results and Discussion |
| Youhan Li        |Processed representation data, extracted features and edited clides |
| Xinni Li        | | 

# Gantt Chart

[Access the Gantt Graph here](./GanttChart_Group34.xlsx
)
