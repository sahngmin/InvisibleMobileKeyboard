## Invisible Mobile Keyboard (IMK) with Self-Attention Neural Character Decoder (SA-NCD)
Official data and Pytorch implementation of "Type Anywhere You Want: An Introduction to Invisible Mobile Keyboard" (IJCAI 2021, Accepted)

This repository provides IMK dataset, codes for training the baseline decoder SA-NCD. Please refer to our [paper]() 


### Dataset Download
The raw dataset is in $data/raw$
The model is trained on normalized dataset which can be downloaded from the [Google Drive](https://drive.google.com/file/d/1eP2ZnxI1zzvtyyr_iQ_AAnRzyXgV59bI/view?usp=sharing).
The normalized dataset contains a x, y location coordinates divided by the used device width and height, so they can be in range of [0, 1].

### Training SA-NCD Network (IMK Decoder)





### Data directory structure
```
Ref_Seq_
|
--- Warehouse_0                              # Environment folder
|       |
|       ---- Seq_0                           # Sequece
|       |      |
|       |      +--- rgb                      # 0.png - xxxx.png      
|       |      +--- depth                    # 0.png - xxxx.png
|       |      +--- semantic_segmentation    # 0.png - xxxx.png     
|       |      ---- raw                   
|       |      |     |
|       |      |     +--- rgb                # 0.png - xxxx.png
|       |      |     +--- depth              # 0.png - xxxx.png
|       |      |     ---- poses.g2o 
|       |      |     ---- rtabmap.yaml
|       |
|       +--- Seq_1
|
+-- Warehouse_1
.
.
+-- Warehouse_N



## Citation
If you find this project helpful, please consider citing this project in your publications. The following is the BibTeX of our work.

```bibtex

```
