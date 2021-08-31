## Invisible Mobile Keyboard (IMK) with Self-Attention Neural Character Decoder (SA-NCD)
Official data and Pytorch implementation of "Type Anywhere You Want: An Introduction to Invisible Mobile Keyboard" (IJCAI 2021, Accepted)

This repository provides IMK dataset, codes for training the baseline decoder SA-NCD. Please refer to our [paper](https://www.ijcai.org/proceedings/2021/0242.pdf) 

Thanks to [huggingface](https://github.com/huggingface/transformers) for a reference of transformers source code.


### Dataset Download
The raw dataset is in ```data/raw```.
The model is trained on normalized dataset which can be downloaded from the [Google Drive](https://drive.google.com/file/d/1eP2ZnxI1zzvtyyr_iQ_AAnRzyXgV59bI/view?usp=sharing).
The normalized dataset contains a x, y location coordinates divided by the used device width and height, so they can be in range of [0, 1].

### Training SA-NCD Network (IMK Decoder)




### Data Construction
```


```


### IMK Decoder Implementation (Test Video)
https://user-images.githubusercontent.com/49089477/131445988-ca1ae0c8-0ae8-4a14-b292-1ec6ffbdfc3a.mp4


### Citation
If you find this project helpful, please consider citing this project in your publications. The following is the BibTeX of our work.

```bibtex
@misc{yoo2021type,
      title={Type Anywhere You Want: An Introduction to Invisible Mobile Keyboard}, 
      author={Sahng-Min Yoo and Ue-Hwan Kim and Yewon Hwang and Jong-Hwan Kim},
      year={2021},
      eprint={2108.09030},
      archivePrefix={arXiv},
      primaryClass={cs.HC}
}
```
