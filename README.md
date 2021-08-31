## Invisible Mobile Keyboard (IMK) with Self-Attention Neural Character Decoder (SA-NCD)
Official data and Pytorch implementation of "Type Anywhere You Want: An Introduction to Invisible Mobile Keyboard" (IJCAI 2021, Accepted)

This repository provides IMK dataset, codes for training the baseline decoder SA-NCD. Please refer to our [paper](https://www.ijcai.org/proceedings/2021/0242.pdf).
Thanks to [huggingface](https://github.com/huggingface/transformers) for a reference of transformers source code.


### Dataset Download
The raw dataset is in ```data/raw```.
The model is trained on normalized dataset which can be downloaded from the [Google Drive](https://drive.google.com/file/d/1eP2ZnxI1zzvtyyr_iQ_AAnRzyXgV59bI/view?usp=sharing).
The normalized dataset contains a x, y location coordinates divided by the used device width and height, so they can be in range of [0, 1].



### Data Construction
The figure below is an example of IMK data. The dataset includes user index, age, device type, typed text, coordinate values of the typed position as a list, size of the screen, and time taken for typing each phrase.
<img width="702" alt="dataset_example" src="https://user-images.githubusercontent.com/49089477/131505405-14baba75-18b4-4240-a7a2-77b39e28fac3.png">


### Training SA-NCD Network (IMK Decoder)
```
conda env create --file environment_imk.yaml
python train.py 
```


### IMK Decoder Implementation (Test Video)
The video is an example of typing "thank you for your help." on a web-implemented Invisible Mobile Keyboard using SA-NCD as a built-in decoder. Note that the decoded output can post-correct its typo by considering the additional input context.

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
