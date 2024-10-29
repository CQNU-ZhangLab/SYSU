@[TOC](Modality-Aware Graph Convolutional Network for Visible-Infrared Person Re-Identification)
### Results on the SYSU-MM01 Dataset an the RegDB Dataset 
| Method | Datasets                    | Rank@1    | mAP       |
| ------ | --------------------------  | --------- | --------- |
| Ours   | #SYSU-MM01 (All-Search)     | ~ 76.02 % | ~ 72.31 % |
| Ours   | #SYSU-MM01 (Indoor-Search)  | ~ 81.02 % | ~ 81.49 % |
| Ours   | #RegDB (Visible-Infrared )  | ~ 94.51 % | ~ 94.63 % |
| Ours   | #RegDB (Infrared-Visible )  | ~ 93.79 % | ~ 94.50 % |

### **1. Prepare the datasets.**

- (1) RegDB Dataset [1]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (2) SYSU-MM01 Dataset [2]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py`  in to pepare the dataset, the training data will be stored in ".npy" format.

### 2. .Environmental requirements.

- python == 3.8

- PyTorch == 1.10

- torchvision == 0.4.0

### 3. Quick start.
- (1) Clone this repository:
[https://github.com/Vickyhehe/MAGCN.git](https://github.com/Vickyhehe/MAGCN.git)
 - (2) Modify the path to datasets:
 The path to datasets can be modified in the following file:
 `data_loader.py`
 - (3) Training:
 `--dataset`: which dataset "sysu" or "regdb".
  `--lr`: initial learning rate.
  `--gpu`:  which gpu to run.

You may need manually define the data path first.


### 4. References

```
[1] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.
```
```
[2] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
```




