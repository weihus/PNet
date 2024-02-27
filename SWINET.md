## skin

> 224 8 100

* NPNet： 0.7938 0.8560
* SWINET：0.8044 0.8756  0.8038 0.8787 0.8073 0.8802  0.7975 0.8735       0.8350 0.8978
* UNET：0.7883 0.8648 0.7726 0.8498

## luna

> 512 4 100

* NPNet:0.9668	0.9775
* SWINET:0.9682  0.9783
* UNET:0.9645 0.9778

![image-20211119161126022](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211119161126022.png)

## CVC

> 384 288 2 100

* NPNet: 0.8018  0.8645
* SWINET: 0.8219  0.8870     0.8113  0.8791
* UNET: 0.7726 0.8498

## resize结果

* dsb2018 96->388，iou大概50左右
* luna 512->64 iou大概50左右
* 上采样过程中加上之前的特征，会提升，dsb2018大概60左右，但luna效果会变差

## 4层 3x3   2,6

* CVC 0.8301 0.8938
* skin   0.7935 0.8656
* luna  0.9684  0.9793
* dsb2018 0.6937  0.6733

## 4层 1x1

* CVC  0.8231 0.8901
* skin  0.7935 0.8656
* luna  0.9684  0.9788
* dsb2018  0.6742 0.7608

## 3层 3x3

* CVC  0.7697  0.8453
* skin   0.8001  0.8731
* luna  0.9701  0.9802 
* dsb2018   0.6742  0.6231

## 3层 1x1

* CVC  0.7762 0.8492
* skin  
* luna  0.9694  0.9790
* dsb2018  0.6688  0.7474

## 3层 1x1 6,12

* CVC  0.7511 0.8348
* skin  0.8084 0.8794
* luna  0.9696  0.9802
* dsb2018  0.5959 0.4811

## 3层 1x1 2,4

* CVC  0.7611 0.8387
* skin  0.8021 0.8733
* luna  0.9690, 0.9797
* dsb2018  0.6817 0.6233

## 3层 1x1 2,3

* CVC  0.7435 0.8246
* skin  0.7985 0.8671
* luna  0.9690 0.9799
* dsb2018  0.6591 0.5516

## 魔改开始

![image-20211126154101155](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211126154101155.png)

* CVC 0.8331 0.8919
* luna 0.9668 0.9773
* skin 0.7976 0.8716
* dsb2018 0.6272 0.6799

![image-20211126154800940](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211126154800940.png)

* dsb2018  0.6274 0.6804
* luna 0.9670 0.9773
* skin 0.8045 0.8747 
* CVC 0.8264 0.8887

![image-20211126162201575](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211126162201575.png)

* dsb2018 0.3002 0.3752
* luna 0.9640 0.9754
* CVC 0.8136 0.8769
* skin 0.7993 0.8717

![image-20211126164551191](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211126164551191.png)

* dsb2018 0.3053 0.3859
* luna 0.9634 0.9757
* CVC 0.8109 0.8767
* skin 0.8047 0.8746

![image-20211126172011029](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211126172011029.png)

2  4,  14,  30

* dsb2018  0.3040 0.3780
* luna  0.9631  0.9754
* CVC  0.7713  0.8460
* skin  0.7960  0.8672

![image-20211126180622554](C:\Users\crash\AppData\Roaming\Typora\typora-user-images\image-20211126180622554.png)

* dsb2018  0.2969  0.3769
* luna  0.9613  0.9738
* CVC  0.8161  0.8793
* skin  0.8064  0.8764

## 写作积累

We introduce a new language representation model called **BERT**, which stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers.

## 参考资料

https://github.com/DebeshJha/ResUNetPlusPlus-with-CRF-and-TTA

[CVC Colo](https://sci-hub.se/10.1109/TMI.2015.2487997)

[CVC-Colon](http://www.cvc.uab.es/CVC-Colon/index.php/databases/)

[Focus U-Net](https://arxiv.org/pdf/2105.07467v2.pdf)

[ParNet Github](https://github.com/DengPingFan/PraNet)

[ParNet Paper](https://arxiv.org/pdf/2006.11392.pdf)

## success

* CVC(384x288) 息肉
  * UNet	0.8534    0.9066
  * SegNet    0.8381    0.8930
  * AttU_Net    0.8261    0.8802
  * NestedUNet    0.8544    0.9063
  * SwinNet    0.9006 0.9385
* skin_dataset(resize 224x224)
  * UNet	0.8211    0.8792
  * SegNet    0.8275    0.8852
  * AttU_Net    0.8259    0.8814
  * NestedUNet    0.8359    0.8894
  * SwinNet    0.8614 0.9089
* luna_dataset(512x512)
  * UNet    0.9688    0.9797
  * SegNet    0.8992    0.9378
  * AttU_Net    0.9406    0.9635
  * NestedUNet    0.9714    0.9818
  * SwinNet    0.9714    0.9810
* Kvasir-SEG(resize 352x352) 息肉
  * UNet	0.7911    0.8551
  * SegNet    0.7102    0.7861
  * AttU_Net    0.7733    0.8490
  * NestedUNet    0.7595    0.8340
  * SwinNet    0.8408 0.8893
* dsb2018(96x96)
  * UNet	**0.8488    0.9029**
  * SwinNet    **0.7468    0.8212**
* SwinNet(1 4)
  * C 0.8756 0.9227
  * K 0.8555 0.8991
  * S 0.8546 0.9053
  * L 0.9705 0.9798
  * E 0.8063 0.8511
* SwinNet(2 5)
  * C 0.8875 0.9311
  * K 0.8551 0.9046
  * S 0.8533 0.9049
  * L 0.9699 0.9794
  * E 0.8833 0.9062
* SwinNet(3 8)
  * C 0.8719 0.9178
  * K 0.8262 0.8831
  * S 0.8491 0.9018
  * L 0.9715 0.9807
  * E 0.8502 0.8796
* SwinNet(4 10)
  * C 0.8723 0.9192
  * K 0.8188 0.8775
  * S 0.8530 0.9059
  * L 0.9723 0.9812
  * E 0.8502 0.8796

***

* SwinNet(2 3)
  * C 0.8870 0.9287
  * K 0.8333 0.8838
  * S 0.8477 0.9002
  * L 0.9671 0.9778
  * E 0.8700 0.9005
* SwinNet(2 4)
  * C 0.8871 0.9291
  * K 0.8391 0.8944
  * S 0.8454 0.9002
  * L 0.9713 0.9805
  * E 0.6696 0.6795
* SwinNet(2 5)
  * C 0.8809 0.9271
  * K 0.8493 0.9001
  * S 0.8538 0.9072
  * L 0.9706 0.9803
  * E 0.8003 0.8414
* SwinNet(2 6)
  * C 0.8841 0.9274
  * K 0.8183 0.8740
  * S 0.8444 0.8988
  * L 0.9699 0.9797
  * E 0.8606 0.9007
* SwinNet(2 7)
  * C 0.8879 0.9309
  * K 0.8530 0.9005
  * WS 0.8482 0.9024
  * L 0.9700 0.9797
  * E 0.8361 0.8749

# new

3e-4

8：0.8755 0.9226

9：0.8899 0.9308

10：0.8782 0.9242

1e-4

8：0.8848 0.9281

9：0.8917 0.9325

10：0.8799 0.9250