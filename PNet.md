# PNet：Patch Network for medical image Segmentation

## abstract

精准、快速对医学图像进行分割具有重要临床意义，然而当下研究方法中卷积神经网络拥有较快的推理速度，却难以对图像上下文特征信息学习，而transformer拥有较好的性能表现却对硬件环境要求较高。在本文中，我们提出了一个Patch Network(PNet) 将transformer思想融入卷积神经网络，可以获取图像更加丰富的上下文信息，在速度和精度上达到一个平衡。we test our PNet on Polyp(CVC-ClinicDB and ETIS-LaribPolypDB), Skin(ISIC-2018 skin lesion segmentation challenge dataset)  segmentation datasets. Our PNet达到了sota表现在 in both 速度和精度。

## introduction

The incidence of colorectal cancer (CRC)已经称为仅次于肺癌的第三常见癌症在世界范围内，同时大部分CRC的都是由于息肉转变而来的，而如果在早期阶段可以及时发现并切除，则可以有效阻止息肉转变为CRC，因此，如何精准预测变成了一个核心任务，当下流行的检测方法是通过结肠镜检测息肉并切除，一般情况下都是优秀的医生手动实现，识别息肉难度较大且消耗大量时间，此时自动精准分割方法就拥有了重要的意义。另外一种普遍存在而不被重视的疾病为skin lesion，其通常情况上对身体影响不大，但其类型多种多样，一些skin lesion如果不及时治疗，可能会变成永久性损失，也可能会导致其他疾病，甚至遗传后代，治疗的重心则为对skin lesion所处位置的准确分割，从而进行后续的工作。综上所述，CRC和skin lesion均需要对病变位置精准分割，医学作为一个对精度和速度要求都非常严格的领域，当下的大部分研究方法这些方法还是有效证明了深度学习对医学图像分割的可行性，但均难以达到要求，表现良好的方法会对硬件要求比较高with消耗大量的时间，检测速度快的方法会对目标物体的细节特征进行一些忽略，这些问题在医学领域均需要有效应对。针对于此，we propose Patch block将拥有良好表现性能的Transformer思想融入到快速的CNN中，以打补丁的形式获取更多的上下文信息，有效将二者的优势融合到一起，同时，我们提出了一个novel的轻量级网络patch network for medical image segmentation，我们测试我们的模型在三个数据集上，在CVC上iou和dice达到了0.9332、0.9599，在ETSI上iou和dice达到了0.9405、0.9646，在Skin上iou和dice达到了0.8946、0.9340，均明显优于其他模型，尤其在对图像边缘的分割效果更为惊艳，模型参数量和MACs仅仅为unet++的1/10，而fps达到了unet++的3倍多。

## Related work

当前主流的语义分割模型主要分为两个阵营，CNN和Transformer方法。

​	**CNN**：CNN主要利用卷积层对图像特征信息进行提取，通过在图像滑动获得整个图像的特征信息，从而对图像进行分类，语义分割作为一种特殊的图像分类形式，其可以对图像进行像素级别的分类，这也不难理解为什么很多网络模型使用图像分类的模型作为backbone，使用成熟的图像分类算法对图像提取到的特征进一步操作达到像素级别的图像分类，可以达到较好的表现性能，同时也简化了网络设计难度，为了满足语义分割对像素级别特征的需求，pspnet在backbone后面使用金字塔池化模块（Spatial Pyramid Pooling, SPP）对不同大小的物体进行适应，同时可以获取多尺度特征信息，deeplab则在backbone后面使用aspp模块获取更加丰富的特征信息through更大的感受野，这类模型均默认其使用的图像分类backbone已经学习到了准确的特征信息，但目前没有充足的证据能够说明图像分类和语义分割的特征信息是否通用，这也引发了后续一系列针对语义分割而设计的网络模型。

这些网络基本都通过不同的结构形式将高层语义特征与低层 

UNet通过跳跃连接融合了高层和低层的语义特征信息，达到了不错的分割效果，但其每层的特征信息提取能力限制了其性能发挥，UNet++、DenseUnet等网络通过密集的连接结构获取到更加丰富的特征，但这种密集结构也带来了冗余信息和繁重的网络模型，为了对每层特征信息更好的提取，ENet融合卷积操作和池化操作来获取下采样过程中的特征信息，CGNet使用CGblock结合普通卷积操作和空洞卷积来获取了上下文的特征信息，并通过添加注意力结构增强特征的提取能力 ，BiSeNet使用双分支的结构融合空间和上下文的特征信息，CNN形式的语义分割结构更多是通过对每层的特征信息进行更好的提取，使用backbone的分割网络结构更多是在backbone后使用多分支结构获取更深层的特征信息，从而更好分割。

​	**Transformer**：Transformer起源于natural language process（NLP），因其在nlp出色的表现被计算机视觉的研究者所关注，很快一系列的相关工作将Transformer应用于语义分割，这些工作得到了出色的表现。Transformer这种特殊的结构形式可以对图像中每一个像素值进行覆盖，获取全局的特征信息，这也导致了庞大的计算量，swin Transformer则对该问题提出了解决方案，将图像切块，对特征提取依据先局部后整体的流程，很好的利用了CNN局部性的思想，但网络模型相对于CNN来说依旧差距较大，且硬件环境要求高。

​	综上所述，我们提出了patch block，将swin Transformer的切块思想以另一种形式融入了CNN结构，我们也借鉴了cnn对上下文信息提取的模块，使得我们的模块在网络每层均可以提取更加全面的特征信息。

## Method

In this work，我们提出了Patch Network（PNet），Patch Network 采用了经典的encoder-decoder结构，在本节，我们按照encoder和decoder两大部分进行展开叙述，对PNet的组成部分进行讲述，并对patch block的类似单元进行了比较。

### Encoder

在该阶段主要有下采样模块和patch block两部分组成，相对于最大池化或平均池化直接通过简单的提取方式对特征进行保留，我们更倾向于使用卷积操作通过学习来实现降采样操作，多次实验证明，卷积实现的降采样操作优于池化操作，同时在【Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs】的启发下，我们没有使用常规的3x3的卷积核，而大胆尝试使用了5x5卷积核得到了更好的表现，因此下采样模块由5x5的卷积核、stride为2，padding为2的设置实现。对于CNN而言，每层对图像特征的提取能力很大程度上影响了模型的最终表现，我们对使用图像分类网络作为backbone的经典网络deeplab进行了分析，在默认图像分类网络已经学习很好特征的情况下，其使用aspp（如图2a所示）通过更大的感受野对上下文特征进一步补足从而得到了很好的效果，像CGNet没有不使用backbone则需要设计模块去提取特征，其使用cgblock（如图2b所示）通过增强每层的特征提取能力来提高模型的表现，我们设计的初衷设计一个使用类似aspp表现的模块去替换cgblock作为每层的特征提取器，而swin Transformer的出现成为了我们模块设计的火花，其对图像分块去进行学习的思想，由于cnn和Transformer的实现和特点并不相同，照搬结构可解释性太差，最终，我们设计了一个模块，先使用一个小空洞率的空洞卷积对特征进行学习，再使用一个大空洞率的空洞卷积去进一步学习，类似于“打补丁”的形式，一个千疮百孔的衣服打过补丁依然可以穿很久，这正因此我们称其为patch block，而两次的空洞率我们分别设置为2、6，为了直观的理解，我们展示了标准的3x3卷积（图3a），空洞卷积为2、padding也为2的3x3卷积（图3b），空洞卷积为6、padding也为6的3x3卷积（图3c），我们暂且将标准3x3卷积区域称为标准区域，第一次空洞卷积中可以看出使用标准区域的部分区域并对周围区域进行联合学习，而滑窗过程会将上部三个区域和下部三个区域进行覆盖学习，接着使用第二次更大空洞率的空洞卷积对中间三个区域的两个区域进行覆盖学习，同时联合更大范围的区域进行学习，最终将原始输入和此处结果通过add进行残差学习，进一步提高特征提取能力并弥补遗漏区域的特征信息。最后，我们组合下采样模块和patch block使用，前者对图像进行降采样，后者对该层的特征信息学习，保证每层特征信息更加丰富且覆盖更大范围的上下文信息，连续四次，至此Encoder阶段结束。

### Decoder

考虑到轻量级模型的设计，加上Encoder阶段强大的特征提取能力，我们在Decoder进行了简化的设计。首先将上一阶段学习到的特征上采样8倍和第一次下采样的特征进行concat，融合深层语义信息和浅层的空间信息，由于patch block在两次空洞卷积优秀的特征提取能力也带来一些冗余信息，因此我们在这里先使用3x3卷积对深层和浅层融合信息特征进行学习，然后使用0.3的dropout对冗余信息进行抑制，并一定程度防止模型过拟合，接着使用两个1x1卷积，前者进行跨通道的优化，后者输出分类类别数量，最终再次使用一个上采样操作回归到原始图像输入大小。

### Patch Network

在Encoder阶段进行四次下采样操作，每次下采样后使用patch block获取更大范围的上下文信息，优秀的特征提取能力可以对获得更加丰富的语义信息，在decoder阶段融合深层语义信息和浅层空间信息，并通过dropout去防止过拟合。一个精心设计的轻量级CNN网络for医学图像分割就这样诞生了，As shown in Fig.1，

## Experiments and Results

### Dataset

**Polyp Segmentation**. Accurate detection of colon polyps is of great significance for the prevention of colon cancer. CVC-ClinicDB[2](CVC for short) includes 612 colon polyp images. We use the original size 384x288 of image and split it into
train set(80%) and test set(20%).

**ETIS.**和CVC一样，ETIS也是息肉数据集，其包含196图像来自于29个序列，这些序列均来自于不同的设备，we resize all the origin iamges to 512x384 and split it into trainset(80%) and testset(20%).

**Skin Lesion Segmentation**. Computer-aided automatic diagnosis of skin cancer is an inevitable trend, and skin lesions segmentation as the first step is urgent.
The data set is from MICCAI 2018 Workshop - ISIC2018: Skin Lesion Analysis Towards Melanoma Detection[8][14](skin for short). It contains 2594 images and is randomly split into train set (80%) and test set (20%). For better model training
and result display, we resize all the original images to 224 × 224. 

### Implementation details

For three benchmarks and multiple segmentation models, we set consistent training parameters. we set epochs as 200 in the three data sets. We use a learning rate(LR) equal to 1e-4 for all task. In addition, we use batch size equal to 2 for ETIS and CVC task, and 4 for the skin task. Cross entropy loss and Adam are used as loss function and optimizer, respectively. All experiments run on the NVIDIA TITAN V GPU with 12GB. Intersection over Union (IOU), dice coefficient, FPS and computational complexity(MACs) are selected as the evaluation metrics in this paper. We used these evaluation metrics for all datasets. 同时，我们使用RandomRotate90、Flip、HueSaturationValue、RandomBrightness、RandomContrast对这些训练数据进行数据增强，增大数据集用以防止过拟合，并提高模型的鲁棒性。

### Experimental Results

为了进一步展示我们模型的优越性，我们在三个数据集上进行评估，并使用IOU、DICE、MACs和FPS四个评价指标展示多个模型的表现对比，quantitative results展示在表1。同时，我们也展示了results visualization在图4。

#### results of 息肉

从表1中可以看到，我们的模型在两个息肉数据集CVC和ETIS上均明显优越其他模型，尤其是在ETIS这个仅有196张图像的小型数据集上，我们的模型在iou和dice分别达到了0.9405和0.9646，极大幅度超越了其他模型，说明我们的模型在小型数据集依然拥有良好的表现。从模型参数量、MACs上来看，我们的模型也更加轻量化与其他模型对比，仅仅是UNet++的1/10，而FPS也快过Unet++3倍。这一对比同样出现在可视化结果上，从图4可以看到我们模型的分割效果在整体和边缘的表现都优于其他模型，这一点在ETIS数据集上表现的更加明显，其他模型对于边缘的分割比较粗糙，而我们的模型和真实mask最为接近，且边缘处理prefect。

#### results of skin lesion Segmentation

在SKIN上，我们也可以从表1中看到，我们的模型表现优于其他所有模型在iou和dice两个评价指标上，此外我们模型参数量和MACs大大小于其他所有模型，这也使得我们的模型拥有了更快的FPS。虽然一些模型的评价指标的表现接近于我们的模型，其他模型中表现最为优异的模型是PSPNet，但从图4的可视化结果来看，我们的模型优势更加明显，其他模型的对边缘的分割过于平滑，和真实mask有较大的差距，以DenseUnet为例，其分割指标最为接近我们的效果，但从其可视化效果来看，对图像边缘的分割过于平滑，这与真实mask存在差距，而这种对于复杂边缘的分割极大程度上考验了模型的分割能力，而我们模型中patch block可以捕获更大上下文信息的特征提取能力则很大提升了分割表现，在边缘的分割表现依旧可以接近真实mask。

#### ablation study

为了进一步证实我们提出模块的有效性，我们对提出的模块进行了消融实验。首先，对于我们模型的核心模块Patch Block中的两次空洞卷积的空洞率进行了验证，从表2的实验结果可以得到我们的提出两次空洞卷积的空洞率分别为2和6达到了最好的效果，这和图3的图像解释相符，当一次卷积的空洞率为2时，我们对比了第二次卷积的空洞率分别为5、6、7时的表现效果，按照我们的设计理念，第二次空洞率需要完整大于第一次的卷积范围，即第一次空洞率为2的时，此时卷积范围为5x5，而空洞率为6时，也就是每两个卷积核之间的距离为5，此时正好涵盖第一次的卷积范围，同理可得，第一次卷积率为3时，第二次卷积率应该为8，我们也将其作为对比进行了实验，实验效果并不好，猜测的原因是卷积范围太大，对于一些小分辨率的图像，经过4次下采样图像大小将会很小，而卷积核为3空洞率为8时的卷积范围为17x17，此时可能会比图像本身还要大，导致需要填充太多的边界进行计算，进而使得效果不佳。此外，我们也对Encoder阶段中的下采样模块进行了对比实验，分别使用3x3的卷积核，stride为2和使用3x3卷积结合最大池化进行比较，从表3的实验结果可以看出来，我们的下采样模块达到了最优，这些消融实验证明了我们模块的效果。

参考![image-20220430110410444](C:\Users\yaoye\Desktop\workspace\model\pnet\PNet.assets\image-20220430110410444.png)

https://arxiv.org/pdf/2102.08005v2.pdf

## Conclusion

在本文中，我们提出了高效的特征提取模块patch block，并在其基础上提出了patch network针对于医学图像分割。我们在三个banchmarks上的实验iou和dice明显优于其他模型，且对于边缘的处理效果展现了patch block的优越的特征提取能力，ETIS的实验结果也展示了我们模型在小数据集上的优越表现。模型大小和MACs仅仅为UNet++的1/10，推理速度仍优于其3倍。

25

26

27

38

