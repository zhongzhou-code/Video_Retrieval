# Video_Retrieval
Pytorch implementation of a video retrieval model on datasets UCF101 and HMDB-51(Human Motion Data Base).

## Approach

### Network Structure

![image](https://user-images.githubusercontent.com/67732424/172331358-4644f229-2e55-4e94-a567-84116fb39dea.png)

1. Model input should be triplets of videos rather than images. That is to say, I get all possilble triplets of videos in a batch as model input.
2. The shared CNN sub-network is replaced with a pre-trained 3D ResNet-18 or  ResNet-34.
3. Divide-and-encode module is simplified as a fully connected layer. It projects the 512-dim extracted feature into a feature whose dimension can be 16, 32, 64 and so on. I call this feature with lower dimension as approximate hash code. During training phase, it is a real number and is imported into the triplet ranking loss function to calculate loss for model optimization. And during inference phase, it is quantized as binary code for retrieval.

## Experiment

1. Train the network with triplet loss on the training set for 100 epochs.
2. Input the training set and testing set into the network to get embeddings and then turn the embeddings into binary hash codes with a simple quantization function.
3. Use testing sample as querie to retrieve videos from training samples. Calculate distance between binary codes of testing sample and training samples with Hamming distance. Use mAP to estimate the model's performance.

### Prerequisites

In order to run this code you will need to install:

1. Python3
2. Pytorch 0.4

### Usage

1. Firstly download and unzip the two datasets.
2. Use video2frames.py to turn each video into a set of frames.
3. Generate two files train.txt and test.txt, each line of which is in the format of [video_name,classid].
4. Change the datapath arguments  in train.py to indicate the file path.
5. Run the command bellow. 
