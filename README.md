# Birdsong Recognition

This is a source code for the manuscript “Automatic recognition of element classes and boundaries in the birdsong with variable sequences” by Takuya Koumura and Kazuo Okanoya (http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0159188).

This program performs automatic recognition of birdsong and parameter training for the recognizer using given ground truth. The source code includes three algorithms for automatic recognition: “BD -> LC -> GS”, “LC -> BD & GS”, and “Lc & GS -> BD & GS”, each of which is implemented in the classes [main.BdLcGs](birdsong-recognition/src/main/BdLcGs.java), [main.LcBdGs](birdsong-recognition/src/main/LcBdGs.java), and [main.LcGsBdGs](birdsong-recognition/src/main/LcGsBdGs.java). A viewer for an output sequence is also provided in another class, [main.Viewer](birdsong-recognition/src/main/Viewer.java). Please read the manuscript for the detailed description of these three algorithms.

This program has been tested on 64 bit windows 7 with a graphics processor GTX 980 or 970.

## Reference
Please cite the following paper if you use this program in your work.

Koumura T, Okanoya K (2016) Automatic Recognition of Element Classes and Boundaries in the Birdsong with Variable Sequences. PLoS ONE 11(7): e0159188. doi:10.1371/journal.pone.0159188

## Test data

Test data for running the program are available at https://figshare.com/articles/BirdsongRecognition/3470165. It contains annotated songs and expected validation errors in eleven birds. The annotations of all songs are provided in Annotation.xml. Expected validation errors are the ones that would be obtained by running the program with the prefixed hyper parameters in the source code. They are provided for each of three algorithms and in ExpectedError/ErrorBdLcGs.xml, ExpectedError/ErrorLcBdGs.xml, and ExpectedError/ErrorLcGsBdGs.xml. Note that the results will be non-deterministic if backward algorithm is FAST_NON_DETERMINISTIC. Please read the manuscript for the definition of the validation errors.

Users who would like to perform the computation with their own data must prepare data with the same format as the provided test data. Sound data must be 16 bit linear PCM wave format with any sampling rate. Annotations of the songs must be given in XML format following the schema [xsd/AnnotationSchema.xsd](xsd/AnnotationSchema.xsd). Alternatively, users may modify the source code to load data with arbitrary format.

## Files
+ [birdsong-recognition](birdsong-recognition): Main java codes for song recognition.
+ [cuda-kernels](cuda-kernels): Cuda kernels used in Java codes.
+ [error_rate](error_rate): Error rates reported in the paper.
+ [fold](fold): Fold information of three-fold cross-validation.
+ [xsd](xsd): XML schema of Annotation.xml shared in figshare (see above).

## Differences from the manuscript

+ Learning rate for parameter updating of the neural network are fixed in the manuscript, but are adaptively determined using Adam method in this source code.
+ All the hyper parameters are prefixed in the source code. In the manuscript, some of them are optimized by cross-validation within training data.
+ In the manuscript, in training with the LC & GS -> BD & GS arrangement, first the network with the same structure as the LC -> BD & GS arrangement is trained, and then additional fully-connected layer was inserted before training the whole network. In this source code, the whole network whose parameters were initialized randomly was trained from the beginning.

## Prerequisites
Users must prepare following libraries and processors.
+ Cuda 7.0 (or later) and a Cuda compatible graphics processor.
+ Cudnn ver. 3.
+ JCuda (http://www.jcuda.org/).
+ JDK 8.

Also, users must compile [cuda-kernels/kernel.cu](cuda-kernels/kernel.cu) into PTX file. The path to the PTX file must be set in the main methods.

## Libraries
Other libraries used in this program are as follows (listed in [pom.xml](birdsong-recognition/pom.xml)).
+ Java native access.
https://github.com/java-native-access/jna
+ JNAerator.
https://github.com/nativelibs4java/JNAerator
+ Commons Math.
https://commons.apache.org/proper/commons-math/
+ Xerces.
https://xerces.apache.org/
+ Matrix toolkits java.
https://github.com/fommil/matrix-toolkits-java

Although this program does not include it as a library, https://github.com/tbennun/cudnn-training has been very helpful for writing the program.

## License
This program is distributed under the term of GNU public license ver. 3. The copy of the license is in  [LICENSE](LICENSE).

## Grants
This project is supported by...
+ Grant-in-Aid for Scientific Research (A) (#26240019), MEXT/JSPS, Japan, to KO.
+ Grant-in-Aid for JSPS Fellows, MEXT/JSPS, Japan (#15J09948) to TK.

---
Copyright &copy; 2016 Takuya KOUMURA
