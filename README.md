#Birdsong Recognition

This is a source code for the manuscript “Automatic recognition of element classes and boundaries in the birdsong with variable sequences” by Takuya Koumura and Kazuo Okanoya (not published yet).

This program performs automatic recognition of birdsong and parameter training for the recognizer using given ground truth. The source code includes three algorithms for automatic recognition: “BD -> LC -> GS”, “LC -> BD & GS”, and “Lc & GS -> BD & GS”, each of which is implemented in the classes [main.BdLcGs](birdsong-recognition/src/main/BdLcGs.java), [main.LcBdGs](birdsong-recognition/src/main/LcBdGs.java), and [main.LcGsBdGs](birdsong-recognition/src/main/LcGsBdGs.java). A viewer for an output sequence is also provided in another class, [main.Viewer](birdsong-recognition/src/main/Viewer.java). Please read the manuscript for the detailed description of these three algorithms.

This program has been tested on 64 bit windows 7 with a graphics processor GTX 980 or 970.

##Test data

Test data for running the program are available at http://marler.c.u-tokyo.ac.jp/files/koumura-okanoya-2016-songs/. It contains annotated songs and expected validation errors in eleven birds. The annotations of all songs are provided in Annotation.xml. Expected validation errors are the ones that would be obtained by running the program with the prefixed hyper parameters in the source code. They are provided for each of three algorithms and in ExpectedError/ErrorBdLcGs.xml, ExpectedError/ErrorLcBdGs.xml, and ExpectedError/ErrorLcGsBdGs.xml. Note that the results will be non-deterministic if backward algorithm is FAST_NON_DETERMINISTIC. Please read the manuscript for the definition of the validation errors.

Users who would like to perform the computation with their own data must prepare data with the same format as the provided test data. Sound data must be 16 bit linear PCM wave format with any sampling rate. Annotations of the songs must be given in XML format following the schema [xsd/AnnotationSchema.xsd](xsd/AnnotationSchema.xsd). Alternatively, users may modify the source code to load data with arbitrary format.

##Differences from the manuscript

+ Learning rate for parameter updating of the neural network are fixed in the manuscript, but are adaptively determined using Adam method in this source code.
+ All the hyper parameters are prefixed in the source code. In the manuscript, some of them are optimized by cross-validation within training data.
+ In the manuscript, in training with the LC & GS -> BD & GS arrangement, first the network with the same structure as the LC -> BD & GS arrangement is trained, and then additional fully-connected layer was inserted before training the whole network. In this source code, the whole network whose parameters were initialized randomly was trained from the beginning.

##Requirements
Users must prepare following libraries and processors.
+ Cuda 7.0 (or later) and a Cuda compatible graphics processor.
+ Cudnn ver. 3.
+ JCuda (http://www.jcuda.org/).

##Libraries
Other libraries used in this program are as follows (described in [pom.xml](birdsong-recognition/pom.xml)).
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

##License
This program is distributed under the term of GNU public license ver. 3. The copy of the license is in  [LICENSE](LICENSE).
