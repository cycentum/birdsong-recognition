/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package main;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.ExecutionException;
import java.util.function.IntBinaryOperator;
import java.util.stream.Collectors;

import javax.sound.sampled.UnsupportedAudioFileException;
import javax.xml.parsers.ParserConfigurationException;

import org.apache.commons.math3.random.MersenneTwister;
import org.xml.sax.SAXException;

import computation.DnnComputation;
import computation.DnnComputation.Config;
import computation.DnnComputation.HyperParam;
import computation.HmmComputation;
import computation.STFTParam;
import computation.Sequence;
import computation.Sequence.LabelList;
import computation.Sequence.Note;
import computation.Thresholding;
import cudnn.CudaException;
import cudnn.CudnnException;
import cudnn.layer.ConvLayer;
import errorcomputation.Levenshtein;
import errorcomputation.Matching;
import no.uib.cipr.matrix.NotConvergedException;
import utils.DnnUtils;
import utils.Executor;
import utils.SoundUtils;


public class BdLcGs 
{
	public static void main(String... arg)
	{
		int numThread=4;
		Executor executor=new Executor(numThread);
		try
		{
			main(executor);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		finally
		{
			executor.get().shutdown();
		}
	}
	
	public static void main(Executor executor) throws IOException, UnsupportedAudioFileException, NotConvergedException, CudaException, CudnnException, SAXException, ParserConfigurationException, InterruptedException, ExecutionException
	{
		/**********************************************************
		 * Files.
		 * Change them according to your environment.
		 *********************************************************/
		//Cuda.
		Path fileCudnnLibrary=Paths.get("I:\\koumura\\EclipseWorkspaceSe\\Dependencies\\cudnn\\cuda\\bin\\cudnn64_70.dll");
		Path fileCudaKernel=Paths.get("L:\\koumura\\Documents\\Visual Studio 2013\\Projects\\CudaKernel\\CudaKernel\\x64\\Release").resolve("kernel.cu.ptx");
		
		//Data.
		Path dirWave=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation\\Wave\\B-W-20150112");
		Path fileAllSequences=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\AllSequences.xml");
		Path fileTrainingSequences=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\TrainingSequences.xml");
		Path fileValidationSequences=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\ValidationSequences.xml");
		
		//Outputs.
		Path fileThresholdingParameter=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\ThresholdBdLcGs");
		Path fileDnnParameter=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\WeightBdLcGs");
		Path fileDnnOutput=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\OutputBdLcGs");
		Path fileOutputSequence=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\OutputSequenceBdLcGs.xml");
		
		
		/**********************************************************
		 * Hyper parameters.
		 * These hyper parameters may be determined using cross-validation within training data.
		 *********************************************************/
		//Spectrogram.
		STFTParam stftParam=new STFTParam(512, 32);
		double frequencyStartHz=1000, frequencyEndHz=8000;
		int dpssParam=4;
		
		//DNN.
		double inputDataLengthUpperSec=8;
		int localInputHeight=96;
		int numIter=200;
		int batchSizeUpper=2;
		int randomSeed=0;
		
		//HMM.
		double smoothingConstant=1e-4;
		boolean posteriorToObservationProb=false;
		
		
		/**********************************************************
		 * Other configurations.
		 * Note that results will not be deterministic if ConvLayer.BackwardAlgorithm.FAST_NON_DETERMINISTIC is used.
		 *********************************************************/
		boolean verbose=true;
		ConvLayer.BackwardAlgorithm backwardAlgorithm=ConvLayer.BackwardAlgorithm.FAST_NON_DETERMINISTIC;



		
		/**********************************************************
		 * Reading sequences.
		 *********************************************************/
		ArrayList<Sequence> allSequence=Sequence.readXml(fileAllSequences);
		LabelList labelList=Sequence.LabelList.create(allSequence);
		ArrayList<Sequence> trainingSequence=Sequence.readXml(fileTrainingSequences);
		ArrayList<Sequence> validationSequence=Sequence.readXml(fileValidationSequences);
		
		
		/**********************************************************
		 * Boundary detection.
		 *********************************************************/
		int samplingRate=(int)SoundUtils.checkSamplingRate(allSequence.stream().map(s->s.getWaveFileName()).collect(Collectors.toList()), dirWave);
		int freqOffset=(int)(frequencyStartHz/stftParam.unitFrequency(samplingRate));
		int freqLength=(int)(frequencyEndHz/stftParam.unitFrequency(samplingRate))-freqOffset;
		int gapLengthLowerUpper=localInputHeight, noteLengthLowerUpper=localInputHeight;
		Thresholding.HyperParameter thresholdingHyperParameter=new Thresholding.HyperParameter(stftParam, dpssParam, freqOffset, freqLength, gapLengthLowerUpper, noteLengthLowerUpper);
		Thresholding.Config thresholdingConfig=new Thresholding.Config(executor, dirWave, verbose);

		//Training.
//		Thresholding.Parameter thresholdingParameter=Thresholding.train(trainingSequence, thresholdingHyperParameter, thresholdingConfig);

		//Parameter saving.
//		thresholdingParameter.writeXml(fileThresholdingParameter);
		
		//Parameter loading.
		//Uncomment to use pre-trained parameters.
		Thresholding.Parameter thresholdingParameter=Thresholding.Parameter.parseXml(fileThresholdingParameter);

		//Boundary detection in the validation data.
		HashMap<Sequence, ArrayList<int[]>> soundInterval=Thresholding.boundaryDetection(validationSequence, thresholdingParameter, thresholdingHyperParameter, thresholdingConfig);

		
		/**********************************************************
		 * Local classification.
		 *********************************************************/
		int inputHeightUpper=stftParam.spectrogramLength((int)(inputDataLengthUpperSec*samplingRate));
		IntBinaryOperator silentLabelFunc=(numUpperLabel, numLowreLabel)->-1;
		IntBinaryOperator softmaxSizeFunc=(numUpperLabel, numLowreLabel)->numUpperLabel*numLowreLabel;
		MersenneTwister random=new MersenneTwister(randomSeed);
		HyperParam dnnHyperParam=new HyperParam(stftParam, dpssParam, localInputHeight, localInputHeight, 1, freqOffset, freqLength, inputHeightUpper, batchSizeUpper, numIter);
		
		//Computing mean & sd of training spectrogram for input normalization.
		HashMap<Sequence, float[]> spectrogram=SoundUtils.spectrogram(Sequence.wavePositionMap(trainingSequence, dirWave), stftParam, dpssParam, freqOffset, freqLength);
		double[] specMeanSd=SoundUtils.spectrogramMeanSd(trainingSequence.stream().map(s->spectrogram.get(s)).collect(Collectors.toList()));
		Config dnnConfig=new Config(verbose, silentLabelFunc, softmaxSizeFunc, specMeanSd, fileCudaKernel, fileCudnnLibrary, dirWave, backwardAlgorithm);
		
		//Training.
		DnnComputation.Param dnnParam=DnnComputation.trainLocalRecognition(trainingSequence, labelList, random, dnnHyperParam, dnnConfig);
		
		//Parameter saving.
		DnnUtils.saveParam(dnnParam.getLayerParam(), fileDnnParameter);
		
		//Parameter loading.
		//Uncomment to use pre-computed parameters.
//		DnnComputation.Param dnnParam=new DnnComputation.Param(DnnUtils.loadParam(fileDnnParameter));
		
		//Local recognition in the validation data.
		HashMap<Sequence, float[]> dnnOutput=DnnComputation.localRecognition(validationSequence, labelList, dnnParam, dnnHyperParam, dnnConfig);
		
		//Saving DNN output.
		DnnUtils.saveOutput(dnnOutput, fileDnnOutput);
		
		
		/**********************************************************
		 * Global sequencing.
		 *********************************************************/
		HmmComputation.HyperParam hmmHyperParam=new HmmComputation.HyperParam(smoothingConstant, posteriorToObservationProb, 1, stftParam);
		HashMap<Sequence, ArrayList<double[]>> observationProb=Thresholding.averageOutput(soundInterval, dnnOutput, stftParam);
		observationProb=HmmComputation.segmentedPosteriorToObservationProb(observationProb, trainingSequence, labelList, hmmHyperParam);
		HmmComputation.Config hmmConfig=new HmmComputation.Config(executor, 0);
		
		//Training
		double[][][] transitionProb=HmmComputation.transitionProbability(trainingSequence, labelList, hmmHyperParam);
		
		//Global sequencing in the validation data.
		HashMap<Sequence, ArrayList<Note>> outputSequence=HmmComputation.globalSequencing(observationProb, transitionProb, labelList, soundInterval, hmmHyperParam, hmmConfig);
		
		//Saving output sequences.
		Sequence.writeOutputSequence(outputSequence, fileOutputSequence);
		
		
		/**********************************************************
		 * Error computation.
		 *********************************************************/
		double levenshteinError=Levenshtein.computeError(outputSequence);
		double matchingError=Matching.computeError(outputSequence, stftParam, labelList);
		System.out.printf("Levenshtein error =\t%.2f%%", (levenshteinError*100));
		System.out.println();
		System.out.printf("Matching error =\t%.2f%%", (matchingError*100));
		System.out.println();
	}
}
