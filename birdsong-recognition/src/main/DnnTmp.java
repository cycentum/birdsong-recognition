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
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.function.IntBinaryOperator;
import java.util.stream.Collectors;

import javax.sound.sampled.UnsupportedAudioFileException;

import org.apache.commons.math3.random.MersenneTwister;

import computation.DnnComputation;
import computation.DnnComputation.Config;
import computation.DnnComputation.HyperParam;
import computation.HmmComputation;
import computation.STFTParam;
import computation.Sequence;
import computation.Sequence.LabelList;
import computation.Sequence.WavePosition;
import computation.Thresholding;
import cudnn.ActivationMode;
import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.FloatType;
import cudnn.IntType;
import cudnn.layer.ConvLayer;
import cudnn.layer.ConvLayer.BackwardAlgorithm;
import cudnn.layer.DataLayer;
import cudnn.layer.Layer;
import cudnn.network.SeqNetwork;
import no.uib.cipr.matrix.NotConvergedException;
import utils.CollectionUtils;
import utils.DnnUtils;
import utils.Executor;
import utils.SoundUtils;
 
public class DnnTmp
{
	public static ArrayList<float[]> spec(List<Sequence> sequence) throws UnsupportedAudioFileException, IOException, NotConvergedException
	{
		Path waveFileDir=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation\\Wave\\B-W-20150112");
		int SAMPLING_RATE=32000;
		STFTParam stftParam=new STFTParam(512, 32);
		int dpssParam=4;
		int freqOffset=(int)(1000/stftParam.unitFrequency(SAMPLING_RATE));
		int freqLength=(int)(8000/stftParam.unitFrequency(SAMPLING_RATE))-freqOffset;
		HashMap<Sequence, WavePosition> wavePosition=Sequence.wavePositionMap(sequence, waveFileDir);
		int LOCAL_INPUT_HEIGHT=96;
		int finalInputHeight=LOCAL_INPUT_HEIGHT;
		int marginBegin=finalInputHeight/2;
		int marginEnd=finalInputHeight/2-1;
		for(WavePosition wp: wavePosition.values())
		{
			int waveBegin=wp.getPosition()-marginBegin*stftParam.getShiftLength();
			int waveEnd=wp.getEnd()+marginEnd*stftParam.getShiftLength();
			wp.setPosition(waveBegin);
			wp.setLength(waveEnd-waveBegin);
		}
		HashMap<Sequence, float[]> spec=SoundUtils.spectrogram(wavePosition, stftParam, dpssParam, freqOffset, freqLength);
		return sequence.stream().map(s->spec.get(s)).collect(CollectionUtils.arrayListCollector());
	}

	/*public static void main(String... arg) throws IOException
	{
		int numThread=4;
		Executor executor=new Executor(numThread);
		try
		{
			BdLcGs.main(executor);
			LcBdGs.main(executor);
			LcGsBdGs.main(executor);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		finally
		{
			executor.get().shutdown();
		}
		
		for(Path file0: Files.list(Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Result0")).collect(Collectors.toList()))
		{
			Path file1=file0.getParent().getParent().resolve(file0.getFileName().toString());
			System.out.println(Arrays.equals(Files.readAllBytes(file0), Files.readAllBytes(file1))+" "+file0.getFileName());
		}
	}*/
	
	
	static int birdIndex=0;
	
	public static void main(String... arg)
	{
		for(birdIndex=0; birdIndex<11; ++birdIndex)
		{
			BdLcGs.main();
			LcBdGs.main();
			LcGsBdGs.main();
		}
	}
	
	
	public static ArrayList<ArrayList<double[]>> observationProbBdLcGs(boolean conv) throws UnsupportedAudioFileException, IOException, NotConvergedException, CudaException, CudnnException
	{
		int numThread=4;
		Executor executor=new Executor(numThread);
		
		Path FILE_CUDNN_LIBRARY=Paths.get("I:\\koumura\\EclipseWorkspaceSe\\Dependencies\\cudnn\\cuda\\bin\\cudnn64_70.dll");
		Path FILE_CUDA_KERNEL=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation").resolve("kernel.cu.ptx");
		Path DIR_WAVE=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation\\Wave\\B-W-20150112");
		Path FILE_ALL_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\AllData.xml");
		Path FILE_TRAINING_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\TrainingData.xml");
		Path FILE_VALIDATION_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\ValidationData.xml");
		Path FILE_THRESHOLDING_PARAMETER=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\ThresholdBdLcGs");
		Path FILE_DNN_PARAMETER=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\WeightBdLcGs");
		
		ArrayList<Sequence> allSequence=Sequence.readXml(FILE_ALL_DATA);
		LabelList labelList=Sequence.LabelList.create(allSequence);
		ArrayList<Sequence> trainingSequence=Sequence.readXml(FILE_TRAINING_DATA);
		
		int SAMPLING_RATE=(int)SoundUtils.checkSamplingRate(allSequence.stream().map(s->s.getWaveFileName()).collect(Collectors.toList()), DIR_WAVE);
		STFTParam stftParam=new STFTParam(512, 32);
		int freqOffset=(int)(1000/stftParam.unitFrequency(SAMPLING_RATE));
		int freqLength=(int)(8000/stftParam.unitFrequency(SAMPLING_RATE))-freqOffset;
		int heightUpper=stftParam.spectrogramLength(8*32000);
		int dpssParam=4;
		
		int gapLowerUpper=96, noteLowerUpper=96;
		Thresholding.HyperParameter thresholdingHyperParameter=new Thresholding.HyperParameter(stftParam, dpssParam, freqOffset, freqLength, gapLowerUpper, noteLowerUpper);
		Thresholding.Config thresholdingConfig=new Thresholding.Config(executor, DIR_WAVE, true);
		ArrayList<Sequence> thresholdingSequence=new ArrayList<>();
		thresholdingSequence.add(trainingSequence.get(0));
		thresholdingSequence.add(trainingSequence.get(1));
//			Thresholding.Parameter thresholdingParameter=Thresholding.bestParameter(thresholdingSequence, thresholdingHyperParameter, thresholdingConfig);
//			thresholdingParameter.writeXml(FILE_THRESHOLDING_PARAMETER);
		Thresholding.Parameter thresholdingParameter=Thresholding.Parameter.parseXml(FILE_THRESHOLDING_PARAMETER);
		
		HashMap<Sequence, float[]> spec=SoundUtils.spectrogram(Sequence.wavePositionMap(trainingSequence, DIR_WAVE), stftParam, dpssParam, freqOffset, freqLength);
		double[] specStat=SoundUtils.spectrogramMeanSd(trainingSequence.stream().map(s->spec.get(s)).collect(Collectors.toList()));
		int LOCAL_INPUT_HEIGHT=96;
		int FINAL_INPUT_HEIGHT=LOCAL_INPUT_HEIGHT;
		int NUM_LOWER_LABEL=1;
		IntBinaryOperator silentLabelFunc=(numUpperLabel, numLowreLabel)->-1;
		IntBinaryOperator softmaxSizeFunc=(numUpperLabel, numLowreLabel)->numUpperLabel*numLowreLabel;
		int numIter=80;
		HyperParam dnnHyperParam=new HyperParam(stftParam, dpssParam, LOCAL_INPUT_HEIGHT, FINAL_INPUT_HEIGHT, NUM_LOWER_LABEL, freqOffset, freqLength, heightUpper, 4, numIter);
		Config dnnConfig=new Config(true, silentLabelFunc, softmaxSizeFunc, specStat, FILE_CUDA_KERNEL, FILE_CUDNN_LIBRARY, DIR_WAVE, ConvLayer.BackwardAlgorithm.FAST_NON_DETERMINISTIC);
		
		ArrayList<Sequence> validationSequence=Sequence.readXml(FILE_VALIDATION_DATA);
		validationSequence=validationSequence.subList(0, 3).stream().collect(CollectionUtils.arrayListCollector());
		HashMap<Sequence, float[]> validationDnnOutput=new HashMap<>();
		
		{
			ByteBuffer buf=ByteBuffer.wrap(Files.readAllBytes(FILE_DNN_PARAMETER.resolveSibling("OutputBdLcGs")));
			int numSequence=buf.getInt();
			for(Sequence seq: validationSequence)
			{
				int len=buf.getInt();
				float[] value=new float[len];
				validationDnnOutput.put(seq, value);
				for(int i=0; i<len; ++i) value[i]=buf.getFloat();
			}
		}
		
		HashMap<Sequence, ArrayList<int[]>> soundInterval=Thresholding.boundaryDetection(validationSequence, thresholdingParameter, thresholdingHyperParameter, thresholdingConfig);
		
		double smoothingConstant=1e-4;
		boolean posteriorToObservationProb=conv;
		HmmComputation.HyperParam hmmHyperParam=new HmmComputation.HyperParam(smoothingConstant, posteriorToObservationProb, NUM_LOWER_LABEL, stftParam);
		
		HashMap<Sequence, ArrayList<double[]>> posteriorProb=Thresholding.averageOutput(soundInterval, validationDnnOutput, stftParam);
		HashMap<Sequence, ArrayList<double[]>> observationProb=HmmComputation.segmentedPosteriorToObservationProb(posteriorProb, trainingSequence, labelList, hmmHyperParam);
		return validationSequence.stream().map(s->observationProb.get(s)).collect(CollectionUtils.arrayListCollector());
	}
	
	public static ArrayList<ArrayList<double[]>> observationProbLcBdGs(boolean conv) throws UnsupportedAudioFileException, IOException, NotConvergedException, CudaException, CudnnException
	{
		int numThread=4;
		Executor executor=new Executor(numThread);
		
		Path FILE_CUDNN_LIBRARY=Paths.get("I:\\koumura\\EclipseWorkspaceSe\\Dependencies\\cudnn\\cuda\\bin\\cudnn64_70.dll");
		Path FILE_CUDA_KERNEL=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation").resolve("kernel.cu.ptx");
		Path DIR_WAVE=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation\\Wave\\B-W-20150112");
		Path FILE_ALL_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\AllData.xml");
		Path FILE_TRAINING_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\TrainingData.xml");
		Path FILE_VALIDATION_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\ValidationData.xml");
		Path FILE_PARAMETER=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\WeightLcBdGs");
		
		ArrayList<Sequence> allSequence=Sequence.readXml(FILE_ALL_DATA);
		LabelList labelList=Sequence.LabelList.create(allSequence);
		ArrayList<Sequence> trainingSequence=Sequence.readXml(FILE_TRAINING_DATA);
		
		int SAMPLING_RATE=(int)SoundUtils.checkSamplingRate(allSequence.stream().map(s->s.getWaveFileName()).collect(Collectors.toList()), DIR_WAVE);
		STFTParam stftParam=new STFTParam(512, 32);
		int freqOffset=(int)(1000/stftParam.unitFrequency(SAMPLING_RATE));
		int freqLength=(int)(8000/stftParam.unitFrequency(SAMPLING_RATE))-freqOffset;
		int heightUpper=stftParam.spectrogramLength(8*32000);
		int dpssParam=4;
//		HashMap<Sequence, float[]> spec=SoundUtils.spectrogram(Sequence.wavePositionMap(trainingSequence, DIR_WAVE), stftParam, dpssParam, freqOffset, freqLength);
//		double[] specStat=SoundUtils.spectrogramMeanSd(trainingSequence.stream().map(s->spec.get(s)).collect(Collectors.toList()));
		int LOCAL_INPUT_HEIGHT=96;
		int FINAL_INPUT_HEIGHT=LOCAL_INPUT_HEIGHT;
		int NUM_LOWER_LABEL=3;
		IntBinaryOperator silentLabelFunc=(numUpperLabel, numLowreLabel)->numUpperLabel*numLowreLabel;
		IntBinaryOperator softmaxSizeFunc=(numUpperLabel, numLowreLabel)->numUpperLabel*numLowreLabel+1;
		int numIter=80;
		HyperParam dnnHyperParam=new HyperParam(stftParam, dpssParam, LOCAL_INPUT_HEIGHT, FINAL_INPUT_HEIGHT, NUM_LOWER_LABEL, freqOffset, freqLength, heightUpper, 4, numIter);
//		Config dnnConfig=new Config(true, silentLabelFunc, softmaxSizeFunc, specStat, FILE_CUDA_KERNEL, FILE_CUDNN_LIBRARY, DIR_WAVE);
		
		MersenneTwister random=new MersenneTwister(0);
//			DnnComputation.Param dnnParam=DnnComputation.trainLocalRecognition(trainingSequence, labelList, random, dnnHyperParam, dnnConfig);
//			DnnUtils.saveParam(dnnParam.getLayerParam(), FILE_PARAMETER);
		DnnComputation.Param dnnParam=new DnnComputation.Param(DnnUtils.loadParam(FILE_PARAMETER));
		
		ArrayList<Sequence> validationSequence=Sequence.readXml(FILE_VALIDATION_DATA);
		validationSequence=validationSequence.subList(0, 3).stream().collect(CollectionUtils.arrayListCollector());
		HashMap<Sequence, float[]> output=new HashMap<>();
		
		{
			ByteBuffer buf=ByteBuffer.wrap(Files.readAllBytes(FILE_PARAMETER.resolveSibling("OutputBdLcGs")));
			int numSequence=buf.getInt();
			for(Sequence seq: validationSequence)
			{
				int len=buf.getInt();
				float[] value=new float[len];
				output.put(seq, value);
				for(int i=0; i<len; ++i) value[i]=buf.getFloat();
			}
		}
		
		double smoothingConstant=1e-4;
		boolean posteriorToObservationProb=conv;
		HmmComputation.HyperParam hmmHyperParam=new HmmComputation.HyperParam(smoothingConstant, posteriorToObservationProb, NUM_LOWER_LABEL, stftParam);
		HmmComputation.Config hmmConfig=new HmmComputation.Config(executor, softmaxSizeFunc.applyAsInt(labelList.size(), NUM_LOWER_LABEL));
		HashMap<Sequence, ArrayList<double[]>> observationProb=HmmComputation.continuousPosteriorToObservationProb(output, trainingSequence, labelList, hmmHyperParam, hmmConfig);
		return validationSequence.stream().map(s->observationProb.get(s)).collect(CollectionUtils.arrayListCollector());
	}
	
	public static ArrayList<ArrayList<double[]>> observationProbLcGsBdGs(boolean conv) throws UnsupportedAudioFileException, IOException, NotConvergedException, CudaException, CudnnException
	{
		int numThread=4;
		Executor executor=new Executor(numThread);
		
		Path FILE_CUDNN_LIBRARY=Paths.get("I:\\koumura\\EclipseWorkspaceSe\\Dependencies\\cudnn\\cuda\\bin\\cudnn64_70.dll");
		Path FILE_CUDA_KERNEL=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation").resolve("kernel.cu.ptx");
		Path DIR_WAVE=Paths.get("I:\\koumura\\MultiDays2\\LabelCrossValidation\\Wave\\B-W-20150112");
		Path FILE_ALL_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\AllData.xml");
		Path FILE_TRAINING_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\TrainingData.xml");
		Path FILE_VALIDATION_DATA=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\Data\\Bird0\\ValidationData.xml");
		Path FILE_PARAMETER=Paths.get("I:\\koumura\\MultiDays2\\BirdsongRecognition\\WeightLcGsBdGs");
		
		ArrayList<Sequence> allSequence=Sequence.readXml(FILE_ALL_DATA);
		LabelList labelList=Sequence.LabelList.create(allSequence);
		ArrayList<Sequence> trainingSequence=Sequence.readXml(FILE_TRAINING_DATA);
		
		int SAMPLING_RATE=(int)SoundUtils.checkSamplingRate(allSequence.stream().map(s->s.getWaveFileName()).collect(Collectors.toList()), DIR_WAVE);
		STFTParam stftParam=new STFTParam(512, 32);
		int freqOffset=(int)(1000/stftParam.unitFrequency(SAMPLING_RATE));
		int freqLength=(int)(8000/stftParam.unitFrequency(SAMPLING_RATE))-freqOffset;
		int heightUpper=stftParam.spectrogramLength(8*32000);
		int dpssParam=4;
		HashMap<Sequence, float[]> spec=SoundUtils.spectrogram(Sequence.wavePositionMap(trainingSequence, DIR_WAVE), stftParam, dpssParam, freqOffset, freqLength);
		double[] specStat=SoundUtils.spectrogramMeanSd(trainingSequence.stream().map(s->spec.get(s)).collect(Collectors.toList()));
		int LOCAL_INPUT_HEIGHT=96;
		int FINAL_INPUT_HEIGHT=LOCAL_INPUT_HEIGHT*3;
		int NUM_LOWER_LABEL=3;
		IntBinaryOperator silentLabelFunc=(numUpperLabel, numLowreLabel)->numUpperLabel*numLowreLabel;
		IntBinaryOperator softmaxSizeFunc=(numUpperLabel, numLowreLabel)->numUpperLabel*numLowreLabel+1;
		int numIter=80;
		HyperParam dnnHyperParam=new HyperParam(stftParam, dpssParam, LOCAL_INPUT_HEIGHT, FINAL_INPUT_HEIGHT, NUM_LOWER_LABEL, freqOffset, freqLength, heightUpper, 4, numIter);
		Config dnnConfig=new Config(true, silentLabelFunc, softmaxSizeFunc, specStat, FILE_CUDA_KERNEL, FILE_CUDNN_LIBRARY, DIR_WAVE, ConvLayer.BackwardAlgorithm.FAST_NON_DETERMINISTIC);
		
		MersenneTwister random=new MersenneTwister(0);
//			DnnComputation.Param dnnParam=DnnComputation.trainLocalRecognition(trainingSequence, labelList, random, dnnHyperParam, dnnConfig);
//			DnnUtils.saveParam(dnnParam.getLayerParam(), FILE_PARAMETER);
		DnnComputation.Param dnnParam=new DnnComputation.Param(DnnUtils.loadParam(FILE_PARAMETER));
		
		ArrayList<Sequence> validationSequence=Sequence.readXml(FILE_VALIDATION_DATA);
		validationSequence=validationSequence.subList(0, 3).stream().collect(CollectionUtils.arrayListCollector());
		HashMap<Sequence, float[]> output=DnnComputation.localRecognition(validationSequence, labelList, dnnParam, dnnHyperParam, dnnConfig);
		
		double smoothingConstant=1e-4;
		boolean posteriorToObservationProb=conv;
		HmmComputation.HyperParam hmmHyperParam=new HmmComputation.HyperParam(smoothingConstant, posteriorToObservationProb, NUM_LOWER_LABEL, stftParam);
		HmmComputation.Config hmmConfig=new HmmComputation.Config(executor, softmaxSizeFunc.applyAsInt(labelList.size(), NUM_LOWER_LABEL));
		HashMap<Sequence, ArrayList<double[]>> observationProb=HmmComputation.continuousPosteriorToObservationProb(output, trainingSequence, labelList, hmmHyperParam, hmmConfig);
		return validationSequence.stream().map(s->observationProb.get(s)).collect(CollectionUtils.arrayListCollector());
	}
}
