/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This file is part of birdsong-recognition.
 * 
 * Birdsong-recognition is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Birdsong-recognition is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with birdsong-recognition.  If not, see <http://www.gnu.org/licenses/>.
 */
package computation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.IntBinaryOperator;

import javax.sound.sampled.UnsupportedAudioFileException;

import org.apache.commons.math3.random.MersenneTwister;

import computation.Sequence.LabelList;
import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.FloatType;
import cudnn.layer.ConvLayer;
import cudnn.layer.SeqSoftmaxConvLayer;
import cudnn.learner.Adam;
import cudnn.network.SeqNetwork;
import jcuda.jcublas.JCublas;
import no.uib.cipr.matrix.NotConvergedException;
import utils.DnnUtils;

public class DnnComputation
{
	public static Param trainLocalRecognition(ArrayList<Sequence> trainingSequence, LabelList labelList, MersenneTwister random, HyperParam hyperParam, Config config) throws IOException, CudnnException, CudaException, UnsupportedAudioFileException, NotConvergedException
	{
		BatchData data=BatchData.create(trainingSequence, random, hyperParam.numSubLabel, config.silentLabelFunc, hyperParam.finalInputHeight, config.dirWaveFile, hyperParam.stftParam, hyperParam.dpssParam, hyperParam.freqOffset, hyperParam.freqLength, config.spectrogramMeanSd, hyperParam.inputHeightUpper);
		data.pack(data.getPackedHeight(), hyperParam.freqLength, hyperParam.batchSizeUpper, labelList, DnnUtils.filterSizeList525242());
		data.batch(data.getMaxBatchSize(), random, hyperParam.freqLength);

		Cudnn cudnn=new Cudnn(config.fileCudnnLibrary);
		CudaDriver driver=new CudaDriver();
		driver.load(config.fileCudaKernel.toAbsolutePath().toString());
		SeqNetwork network=DnnUtils.netinnetNetwork(data.getDataLayerHeight(), hyperParam.freqLength, config.softmaxSizeFunc.applyAsInt(labelList.size(), hyperParam.numSubLabel), data.getMaxBatchSize(), cudnn, driver, hyperParam.localInputHeight, hyperParam.finalInputHeight, config.backwardAlogorithm, hyperParam.numConvChannel, hyperParam.fullConnectionSize);

		network.cudaMalloc();
		if(config.verbose)
		{
			network.initError();
			data.setValidLabelSize(network);
		}
		
		DnnUtils.initParam(network.getLayer(), random);
		network.copyParamToDev();
		Adam learner=new Adam(network, FloatType.SINGLE, cudnn, driver);
		learner.init();
		for(int iter=0; iter<hyperParam.numIter; ++iter)
		{
			{
				int batch=random.nextInt(data.getNumBatch());
				int shift=random.nextInt(network.getLabelShiftUpperH());
				network.setDataF(data.getBatchData().get(batch));
				network.copyLabelToDevice(data.getBatchLabel().get(batch));
				network.setLabelShift(shift, 0);
				network.copyDataToDeviceHeightDisp();
//				Cuda.deviceSynchronize();
				learner.iteration();
			}
			if(config.verbose)
			{
				double error=0;
				for(int batch=0; batch<data.getNumBatch(); ++batch)
				{
					network.setDataF(data.getBatchData().get(batch));
					network.copyLabelToDevice(data.getBatchLabel().get(batch));
					for(int shift=0; shift<network.getLabelShiftUpperH(); ++shift)
					{
						network.setLabelShift(shift, 0);
						network.copyDataToDeviceHeightDisp();
//						Cuda.deviceSynchronize();
						network.compForward(driver, cudnn);
						error+=network.compError(driver);
					}
				}
				error/=data.getSumValidLabelSize();
				System.out.println("iter:"+iter+" Er:"+error);
			}
		}
		network.copyParamFromDev();
		network.destroy(cudnn);
		learner.destroy();
		cudnn.destroty();
		JCublas.cublasShutdown();  //memory leak will happen if cublasShutdown is called after driver.destroy()
		driver.destroy();
		
		return new Param(DnnUtils.copyParamFromLayer(network.getLayer()));
	}
	
	public static HashMap<Sequence, float[]> localRecognition(ArrayList<Sequence> validationSequence, LabelList labelList, Param param, HyperParam hyperParam, Config config) throws IOException, UnsupportedAudioFileException, NotConvergedException, CudaException, CudnnException
	{
		BatchData data=BatchData.create(validationSequence, null, hyperParam.numSubLabel, config.silentLabelFunc, hyperParam.finalInputHeight, config.dirWaveFile, hyperParam.stftParam, hyperParam.dpssParam, hyperParam.freqOffset, hyperParam.freqLength, config.spectrogramMeanSd, hyperParam.inputHeightUpper);
		data.pack(data.getPackedHeight(), hyperParam.freqLength, hyperParam.batchSizeUpper, labelList, DnnUtils.filterSizeList525242());
		data.batch(data.getMaxBatchSize(), null, hyperParam.freqLength);
		
		Cudnn cudnn=new Cudnn(config.fileCudnnLibrary);
		CudaDriver driver=new CudaDriver();
		driver.load(config.fileCudaKernel.toAbsolutePath().toString());
		SeqNetwork network=DnnUtils.netinnetNetwork(data.getDataLayerHeight(), hyperParam.freqLength, config.softmaxSizeFunc.applyAsInt(labelList.size(), hyperParam.numSubLabel), data.getMaxBatchSize(), cudnn, driver, hyperParam.localInputHeight, hyperParam.finalInputHeight, config.backwardAlogorithm, hyperParam.numConvChannel, hyperParam.fullConnectionSize);
		SeqSoftmaxConvLayer outputLayer=(SeqSoftmaxConvLayer)network.getLayer().get(network.getLayer().size()-1);

		network.cudaMalloc();
		ArrayList<ArrayList<float[]>> networkOutput=new ArrayList<>(data.getNumBatch());
		DnnUtils.copyParamToLayer(param.getLayerParam(), network.getLayer());
		network.copyParamToDev();
		for(int batch=0; batch<data.getNumBatch(); ++batch)
		{
			networkOutput.add(new ArrayList<>(network.getLabelShiftUpperH()));
			network.setDataF(data.getBatchData().get(batch));
			for(int shift=0; shift<network.getLabelShiftUpperH(); ++shift)
			{
				network.setLabelShift(shift, 0);
				network.copyDataToDeviceHeightDisp();
//				Cuda.deviceSynchronize();
				network.compForward(driver, cudnn);
				float[] o=Cuda.memcpyDeviceToFloatArray(outputLayer.getValueDev(), outputLayer.sizeCHW()*data.getMaxBatchSize());
				networkOutput.get(batch).add(o);
			}
		}
		network.destroy(cudnn);
		cudnn.destroty();
		JCublas.cublasShutdown();  //memory leak will happen if cublasShutdown is called after driver.destroy()
		driver.destroy();
		
		HashMap<Sequence, float[]> sequenceOutput=new HashMap<>(validationSequence.size()*4/3);
		for(int s=0; s<data.getSpecLabelPackBegin().size(); ++s)
		{
			int pack=data.getSpecLabelPackBegin().get(s)[0];
			int batch=data.getPackBatchIndex().get(pack)[0];
			int batchIndex=data.getPackBatchIndex().get(pack)[1];
			int packBegin=data.getSpecLabelPackBegin().get(s)[1];
			
			int outputHeight=data.getSpectrogram().get(s).length/hyperParam.freqLength-data.specMarginSum();
			int outputWidth=outputLayer.getSingleSize();
			float[] array=new float[outputHeight*outputWidth];
			sequenceOutput.put(validationSequence.get(s), array);
			int arrayIndex=0;
			for(int sy=0; sy<outputHeight; ++sy)
			{
				int py=packBegin+sy;
				int shift=py%network.getLabelShiftUpperH();
				int oy=py/network.getLabelShiftUpperH();
				for(int oc=0; oc<outputLayer.getSingleSize(); ++oc)
				{
					int index=oy+(oc+batchIndex*outputLayer.getSingleSize())*outputLayer.getHeight();
					float value=networkOutput.get(batch).get(shift)[index];
					array[arrayIndex++]=value;
				}
			}
		}
		return sequenceOutput;
	}

	public static class Config
	{
		private boolean verbose;
		private IntBinaryOperator silentLabelFunc, softmaxSizeFunc;
		private double[] spectrogramMeanSd;
		private Path fileCudaKernel, fileCudnnLibrary, dirWaveFile;
		private ConvLayer.BackwardAlgorithm backwardAlogorithm;

		public Config(boolean verbose, IntBinaryOperator silentLabelFunc, IntBinaryOperator softmaxSizeFunc,
				double[] spectrogramMeanSd, Path fileCudaKernel, Path fileCudnnLibrary, Path dirWaveFile, ConvLayer.BackwardAlgorithm backwardAlogorithm) {
			this.verbose = verbose;
			this.silentLabelFunc = silentLabelFunc;
			this.softmaxSizeFunc = softmaxSizeFunc;
			this.spectrogramMeanSd = spectrogramMeanSd;
			this.fileCudaKernel = fileCudaKernel;
			this.fileCudnnLibrary = fileCudnnLibrary;
			this.dirWaveFile = dirWaveFile;
			this.backwardAlogorithm=backwardAlogorithm;
		}
	}
	
	public static class HyperParam
	{
		private STFTParam stftParam;
		private int dpssParam, localInputHeight, finalInputHeight, numSubLabel, freqOffset, freqLength, inputHeightUpper, batchSizeUpper, numIter, numConvChannel, fullConnectionSize;
		
		public HyperParam(STFTParam stftParam, int dpssParam, int localInputHeight, int finalInputHeight, int numSubLabel,
				int freqOffset, int freqLength, int inputHeightUpper, int batchSizeUpper, int numIter, int numConvChannel, int fullConnectionSize) {
			this.stftParam = stftParam;
			this.dpssParam = dpssParam;
			this.localInputHeight = localInputHeight;
			this.finalInputHeight = finalInputHeight;
			this.numSubLabel = numSubLabel;
			this.freqOffset = freqOffset;
			this.freqLength = freqLength;
			this.inputHeightUpper = inputHeightUpper;
			this.batchSizeUpper = batchSizeUpper;
			this.numIter = numIter;
			this.numConvChannel=numConvChannel;
			this.fullConnectionSize=fullConnectionSize;
		}
	}
	
	public static class Param
	{
		private ArrayList<float[]> layerParam;

		public Param(ArrayList<float[]> layerParam)
		{
			this.layerParam = layerParam;
		}

		public ArrayList<float[]> getLayerParam() {
			return layerParam;
		}
		
		public void save(Path file) throws IOException
		{
			DnnUtils.saveParam(layerParam, file);
		}
		
		public static Param load(Path file) throws IOException
		{
			return new Param(DnnUtils.loadParam(file));
		}
	}
}
