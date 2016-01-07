/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This file is part of Birdsong Recognition.
 * 
 * Birdsong Recognition is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Birdsong Recognition is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Birdsong Recognition.  If not, see <http://www.gnu.org/licenses/>.
 */
package computation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.function.IntBinaryOperator;
import java.util.stream.Collectors;

import javax.sound.sampled.UnsupportedAudioFileException;

import org.apache.commons.math3.random.MersenneTwister;

import computation.Sequence.LabelList;
import computation.Sequence.Note;
import computation.Sequence.WavePosition;
import cudnn.network.SeqNetwork;
import no.uib.cipr.matrix.NotConvergedException;
import utils.ArrayUtils;
import utils.DnnUtils;
import utils.MathUtils;
import utils.Pair;
import utils.RandomUtils;
import utils.SoundUtils;

/**
 * This class handles data for DNN computation.
 * 
 * @author koumura
 */
public class BatchData
{
	private ArrayList<ArrayList<float[]>> batchData;
	private ArrayList<byte[]> batchLabel;
	private int dataLayerHeight, maxBatchSize, numBatch, labelHeight, sumValidLabelSize, packedHeight;
	private ArrayList<int[]> batchValidLabelSize;
	private ArrayList<float[]> spectrogram;
	private ArrayList<ArrayList<Integer>> packedSpectrogram;
	private List<Sequence> sequence;
	
	private ArrayList<float[]> inputData;
	private ArrayList<byte[]> inputLabel;
	
	private ArrayList<int[]> specLabelPackBegin, packBatchIndex;
	
	private int numLowerLabel, finalInputHeight;
	private IntBinaryOperator silentLabelFunc;
	private byte silentLabel;
	private int specMarginBegin, specMarginEnd;
	
	private STFTParam stftParam;
	
	public int getSumValidLabelSize() {
		return sumValidLabelSize;
	}

	public int specMarginSum(){return specMarginBegin+specMarginEnd;}
	
	public ArrayList<byte[]> getBatchLabel() {
		return batchLabel;
	}

	public int getDataLayerHeight() {
		return dataLayerHeight;
	}

	public int getMaxBatchSize() {
		return maxBatchSize;
	}

	public int getPackedHeight() {
		return packedHeight;
	}

	public int getNumBatch() {
		return numBatch;
	}

	public ArrayList<ArrayList<float[]>> getBatchData() {
		return batchData;
	}

	public ArrayList<int[]> getSpecLabelPackBegin() {
		return specLabelPackBegin;
	}

	public ArrayList<int[]> getPackBatchIndex() {
		return packBatchIndex;
	}

	public ArrayList<float[]> getSpectrogram() {
		return spectrogram;
	}

	private BatchData(ArrayList<float[]> spectrogram, int packedHeight, ArrayList<ArrayList<Integer>> packedSpectrogram, List<Sequence> sequence, int numLowerLabel, IntBinaryOperator silentLabelFunc, int specMarginBegin, int specMarginEnd, int finalInputHeight, STFTParam stftParam)
	{
		this.spectrogram = spectrogram;
		this.packedHeight = packedHeight;
		this.packedSpectrogram=packedSpectrogram;
		this.sequence=sequence;
		this.numLowerLabel=numLowerLabel;
		this.finalInputHeight=finalInputHeight;
		this.silentLabelFunc=silentLabelFunc;
		this.specMarginBegin=specMarginBegin;
		this.specMarginEnd=specMarginEnd;
		this.stftParam=stftParam;
	}
	
	private static ArrayList<ArrayList<Integer>> packedSpectrogramIndex(int heightUpper, List<float[]> spectrogram, MersenneTwister random, int inputWidth)
	{
		int[] shuffleSpectrogram;
		if(random!=null) shuffleSpectrogram=RandomUtils.permutation(spectrogram.size(), random);
		else shuffleSpectrogram=ArrayUtils.createSequence(0, spectrogram.size());
		ArrayList<int[]> dataHeight=new ArrayList<int[]>();
		HashMap<Integer, LinkedList<Integer>> inputSpec=new HashMap<>();
		for(int s: shuffleSpectrogram)
		{
			float[] spec=spectrogram.get(s);
			int specHeight=spec.length/inputWidth;
			int[] minHeight=null;
			if(dataHeight.size()>0) minHeight=Collections.min(dataHeight, (o1, o2)->o1[1]-o2[1]);

			if(dataHeight.size()>0 && heightUpper-minHeight[1]>=specHeight)
			{
				minHeight[1]+=specHeight;
				inputSpec.putIfAbsent(minHeight[0], new LinkedList<>());
				inputSpec.get(minHeight[0]).add(s);
			}
			else
			{
				inputSpec.putIfAbsent(dataHeight.size(), new LinkedList<>());
				inputSpec.get(dataHeight.size()).add(s);
				dataHeight.add(new int[]{dataHeight.size(), specHeight});
			}
		}
		ArrayList<ArrayList<Integer>> packedIndex=new ArrayList<>(inputSpec.size());
		for(int packed=0; packed<inputSpec.size(); ++packed) packedIndex.add(new ArrayList<Integer>(inputSpec.get(packed)));
		return packedIndex;
	}
	
	private static ArrayList<float[]> packSpectrogram(List<ArrayList<Integer>> packedIndex, ArrayList<float[]> spectrogram, int packedHeight, int inputWidth)
	{
		ArrayList<float[]> inputData=new ArrayList<>();
		for(ArrayList<Integer> index: packedIndex)
		{
			float[] d=new float[packedHeight*inputWidth];
			inputData.add(d);
			int cumLength=0;
			for(int s: index)
			{
				float[] spec=spectrogram.get(s);
				System.arraycopy(spec, 0, d, cumLength, spec.length);
				cumLength+=spec.length;
			}
		}
		return inputData;
	}
	
	/**
	 * Packs input spectrograms for efficient computation. 
	 * This method must be called before {@link #batch}.
	 * 
	 * @param packedHeight
	 * @param inputWidth
	 * @param batchSizeUpper
	 * @param labelList
	 * @param layerFilterSizeList
	 * @throws IOException
	 */
	public void pack(int packedHeight, int inputWidth, int batchSizeUpper, LabelList labelList, ArrayList<Pair<DnnUtils.LayerType, Integer>> layerFilterSizeList) throws IOException
	{
		this.packedHeight=packedHeight;
		dataLayerHeight=DnnUtils.nextSmallestInputSize(layerFilterSizeList, packedHeight);
		this.packedHeight=dataLayerHeight+DnnUtils.sumStride(layerFilterSizeList)-1;
		labelHeight=this.packedHeight-finalInputHeight+1;
		
		inputData=packSpectrogram(packedSpectrogram, spectrogram, this.packedHeight, inputWidth);

		silentLabel=(byte)silentLabelFunc.applyAsInt(labelList.size(), numLowerLabel);
		
		inputLabel=new ArrayList<>();
		specLabelPackBegin=new ArrayList<int[]>(spectrogram.size());
		for(int i=0; i<spectrogram.size(); ++i) specLabelPackBegin.add(null);
		for(int in=0; in<inputData.size(); ++in)
		{
			byte[] la=new byte[labelHeight];
			Arrays.fill(la, silentLabel);
			inputLabel.add(la);
			int y=0;
			for(int sp: packedSpectrogram.get(in))
			{
				specLabelPackBegin.set(sp, new int[]{in, y});
				for(Note li: sequence.get(sp).getNote())
				{
					int specBegin=stftParam.spectrogramPosition(li.getPosition())+y;
					int specLen=stftParam.spectrogramPosition(li.getPosition()+li.getLength())-specBegin+y;
					for(int sub=0; sub<numLowerLabel; ++sub)
					{
						int begin=MathUtils.subIntervalBegin(specLen, numLowerLabel, sub)+specBegin;
						int end=MathUtils.subIntervalBegin(specLen, numLowerLabel, sub+1)+specBegin;
						Arrays.fill(la, begin, end, (byte)(labelList.indexOf(li.getLabel())*numLowerLabel+sub));
					}
				}
				y+=spectrogram.get(sp).length/inputWidth-(specMarginBegin+specMarginEnd);
				if(sp==packedSpectrogram.get(in).get(packedSpectrogram.get(in).size()-1)) break;
				Arrays.fill(la, y, y+specMarginBegin+specMarginEnd, (byte)-1);
				y+=specMarginBegin+specMarginEnd;
			}
			Arrays.fill(la, y, la.length, (byte)-1);
		}

		numBatch=MathUtils.ceil(inputData.size(), batchSizeUpper);
		maxBatchSize=MathUtils.ceil(inputData.size(), numBatch);
	}

	/**
	 * Makes batch inputs.
	 * This method must be called before training or recognition with a DNN.
	 * @param maxBatchSize
	 * @param randomShuffle
	 * @param inputWidth
	 * @throws IOException
	 */
	public void batch(int maxBatchSize, MersenneTwister randomShuffle, int inputWidth) throws IOException
	{
		this.maxBatchSize=maxBatchSize;
		numBatch=MathUtils.ceil(inputData.size(), maxBatchSize);
		batchData=new ArrayList<>(numBatch);
		batchLabel=new ArrayList<>(numBatch); 
		int[] shuffleInput;
		if(randomShuffle!=null) shuffleInput=RandomUtils.permutation(inputData.size(), randomShuffle);
		else shuffleInput=ArrayUtils.createSequence(0, inputData.size());
		packBatchIndex=new ArrayList<int[]>();
		for(int p=0; p<inputData.size(); ++p) packBatchIndex.add(null);
		for(int b=0; b<numBatch; ++b)
		{
			batchData.add(new ArrayList<float[]>(maxBatchSize));
			batchLabel.add(new byte[labelHeight*maxBatchSize]);
			for(int i=0; i<maxBatchSize; ++i)
			{
				int in=i*numBatch+b;
				if(in<inputData.size())
				{
					int si=shuffleInput[in];
					batchData.get(b).add(inputData.get(si));
					System.arraycopy(inputLabel.get(si), 0, batchLabel.get(b), i*labelHeight, labelHeight);
					packBatchIndex.set(si, new int[]{b, i});
				}
				else
				{
					for(int i1=i; i1<maxBatchSize; ++i1) batchData.get(b).add(new float[packedHeight*inputWidth]);
					Arrays.fill(batchLabel.get(b), i*packedHeight, batchLabel.get(b).length, (byte)-1);
					break;
				}
			}
		}
	}
	
	/**
	 * Counts the number of valid labels in the batch data.
	 * @param network
	 */
	public void setValidLabelSize(SeqNetwork network)
	{
		batchValidLabelSize=new ArrayList<>(numBatch);
		for(int b=0; b<numBatch; ++b) batchValidLabelSize.add(network.countValidLabelSize(batchLabel.get(b)));
		
		sumValidLabelSize=batchValidLabelSize.stream().mapToInt(size->Arrays.stream(size).sum()).sum();
	}
	
	/**
	 * Creates {@link BatchData} by loading wave files and converting them into spectrograms.  
	 * 
	 * @param numLowerLabel Number of sub-divisions in a single element.
	 * @param silentLabelFunc A converter from the number of labels and sub-divisions to the index for the silent label.
	 * @param finalInputHeight Input height of the spectrogram, seen from the final output layer.
	 * @param waveFileDir Directory of wave files.
	 * @param stftParam Parameters for short time Fourier transformation.
	 * @param dpssParam Parameter for discrete prolate spheroidal sequences. 
	 * @param freqOffset Beginning of the frequency band.
	 * @param freqLength Length of the frequency band.
	 * @param spectrogramMeanSd Mean and SD of the spectrograms in the training data. Used for input scaling.
	 * @param inputHeightUpper Upper value of the combined input spectrogram.
	 * @return
	 * @throws IOException
	 * @throws UnsupportedAudioFileException
	 * @throws NotConvergedException
	 */
	public static BatchData create(List<Sequence> sequence, MersenneTwister random, int numLowerLabel, IntBinaryOperator silentLabelFunc, int finalInputHeight, Path waveFileDir, STFTParam stftParam, int dpssParam, int freqOffset, int freqLength, double[] spectrogramMeanSd, int inputHeightUpper) throws IOException, UnsupportedAudioFileException, NotConvergedException
	{
		int marginBegin=finalInputHeight/2;
		int marginEnd=finalInputHeight/2-1;
		HashMap<Sequence, WavePosition> wavePosition=Sequence.wavePositionMap(sequence, waveFileDir);
		for(WavePosition wp: wavePosition.values())
		{
			int waveBegin=wp.getPosition()-marginBegin*stftParam.getShiftLength();
			int waveEnd=wp.getEnd()+marginEnd*stftParam.getShiftLength();
			wp.setPosition(waveBegin);
			wp.setLength(waveEnd-waveBegin);
		}
		HashMap<Sequence, float[]> spectrogram=SoundUtils.spectrogram(wavePosition, stftParam, dpssParam, freqOffset, freqLength);
		SoundUtils.whiteSpectrogram(spectrogram.values(), spectrogramMeanSd[0], spectrogramMeanSd[1]);
		
		ArrayList<float[]> spectrogramList=sequence.stream().map(s->spectrogram.get(s)).collect(Collectors.toCollection(ArrayList::new));
		ArrayList<ArrayList<Integer>> packedSpectrogram=packedSpectrogramIndex(inputHeightUpper, spectrogramList, random, freqLength);
		
		int packedHeight=packedSpectrogram.stream()
				.mapToInt(spec->spec.stream().mapToInt(s->spectrogram.get(sequence.get(s)).length/freqLength).sum())
				.max().getAsInt();
		
		return new BatchData(spectrogramList, packedHeight, packedSpectrogram, sequence, numLowerLabel, silentLabelFunc, marginBegin, marginEnd, finalInputHeight, stftParam);
	}
}
