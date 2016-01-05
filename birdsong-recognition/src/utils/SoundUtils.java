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
package utils;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

import computation.STFTParam;
import computation.Sequence;
import computation.Sequence.WavePosition;
import no.uib.cipr.matrix.NotConvergedException;

public class SoundUtils
{
	public static short[] readWaveShort(Path file) throws UnsupportedAudioFileException, IOException
	{
		AudioInputStream stream=AudioSystem.getAudioInputStream(file.toFile());
		AudioFormat format=stream.getFormat();
		int numChannel=format.getChannels();
		int bitDepth=format.getSampleSizeInBits();
		if(numChannel!=1) throw new UnsupportedAudioFileException("Number of channel must be 1.");
		if(bitDepth!=16) throw new UnsupportedAudioFileException("Bit depth must be 16.");
		long frameLength=stream.getFrameLength();
		if(frameLength>Integer.MAX_VALUE) throw new UnsupportedAudioFileException("Too long audio.");
		int frameLengthInt=(int)frameLength;
		float samplingRate=format.getSampleRate();
		byte[] data=new byte[frameLengthInt*bitDepth/8];
		int readLen=0;
		int len;
		while((len=stream.read(data, readLen, data.length-readLen))>=0)
		{
			readLen+=len;
		}
		stream.close();
		ByteBuffer buf=ByteBuffer.wrap(data);
		if(!format.isBigEndian()) buf=buf.order(ByteOrder.LITTLE_ENDIAN);
		short[] dataShort=new short[data.length/2];
		for(int i=0; i<dataShort.length; ++i) dataShort[i]=buf.getShort();
		return dataShort;
	}
	
	public static AudioFormat readAudioFormat(Path file) throws UnsupportedAudioFileException, IOException
	{
		AudioInputStream stream=AudioSystem.getAudioInputStream(file.toFile());
		AudioFormat format=stream.getFormat();
		stream.close();
		return format;
	}

	public static float[] spectrogram(short[] wave, STFTParam stftParam, double[] taper, int freqOffset, int freqLength, int wavePosition, int waveLength, double[] realData)
	{
		if(realData==null||realData.length!=stftParam.getFftLength()) realData=new double[stftParam.getFftLength()];
		int length=stftParam.spectrogramLength(waveLength);
		float[] spec=new float[length*freqLength];
		FastFourierTransformer fft=new FastFourierTransformer(DftNormalization.STANDARD);
		for(int i=0; i<length; ++i)
		{
			if(i<length-1)
			{
				for(int t=0; t<stftParam.getFftLength(); ++t) realData[t]=wave[wavePosition+i*stftParam.getShiftLength()+t]*taper[t];
			}
			else
			{
				int len=waveLength-(length-1)*stftParam.getShiftLength();
				for(int t=0; t<len; ++t) realData[t]=wave[wavePosition+i*stftParam.getShiftLength()+t]*taper[t];
				for(int t=len; t<stftParam.getFftLength(); ++t) realData[t]=0;
			}
			Complex[] complex=fft.transform(realData, TransformType.FORWARD);
			for(int f=0; f<freqLength; ++f) spec[i*freqLength+f]=(float)Math.log10(complex[f+freqOffset].abs());
		}
		return spec;
	}
	
	public static <K> HashMap<K, float[]> spectrogram(HashMap<K, WavePosition> waveFilePositionLength, STFTParam stftParam, int dpssParam, int freqOffset, int freqLength) throws UnsupportedAudioFileException, IOException, NotConvergedException
	{
		HashMap<Path, LinkedList<K>> fileMap=new HashMap<>();
		for(K key: waveFilePositionLength.keySet())
		{
			Path path=waveFilePositionLength.get(key).getWaveFile();
			fileMap.putIfAbsent(path, new LinkedList<>());
			fileMap.get(path).add(key);
		}
		HashMap<K, float[]> spectrogram=new HashMap<>(waveFilePositionLength.size()*4/3);
		double[] window=MathUtils.dpss(stftParam.getFftLength(), dpssParam);
		double[] realData=new double[stftParam.getFftLength()];
		for(Path file: fileMap.keySet())
		{
			short[] wave=readWaveShort(file);
			for(K seq: fileMap.get(file))
			{
				int pos=waveFilePositionLength.get(seq).getPosition();
				int len=waveFilePositionLength.get(seq).getLength();
				float[] spec=spectrogram(wave, stftParam, window, freqOffset, freqLength, pos, len, realData);
				spectrogram.put(seq, spec);
			}
		}
		return spectrogram;
	}

	public static float[] spectrogram(Sequence sequence, Path dirWave, STFTParam stftParam, int dpssParam, int frequencyOffset, int frequencyLength) throws UnsupportedAudioFileException, IOException, NotConvergedException
	{
		ArrayList<Sequence> sequenceList=new ArrayList<>(1);
		HashMap<Sequence, WavePosition> waveFilePosition=Sequence.wavePositionMap(sequenceList, dirWave);
		return spectrogram(waveFilePosition, stftParam, dpssParam, frequencyOffset, frequencyLength).get(sequence);
	}
	
	public static void whiteSpectrogram(Collection<float[]> spectrogram, double mean, double sd) throws IOException
	{
		for(float[] spec: spectrogram) for(int i=0; i<spec.length; ++i) spec[i]=(float)((spec[i]-mean)/sd);
	}
	
	public static double[] spectrogramMeanSd(List<float[]> spectrogram)
	{
		SummaryStatistics stat=new SummaryStatistics();
		for(float[] spec: spectrogram) for(float v: spec) stat.addValue(v);
		return new double[]{stat.getMean(), stat.getStandardDeviation()};
	}
	
	public static float checkSamplingRate(Collection<String> filename, Path dirWaveFile) throws UnsupportedAudioFileException, IOException
	{
		float samplingRate=Float.NaN;
		for(String fn: filename)
		{
			Path file=dirWaveFile.resolve(fn);
			AudioFormat format=readAudioFormat(file);
			if(Float.isNaN(samplingRate)) samplingRate=format.getSampleRate();
			if(samplingRate!=format.getSampleRate())
			{
				System.err.println("Sampling rate must be same across sequences.");
				return Float.NaN;
			}
		}
		return samplingRate;
	}
	
	public static BufferedImage spectrogramImage(float[] spectrogram, float blackValue, float whiteValue, int frequencyLength, int bottomMargin)
	{
		BufferedImage image=new BufferedImage(spectrogram.length/frequencyLength, frequencyLength+bottomMargin, BufferedImage.TYPE_INT_ARGB);
		Graphics2D graphics=(Graphics2D)image.getGraphics();
		for(int i=0; i<spectrogram.length; ++i)
		{
			double rgbd=(spectrogram[i]-blackValue)/(whiteValue-blackValue)*255;
			int rgbi;
			if(rgbd>255) rgbi=255;
			else if(rgbd<0) rgbi=0;
			else rgbi=(int)rgbd;
			Color color=new Color(rgbi, rgbi, rgbi);
			graphics.setPaint(color);
			int y=frequencyLength-1-i%frequencyLength;
			int x=i/frequencyLength;
			graphics.fillRect(x, y, 1, 1);
		}
		return image;
	}
}
