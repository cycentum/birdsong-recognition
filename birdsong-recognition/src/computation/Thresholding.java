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
package computation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import javax.sound.sampled.UnsupportedAudioFileException;

import org.w3c.dom.Element;
import org.w3c.dom.Node;

import computation.Sequence.Note;
import errorcomputation.Matching;
import no.uib.cipr.matrix.NotConvergedException;
import utils.CollectionUtils;
import utils.Executor;
import utils.Pair;
import utils.SoundUtils;
import utils.XmlUtils;

public class Thresholding
{
	private static LinkedList<int[]> soundInterval(double[] amplitude, double threshold)
	{
		LinkedList<int[]> soundInterval=new LinkedList<int[]>();
		for(int t=0; t<amplitude.length; ++t)
		{
			if(amplitude[t]>=threshold && (soundInterval.size()==0 || soundInterval.getLast()[1]!=-1)) soundInterval.add(new int[]{t, -1, 0});
			else if(amplitude[t]<threshold && soundInterval.size()>0 && soundInterval.getLast()[1]==-1) soundInterval.getLast()[1]=(t);
		}
		if(soundInterval.size()>0&&soundInterval.getLast()[1]==-1) soundInterval.getLast()[1]=(amplitude.length);
		
		for(int[] si: soundInterval) si[1]-=si[0];
		
		return soundInterval;
	}
	
	private static void removeShortGap(ArrayList<int[]> soundInterval, int gapLengthLower)
	{
		if(gapLengthLower>0)
		{
			int remmainingTail=0;
			LinkedList<int[]> remove=new LinkedList<>();
			for(int i=0; i<soundInterval.size()-1; ++i)
			{
				if(soundInterval.get(i+1)[0]-(soundInterval.get(i)[0]+soundInterval.get(i)[1])<=gapLengthLower)
				{
					remove.add(soundInterval.get(i+1));
					soundInterval.get(remmainingTail)[1]=(soundInterval.get(i+1)[0]+soundInterval.get(i+1)[1]-soundInterval.get(remmainingTail)[0]);
				}
				else remmainingTail=i+1;
			}
			soundInterval.removeAll(remove);
		}
	}
	
	private static void removeShortNote(ArrayList<int[]> soundInterval, int noteLengthLower)
	{
		if(noteLengthLower>0)
		{
			soundInterval.removeIf(si -> si[1]<=noteLengthLower);
		}
	}
	
	private static double[] spectrogramAmplitude(float[] spec, int freqLength) throws IOException
	{
		double[] amp=new double[spec.length/freqLength];
		for(int t=0; t<amp.length; ++t)
		{
			for(int f=0; f<freqLength; ++f) amp[t]+=spec[t*freqLength+f];
			amp[t]/=freqLength;
		}
		return amp;
	}
	private static HashMap<Sequence, double[]> spectrogramAmplitude(Collection<Sequence> sequence, HyperParameter hyperParameter, Config config) throws IOException, UnsupportedAudioFileException, NotConvergedException
	{
		HashMap<Sequence, float[]> spectrogram=SoundUtils.spectrogram(Sequence.wavePositionMap(sequence, config.waveFileDir), hyperParameter.stftParam, hyperParameter.dpssParam, hyperParameter.freqOffset, hyperParameter.freqLength);
		HashMap<Sequence, double[]> spectrogramAmplitude=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence) spectrogramAmplitude.put(seq, spectrogramAmplitude(spectrogram.get(seq), hyperParameter.freqLength));
		return spectrogramAmplitude;
	}
	
	private static LinkedList<Pair<Sequence, Double>> specIndexAmplitude(HashMap<Sequence, double[]> spectrogramAmplitude)
	{
		LinkedList<Pair<Sequence, Double>> ampAll=new LinkedList<>();
		for(Sequence seq: spectrogramAmplitude.keySet()) for(double a: spectrogramAmplitude.get(seq)) ampAll.add(new Pair<>(seq, a));
		return ampAll;
	}
	
	public static Parameter train(Collection<Sequence> sequence, HyperParameter hyperParameter, Config config) throws InterruptedException, ExecutionException, UnsupportedAudioFileException, IOException, NotConvergedException
	{
		HashMap<Sequence, double[]> spectrogramAmplitude=spectrogramAmplitude(sequence, hyperParameter, config);
		ArrayList<Pair<Sequence, Double>> specIndexAmp=specIndexAmplitude(spectrogramAmplitude).stream().collect(CollectionUtils.arrayListCollector());
		specIndexAmp.sort((o1, o2)->Double.compare(o1.get1(), o2.get1()));
		
		HashMap<Sequence, ArrayList<int[]>> soundInterval=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence) soundInterval.put(seq, soundInterval(spectrogramAmplitude.get(seq), specIndexAmp.get(0).get1()).stream().collect(CollectionUtils.arrayListCollector()));
		
		HashMap<Sequence, ArrayList<int[]>> correctSoundSpecInterval=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence)
		{
			ArrayList<int[]> correct=new ArrayList<>(seq.getNote().size());
			correctSoundSpecInterval.put(seq, correct);
			for(Note note: seq.getNote())
			{
				int begin=hyperParameter.stftParam.spectrogramPosition(note.getPosition());
				int end=hyperParameter.stftParam.spectrogramPosition(note.getPosition()+note.getLength());
				correct.add(new int[]{begin, end-begin, 0});
			}
		}
		
		HashMap<Sequence, Integer> sequenceSpecLength=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence) sequenceSpecLength.put(seq, hyperParameter.stftParam.spectrogramLength(seq.getLength()));
		
		HashMap<Sequence, int[][]> currentScore=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence)
		{
			int[][] score=new int[hyperParameter.gapLowerUpper][hyperParameter.noteLowerUpper];
			currentScore.put(seq, score);
			
			ArrayList<Integer> gapLengthList=new ArrayList<Integer>(soundInterval.get(seq).size()-1);
			for(int i=0; i<soundInterval.get(seq).size()-1; ++i) gapLengthList.add(soundInterval.get(seq).get(i+1)[0]-(soundInterval.get(seq).get(i)[0]+soundInterval.get(seq).get(i)[1]));
			gapLengthList=gapLengthList.stream().collect(Collectors.toSet()).stream().filter(gl->gl<hyperParameter.gapLowerUpper).collect(CollectionUtils.arrayListCollector());
			gapLengthList.add(hyperParameter.gapLowerUpper);
			gapLengthList.add(0);
			Collections.sort(gapLengthList);
			for(int gi=0; gi<gapLengthList.size()-1; ++gi)
			{
				ArrayList<int[]> shortGapRemoved=CollectionUtils.deepCopy(soundInterval.get(seq));
				removeShortGap(shortGapRemoved, gapLengthList.get(gi));
				
				ArrayList<Integer> noteLengthList=shortGapRemoved.stream()
					.map(si->si[1])
					.filter(nl->nl<hyperParameter.noteLowerUpper)
					.collect(Collectors.toSet()).stream()
					.collect(CollectionUtils.arrayListCollector());
				noteLengthList.add(0);
				noteLengthList.add(hyperParameter.noteLowerUpper);
				Collections.sort(noteLengthList);
				for(int ni=0; ni<noteLengthList.size()-1; ++ni)
				{
					ArrayList<int[]> shortNoteRemoved=CollectionUtils.deepCopy(shortGapRemoved);
					removeShortNote(shortNoteRemoved, noteLengthList.get(ni));
					int ms=Matching.computeMatchedLength(correctSoundSpecInterval.get(seq), shortNoteRemoved, sequenceSpecLength.get(seq));
					
					for(int i=noteLengthList.get(ni); i<noteLengthList.get(ni+1); ++i) score[gapLengthList.get(gi)][i]=ms;
				}
				
				for(int i=gapLengthList.get(gi)+1; i<gapLengthList.get(gi+1); ++i) System.arraycopy(score[gapLengthList.get(gi)], 0, score[i], 0, score[i].length);
			}
		}
		
		int bestScore=0;
		LinkedList<Parameter> bestParam=new LinkedList<>();
		for(int index=0; index<specIndexAmp.size(); ++index)
		{
			if(config.verbose && index%(specIndexAmp.size()/20)==0) System.out.println("Thresholding training "+(int)((double)index/specIndexAmp.size()*100)+"%");
			
			Sequence changedSpec=specIndexAmp.get(index).get0();
			double threshold=specIndexAmp.get(index).get1();
			ArrayList<int[]> changedSoundInterval=soundInterval(spectrogramAmplitude.get(changedSpec), threshold).stream().collect(CollectionUtils.arrayListCollector());
			int[][] changedScore=currentScore.get(changedSpec);
			
			ArrayList<Integer> gapLengthList=new ArrayList<Integer>(changedSoundInterval.size()-1);
			for(int i=0; i<changedSoundInterval.size()-1; ++i) gapLengthList.add(changedSoundInterval.get(i+1)[0]-(changedSoundInterval.get(i)[0]+changedSoundInterval.get(i)[1]));
			gapLengthList=gapLengthList.stream().collect(Collectors.toSet()).stream().filter(gl->gl<hyperParameter.gapLowerUpper).collect(CollectionUtils.arrayListCollector());
			gapLengthList.add(0);

			ArrayList<Callable<Pair<Integer, int[]>>> job=new ArrayList<>(config.executor.getNumThread());
			for(int gapLength: gapLengthList)
			{
				job.add(new Callable<Pair<Integer, int[]>>(){
					@Override
					public Pair<Integer, int[]> call()
					{
						ArrayList<int[]> shortGapRemoved=CollectionUtils.deepCopy(changedSoundInterval);
						removeShortGap(shortGapRemoved, gapLength);
						
						ArrayList<Integer> noteLengthList=new ArrayList<Integer>(shortGapRemoved.stream()
							.map(si->si[1])
							.filter(nl->nl<hyperParameter.noteLowerUpper).collect(Collectors.toSet()));
						noteLengthList.add(0);
						noteLengthList.add(hyperParameter.noteLowerUpper);
						Collections.sort(noteLengthList);
						int[] score=new int[hyperParameter.noteLowerUpper];
						for(int ni=0; ni<noteLengthList.size()-1; ++ni)
						{
							int noteLength=noteLengthList.get(ni);
							ArrayList<int[]> shortNoteRemoved=CollectionUtils.deepCopy(shortGapRemoved);
							removeShortNote(shortNoteRemoved, noteLength);
							int ms=Matching.computeMatchedLength(correctSoundSpecInterval.get(changedSpec), shortNoteRemoved, sequenceSpecLength.get(changedSpec));
							for(int i=noteLengthList.get(ni); i<noteLengthList.get(ni+1); ++i) score[i]=ms;
						}
						return new Pair<>(gapLength, score);
					}
				});
			}
			List<Future<Pair<Integer, int[]>>> future=config.executor.get().invokeAll(job);
			HashMap<Integer, int[]> parallel=new HashMap<>(future.size()*4/3);
			for(Future<Pair<Integer, int[]>> f: future)
			{
				Pair<Integer, int[]> p=f.get();
				parallel.put(p.get0(), p.get1());
			}
			gapLengthList.add(hyperParameter.gapLowerUpper);
			Collections.sort(gapLengthList);
			
			for(int gi=0; gi<gapLengthList.size()-1; ++gi)
			{
				for(int i=gapLengthList.get(gi); i<gapLengthList.get(gi+1); ++i) System.arraycopy(parallel.get(gapLengthList.get(gi)), 0, changedScore[i], 0, changedScore[i].length);
			}
			
			for(int gapLengthLower=0; gapLengthLower<hyperParameter.gapLowerUpper; ++gapLengthLower)
				for(int noteLengthLower=0; noteLengthLower<hyperParameter.noteLowerUpper; ++noteLengthLower)
			{
				int gl=gapLengthLower, nl=noteLengthLower;
				int sumScore=currentScore.values().stream().mapToInt(score->score[gl][nl]).sum();
				if(sumScore>bestScore)
				{
					bestScore=sumScore;
					bestParam.clear();
				}
				if(sumScore==bestScore)
				{
					bestParam.add(new Parameter(threshold, noteLengthLower, gapLengthLower));
				}
			}
		}
		
		Parameter parameter=bestParam.getFirst();
		for(Parameter th: bestParam)
		{
			if(th.amplitude>parameter.amplitude || 
				th.amplitude==parameter.amplitude && (th.noteLengthLower>parameter.noteLengthLower ||
				th.noteLengthLower==parameter.noteLengthLower && th.gapLengthLower>parameter.gapLengthLower))
				parameter=th;
		}
		return parameter;
	}
	
	public static HashMap<Sequence, ArrayList<int[]>> boundaryDetection(Collection<Sequence> sequence, Parameter parameter, HyperParameter hyperParameter, Config config) throws IOException, UnsupportedAudioFileException, NotConvergedException
	{
		HashMap<Sequence, double[]> spectrogramAmplitude=spectrogramAmplitude(sequence, hyperParameter, config);
		HashMap<Sequence, ArrayList<int[]>> noteList=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence)
		{
			ArrayList<int[]> soundInterval=soundInterval(spectrogramAmplitude.get(seq), parameter.amplitude).stream().collect(Collectors.toCollection(ArrayList::new));
			removeShortGap(soundInterval, parameter.gapLengthLower);
			removeShortNote(soundInterval, parameter.noteLengthLower);
			ArrayList<int[]> note=soundInterval.stream()
					.map(si->new int[]{si[0], si[1]})
					.collect(Collectors.toCollection(ArrayList::new));
			noteList.put(seq, note);
		}
		return noteList;
	}
	
	public static HashMap<Sequence, ArrayList<double[]>> averageOutput(HashMap<Sequence, ArrayList<int[]>> soundInterval, HashMap<Sequence, float[]> output, STFTParam stftParam)
	{
		HashMap<Sequence, ArrayList<double[]>> average=new HashMap<>(output.size()*4/3);
		for(Sequence seq: output.keySet())
		{
			ArrayList<double[]> averageList=new ArrayList<>(soundInterval.get(seq).size());
			average.put(seq, averageList);
			int height=stftParam.spectrogramLength(seq.getLength());
			int width=output.get(seq).length/height;
			for(int[] si: soundInterval.get(seq))
			{
				double[] ave=new double[width];
				averageList.add(ave);
				for(int y=si[0]; y<si[0]+si[1]; ++y) for(int x=0; x<width; ++x) ave[x]+=output.get(seq)[y*width+x];
				for(int x=0; x<width; ++x) ave[x]/=si[1];
			}
		}
		return average;
	}
	
	public static class HyperParameter
	{
		private STFTParam stftParam;
		private int dpssParam, freqOffset, freqLength, gapLowerUpper, noteLowerUpper;
		public HyperParameter(STFTParam stftParam, int dpssParam, int freqOffset, int freqLength, int gapLowerUpper,
				int noteLowerUpper) {
			this.stftParam = stftParam;
			this.dpssParam = dpssParam;
			this.freqOffset = freqOffset;
			this.freqLength = freqLength;
			this.gapLowerUpper = gapLowerUpper;
			this.noteLowerUpper = noteLowerUpper;
		}
	}
	
	public static class Parameter
	{
		private double amplitude;
		private int noteLengthLower, gapLengthLower;
		
		public Parameter(double amplitude, int noteLengthLower, int gapLengthLower)
		{
			this.amplitude = amplitude;
			this.noteLengthLower = noteLengthLower;
			this.gapLengthLower = gapLengthLower;
		}

		public double getAmplitude() {
			return amplitude;
		}

		public int getNoteLengthLower() {
			return noteLengthLower;
		}

		public int getGapLengthLower() {
			return gapLengthLower;
		}
		
		public void writeXml(Path file) throws IOException
		{
			Element rootEl=XmlUtils.rootElement("ThresholdingParameter");
			XmlUtils.addChild(rootEl, "Amplitude", amplitude);
			XmlUtils.addChild(rootEl, "NoteLengthLower", noteLengthLower);
			XmlUtils.addChild(rootEl, "GapLengthLower", gapLengthLower);
			XmlUtils.write(rootEl.getOwnerDocument(), file);
		}
		
		public static Parameter parseXml(Path file) throws IOException
		{
			Node rootChild=XmlUtils.parse(file).getFirstChild().getFirstChild();
			double amplitude=XmlUtils.nodeDouble(rootChild);
			rootChild=rootChild.getNextSibling();
			int noteLengthLower=XmlUtils.nodeInt(rootChild);
			rootChild=rootChild.getNextSibling();
			int gapLengthLower=XmlUtils.nodeInt(rootChild);
			return new Parameter(amplitude, noteLengthLower, gapLengthLower);
		}
	}
	
	public static class Config
	{
		private Executor executor;
		private Path waveFileDir;
		private boolean verbose;

		public Config(Executor executor, Path waveFileDir, boolean verbose) {
			this.executor = executor;
			this.waveFileDir = waveFileDir;
			this.verbose = verbose;
		}
	}
}
