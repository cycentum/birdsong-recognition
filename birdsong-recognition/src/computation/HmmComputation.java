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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import computation.Sequence.LabelList;
import computation.Sequence.Note;
import computation.ViterbiSequencer.SecondOrderLowerTransition;
import computation.ViterbiSequencer.SecondOrderTransition;
import utils.ArrayUtils;
import utils.CollectionUtils;
import utils.Executor;
import utils.MathUtils;

public class HmmComputation
{
	private static double[][][] transitionProbability(List<Sequence> sequence, LabelList labelList)
	{
		int numLabel=labelList.size();
		int[][][] count=new int[numLabel][numLabel][numLabel];
		for(Sequence seq: sequence)
		{
			ArrayList<Note> noteList=seq.getNote();
			for(int i=2; i<noteList.size(); ++i) ++count[labelList.indexOf(noteList.get(i-2).getLabel())][labelList.indexOf(noteList.get(i-1).getLabel())][labelList.indexOf(noteList.get(i).getLabel())];
		}
		double[][][] prob=new double[numLabel][numLabel][numLabel];
		for(int l0=0; l0<numLabel; ++l0) for(int l1=0; l1<numLabel; ++l1)
		{
			int sum=MathUtils.sum(count[l0][l1]);
			if(sum>0) for(int l2=0; l2<numLabel; ++l2) prob[l0][l1][l2]=(double)count[l0][l1][l2]/sum;
		}
		return prob;
	}
	
	private static void smoothTransitionProbability(double[][][] prob, double smoothing)
	{
		for(int l0=0; l0<prob.length; ++l0) for(int l1=0; l1<prob[l0].length; ++l1)
		{
			for(int l2=0; l2<prob[l0][l1].length; ++l2) prob[l0][l1][l2]+=smoothing;
			MathUtils.divideBySum(prob[l0][l1]);
		}
	}
	
	private static double[] fullSpecLabelFrequency(List<Sequence> sequence, int numLowerLabel, LabelList labelList, STFTParam stftParam)
	{
		int numUpperLabel=labelList.size();
		int[] count=new int[numUpperLabel*numLowerLabel+1];
		for(Sequence seq: sequence)
		{
			ArrayList<int[]> specLabelInterval=seq.getNote().stream()
					.map(n->{
						int begin=stftParam.spectrogramPosition(n.getPosition());
						int end=stftParam.spectrogramPosition(n.end());
						return new int[]{begin, end-begin, labelList.indexOf(n.getLabel())};
					}).collect(CollectionUtils.arrayListCollector());
			
			for(int[] li: specLabelInterval) for(int lower=0; lower<numLowerLabel; ++lower)
			{
				int begin=MathUtils.subIntervalBegin(li[1], numLowerLabel, lower);
				int end=MathUtils.subIntervalBegin(li[1], numLowerLabel, lower+1);
				int fullLabel=li[2]*numLowerLabel+lower;
				count[fullLabel]+=end-begin;
			}
			for(int i=0; i<specLabelInterval.size()-1; ++i)
			{
				count[count.length-1]+=specLabelInterval.get(i+1)[0]-ArrayUtils.sum01(specLabelInterval.get(i));
			}
			count[count.length-1]+=specLabelInterval.get(0)[0];
			int spectrogramLength=stftParam.spectrogramLength(seq.getLength());
			count[count.length-1]+=spectrogramLength-ArrayUtils.sum01(specLabelInterval.get(specLabelInterval.size()-1));
		}
		return MathUtils.divideBySum(count);
	}

	private static double[] upperLabelFrequency(List<Sequence> specLabelInterval, LabelList labelList)
	{
		int numUpperLabel=labelList.size();
		int[] count=new int[numUpperLabel];
		for(Sequence seq: specLabelInterval)
		{
			for(Note li: seq.getNote())
			{
				++count[labelList.indexOf(li.getLabel())];
			}
		}
		return MathUtils.divideBySum(count);
	}
	
	private static void posteriorToObservationProb(ArrayList<double[]> posterior, double[] fullLabelFrequency)
	{
		for(double[] p: posterior) for(int la=0; la<p.length; ++la) p[la]/=fullLabelFrequency[la];
	}
	
	public static double[][][] transitionProbability(List<Sequence> trainingSequence, LabelList labelList, HyperParam hyperParam)
	{
		double[][][] transitionProbability=transitionProbability(trainingSequence, labelList);
		if(hyperParam.smoothingConstant>0) smoothTransitionProbability(transitionProbability, hyperParam.smoothingConstant);
		return transitionProbability;
	}
	
	public static HashMap<Sequence, ArrayList<double[]>> segmentedPosteriorToObservationProb(HashMap<Sequence, ArrayList<double[]>> posteriorProb, List<Sequence> trainingSequence, LabelList labelList, HyperParam hyperParam)
	{
		HashMap<Sequence, ArrayList<double[]>> observationProb=posteriorProb;
		if(hyperParam.posteriorToObservationProb)
		{
			double[] fullLabelFreq=upperLabelFrequency(trainingSequence, labelList);
			for(ArrayList<double[]> op: observationProb.values()) posteriorToObservationProb(op, fullLabelFreq);
		}
		return observationProb;
	}
	
	public static HashMap<Sequence, ArrayList<Note>> globalSequencing(HashMap<Sequence, ArrayList<double[]>> observationProb, double[][][] transitionProb, LabelList labelList, HashMap<Sequence, ArrayList<int[]>> soundInterval, HyperParam hyperParam, Config config) throws InterruptedException, ExecutionException
	{
		ArrayList<Sequence> sequence=observationProb.keySet().stream().collect(CollectionUtils.arrayListCollector());
		SecondOrderTransition transition=new SecondOrderTransition(transitionProb);
		ArrayList<UpperSequencerJob> job=new ArrayList<>(config.executor.getNumThread());
		for(int th=0; th<config.executor.getNumThread(); ++th)
		{
			int subIntervalBegin=MathUtils.subIntervalBegin(observationProb.size(), config.executor.getNumThread(), th);
			int subIntervalEnd=MathUtils.subIntervalBegin(observationProb.size(), config.executor.getNumThread(), th+1);
			HashMap<Sequence, ArrayList<double[]>> op=new HashMap<>((subIntervalEnd-subIntervalBegin)*4/3);
			for(int i=subIntervalBegin; i<subIntervalEnd; ++i) op.put(sequence.get(i), observationProb.get(sequence.get(i)));
			job.add(new UpperSequencerJob(transition, op));
		}

		List<Future<HashMap<Sequence, int[]>>> future=config.executor.get().invokeAll(job);
		
		HashMap<Sequence, ArrayList<Note>> viterbiLabelInterval=new HashMap<>(observationProb.size()*4/3);
		for(Future<HashMap<Sequence, int[]>> f: future)
		{	
			HashMap<Sequence, int[]> answerMap=f.get();
			for(Sequence seq: answerMap.keySet())
			{
				int[] answer=answerMap.get(seq);
				ArrayList<Note> answerInterval=new ArrayList<Note>(answer.length);
				viterbiLabelInterval.put(seq, answerInterval);
				for(int i=0; i<soundInterval.get(seq).size(); ++i)
				{
					int[] si=soundInterval.get(seq).get(i);
					int start=hyperParam.stftParam.wavePosition(si[0]);
					int end=hyperParam.stftParam.wavePosition(si[0]+si[1]);
					String label=labelList.get(answer[i]);
					answerInterval.add(new Note(start, end-start, label));
				}
			}	
		}
		return viterbiLabelInterval;
	}

	public static HashMap<Sequence, ArrayList<double[]>> continuousPosteriorToObservationProb(HashMap<Sequence, float[]> posteriorProb, List<Sequence> trainingSequence, LabelList labelList, HyperParam hyperParam, Config config)
	{
		int posteriorWidth=config.posteriorWidth;
		HashMap<Sequence, ArrayList<double[]>> observationProb=new HashMap<>(posteriorProb.size()*4/3);
		for(Sequence seq: posteriorProb.keySet())
		{
			int height=posteriorProb.get(seq).length/posteriorWidth;
			ArrayList<double[]> prob=new ArrayList<>(height);
			observationProb.put(seq, prob);
			int i=0;
			for(int h=0; h<height; ++h)
			{
				double[] array=new double[posteriorWidth];
				prob.add(array);
				for(int w=0; w<posteriorWidth; ++w) array[w]=posteriorProb.get(seq)[i++];
			}
		}
		if(hyperParam.posteriorToObservationProb)
		{
			double[] fullLabelFreq=fullSpecLabelFrequency(trainingSequence, hyperParam.numSubLabel, labelList, hyperParam.stftParam);
			for(ArrayList<double[]> op: observationProb.values()) posteriorToObservationProb(op, fullLabelFreq);
		}
		return observationProb;
	}
	
	public static HashMap<Sequence, ArrayList<Note>> globalSequencingWithBoundaryDetection(HashMap<Sequence, ArrayList<double[]>> observationProb, double[][][] transitionProb, LabelList labelList, HyperParam hyperParam, Config config) throws InterruptedException, ExecutionException
	{
		ArrayList<Sequence> sequence=observationProb.keySet().stream().collect(CollectionUtils.arrayListCollector());
		SecondOrderLowerTransition transition=new SecondOrderLowerTransition(transitionProb, labelList.size(), hyperParam.numSubLabel);
		ArrayList<SequencerJob> job=new ArrayList<>(config.executor.getNumThread());
		for(int th=0; th<config.executor.getNumThread(); ++th)
		{
			int subIntervalBegin=MathUtils.subIntervalBegin(observationProb.size(), config.executor.getNumThread(), th);
			int subIntervalEnd=MathUtils.subIntervalBegin(observationProb.size(), config.executor.getNumThread(), th+1);
			HashMap<Sequence, ArrayList<double[]>> op=new HashMap<>((subIntervalEnd-subIntervalBegin)*4/3);
			for(int i=subIntervalBegin; i<subIntervalEnd; ++i) op.put(sequence.get(i), observationProb.get(sequence.get(i)));
			job.add(new SequencerJob(transition, op));
		}
		
		List<Future<Object>> future=config.executor.get().invokeAll(job);
		for(Future<Object> f: future) f.get();
		
		HashMap<Sequence, ArrayList<Note>> viterbiLabelInterval=new HashMap<>(observationProb.size()*4/3);
		for(SequencerJob j: job) for(Sequence seq: j.observationProb.keySet())
		{	
			ArrayList<Note> noteList=viterbiLabelInterval(j.answer.get(seq), labelList.size(), hyperParam.numSubLabel).stream()
					.map(li->{
						int start=hyperParam.stftParam.wavePosition(li[0]);
						int end=hyperParam.stftParam.wavePosition(li[0]+li[1]);
						String label=labelList.get(li[2]);
						return new Note(start, end-start, label);
					}).collect(Collectors.toCollection(ArrayList::new));
			viterbiLabelInterval.put(seq, noteList);
		}
		return viterbiLabelInterval;
	}
	
	public static class Config
	{
		private Executor executor;
		private int posteriorWidth;

		public Config(Executor executor, int posteriorWidth) {
			this.executor = executor;
			this.posteriorWidth=posteriorWidth;
		}
	}
	
	public static class HyperParam
	{
		private double smoothingConstant;
		private boolean posteriorToObservationProb;
		private int numSubLabel;
		private STFTParam stftParam;
		
		public HyperParam(double smoothingConstant, boolean posteriorToObservationProb, int numSubLabel,
				STFTParam stftParam) {
			this.smoothingConstant = smoothingConstant;
			this.posteriorToObservationProb = posteriorToObservationProb;
			this.numSubLabel = numSubLabel;
			this.stftParam = stftParam;
		}
	}
	
	public static class SequencerJob implements Callable<Object>
	{
		private SecondOrderLowerTransition transition;
		private ViterbiSequencer sequencer;
		private HashMap<Sequence, ArrayList<double[]>> observationProb;
		private HashMap<Sequence, ArrayList<int[]>> answer;
		
		public SequencerJob(SecondOrderLowerTransition transition, HashMap<Sequence, ArrayList<double[]>> observationProb)
		{
			this.observationProb=observationProb;
			this.transition=transition;
			sequencer=new ViterbiSequencer(transition);
		}

		@Override
		public Object call()
		{
			answer=new HashMap<>(observationProb.size()*4/3);
			for(Sequence seq: observationProb.keySet())
			{
				ArrayList<double[]> op=observationProb.get(seq);
				ArrayList<Integer> a=sequencer.compLabelSequence(op);
				ArrayList<int[]> answerArray=new ArrayList<>(a.size());
				answer.put(seq, answerArray);
				for(int t=0; t<a.size(); ++t)
				{
					answerArray.add(new int[]{transition.upperLabel(a.get(t)), transition.lowerLabel(a.get(t))});
				}
			}
			return null;
		}
	}
	
	private static class UpperSequencerJob implements Callable<HashMap<Sequence, int[]>>
	{
		private ViterbiSequencer sequencer;
		private HashMap<Sequence, ArrayList<double[]>> observationProb;
		
		private UpperSequencerJob(SecondOrderTransition transition, HashMap<Sequence, ArrayList<double[]>> observationProb)
		{
			this.observationProb=observationProb;
			sequencer=new ViterbiSequencer(transition);
		}

		@Override
		public HashMap<Sequence, int[]> call()
		{
			HashMap<Sequence, int[]> answer=new HashMap<Sequence, int[]>(observationProb.size()*4/3);
			for(Sequence seq: observationProb.keySet())
			{
				ArrayList<Integer> a=sequencer.compLabelSequence(observationProb.get(seq));
				int[] array=a.stream().mapToInt(x->x).toArray();
				answer.put(seq, array);
			}
			return answer;
		}
	}

	private static ArrayList<int[]> viterbiLabelInterval(ArrayList<int[]> viterbiSequence, int numUpperLabel, int numLowerLabel)
	{
		ArrayList<int[]> classified=new ArrayList<int[]>();
		for(int x=0; x<viterbiSequence.size(); ++x)
		{
			int[] vp=viterbiSequence.get(x);
			if(vp[0]<numUpperLabel&&vp[1]==0)
			{
				if(x>0 && viterbiSequence.get(x-1)[1]==numLowerLabel-1) classified.get(classified.size()-1)[1]=(x);
				if(classified.size()==0 || classified.get(classified.size()-1)[1]!=-1) classified.add(new int[]{x, -1, vp[0]});
			}
			else if((vp[0]==numUpperLabel || vp[1]==numLowerLabel) && classified.size()>0&&classified.get(classified.size()-1)[1]==-1) classified.get(classified.size()-1)[1]=(x);
		}
		if(classified.size()>0&&classified.get(classified.size()-1)[1]==-1) classified.get(classified.size()-1)[1]=(viterbiSequence.size());
		for(int[] li: classified) li[1]=(li[1]-li[0]);
		
		return classified;
	}
}
