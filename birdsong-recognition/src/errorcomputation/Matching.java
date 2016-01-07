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
package errorcomputation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import computation.STFTParam;
import computation.Sequence;
import computation.Sequence.LabelList;
import computation.Sequence.Note;
import utils.ArrayUtils;
import utils.CollectionUtils;

/**
 * A class to compute matching errors.
 * @author koumura
 *
 */
public class Matching
{
	private static int[] indexArray(ArrayList<int[]> interval, int sequenceLength)
	{
		int[] array=ArrayUtils.createFilled(sequenceLength, -1);
		for(int i=0; i<interval.size(); ++i) Arrays.fill(array, interval.get(i)[0], interval.get(i)[0]+interval.get(i)[1], i);
		return array;
	}
	
	private static ArrayList<int[]> remainingInterval(ArrayList<int[]> interval, List<int[]> matching, int matchingIndex)
	{
		ArrayList<int[]> intervalNext=new ArrayList<int[]>();
		HashSet<Integer> index=new HashSet<Integer>();
		for(int[] m: matching) index.add(m[matchingIndex]);
		for(int i=0; i<interval.size(); ++i) if(!index.contains(i)) intervalNext.add(interval.get(i));
		return intervalNext;
	}
	
	private static LinkedList<Integer> matchPosition(int[] interval0, int[] interval1)
	{
		int begin=Math.max(interval0[0], interval1[0]);
		int end=Math.min(interval0[0]+interval0[1], interval1[0]+interval1[1]);
		LinkedList<Integer> matchPosition=new LinkedList<Integer>();
		for(int i=begin; i<end; ++i) matchPosition.add(i);
		return matchPosition;
	}
	private static LinkedList<Integer> matchPosition(ArrayList<int[]> correctInterval, ArrayList<int[]> answerInterval, int[] matching)
	{
		LinkedList<Integer> matchPosition=matchPosition(correctInterval.get(matching[0]), answerInterval.get(matching[1]));
		if(matchPosition.size()!=matching[2]) System.err.println("matchPosition.size()="+matchPosition.size()+" != matching.get2()="+matching[2]);
		return matchPosition;
	}
	
	/**
	 * @return Matched Positions
	 */
	private static LinkedList<Integer> matchSingleLabel(ArrayList<int[]> correctInterval, ArrayList<int[]> answerInterval, int sequenceLength)
	{
//		int[] correctIndexArray=indexArray(correctInterval, sequenceLength);
		int[] answerIndexArray=indexArray(answerInterval, sequenceLength);
		
		ArrayList<ArrayList<int[]>> overlapLength=new ArrayList<>(answerInterval.size());
		for(int a=0; a<answerInterval.size(); ++a) overlapLength.add(new ArrayList<int[]>());
		for(int c=0; c<correctInterval.size(); ++c)
		{
			HashMap<Integer, Integer> overlap=new HashMap<>();
			for(int t=correctInterval.get(c)[0]; t<correctInterval.get(c)[0]+correctInterval.get(c)[1]; ++t)
			{
				int ai=answerIndexArray[t];
				if(ai>=0)
				{
					overlap.putIfAbsent(ai, 0);
					overlap.put(ai, overlap.get(ai)+1);
				}
			}
			
			if(overlap.size()==0) continue;
			ArrayList<int[]> list=new ArrayList<>(overlap.size());
			for(Integer key: overlap.keySet()) list.add(new int[]{key, overlap.get(key)});
			list.sort((o1, o2)->-(o1[1]-o2[1]));
			for(int i=0; i<list.size()&&list.get(i)[1]==list.get(0)[1]; ++i) overlapLength.get(list.get(i)[0]).add(new int[]{c, list.get(i)[1]});
		}
		
		LinkedList<int[]> candidate=new LinkedList<int[]>();  //int[](c, a, overlapLength)
		for(int a=0; a<answerInterval.size(); ++a)
		{
			ArrayList<int[]> ol=overlapLength.get(a);
			ol.sort((o1, o2)->-(o1[1]-o2[1]));
			for(int i=0; i<ol.size()&&ol.get(i)[1]==ol.get(0)[1]; ++i) candidate.add(new int[]{ol.get(i)[0], a, ol.get(i)[1]});
		}
		
		int[] correctCandidateSize=new int[correctInterval.size()];
		int[] answerCandidateSize=new int[answerInterval.size()];
		for(int[] m: candidate)
		{
			++correctCandidateSize[m[0]];
			++answerCandidateSize[m[1]];
		}
		
		LinkedList<int[]> matching=new LinkedList<int[]>();
		for(int[] m: candidate)
		{
			if(correctCandidateSize[m[0]]==1&&answerCandidateSize[m[1]]==1) matching.add(m);
		}
		candidate.removeAll(matching);
		
		LinkedList<Integer> matchPosition=new LinkedList<Integer>();
		for(int[] m: matching)
		{
			LinkedList<Integer> mp=matchPosition(correctInterval, answerInterval, m);
			if(mp.size()!=m[2]) System.err.println("mp.size()="+mp.size()+" != m.get2()="+m[2]);
			matchPosition.addAll(mp);
		}
		if(candidate.size()==0) return matchPosition;
		
		if(matching.size()==0)
		{
			for(int[] fix: candidate)
			{
				ArrayList<int[]> matchingNext=new ArrayList<int[]>(matching);
				matchingNext.add(fix);
				
				ArrayList<int[]> correctIntervalNext=remainingInterval(correctInterval, matchingNext, 0);
				ArrayList<int[]> answerIntervalNext=remainingInterval(answerInterval, matchingNext, 1);
				LinkedList<Integer> nextMatchPosition=matchSingleLabel(correctIntervalNext, answerIntervalNext, sequenceLength);
				nextMatchPosition.addAll(matchPosition(correctInterval, answerInterval, fix));
				if(nextMatchPosition.size()>matchPosition.size()) matchPosition=nextMatchPosition;
			}
			return matchPosition;
		}
		
		ArrayList<int[]> correctIntervalNext=remainingInterval(correctInterval, matching, 0);
		ArrayList<int[]> answerIntervalNext=remainingInterval(answerInterval, matching, 1);
		matchPosition.addAll(matchSingleLabel(correctIntervalNext, answerIntervalNext, sequenceLength));
		return matchPosition;
	}
	
	private static HashMap<Integer, ArrayList<int[]>> labelMap(ArrayList<int[]> labelInterval)
	{
		HashMap<Integer, ArrayList<int[]>> labelMap=new HashMap<>();
		for(int[] li: labelInterval)
		{
			labelMap.putIfAbsent(li[2], new ArrayList<>());
			labelMap.get(li[2]).add(new int[]{li[0], li[1]});
		}
		return labelMap;
	}
	
	/**
	 * @param correctSpecLabelInterval int[](start, length, label)
	 * @param answerSpecLabelInterval int[](start, length, label)
	 * @return Matched Positions
	 */
	private static LinkedList<Integer> compMatching(ArrayList<int[]> correctSpecLabelInterval, ArrayList<int[]> answerSpecLabelInterval, int sequenceSpecLength)
	{
		int silentLabel=0;
		for(int[] li: correctSpecLabelInterval) if(li[2]>=silentLabel) silentLabel=li[2]+1;
		for(int[] li: answerSpecLabelInterval) if(li[2]>=silentLabel) silentLabel=li[2]+1;
		
		HashMap<Integer, ArrayList<int[]>> correctLabelMap=labelMap(correctSpecLabelInterval);
		HashMap<Integer, ArrayList<int[]>> answerLabelMap=labelMap(answerSpecLabelInterval);
		
		LinkedList<Integer> matchPosition=new LinkedList<Integer>();
		for(int la: correctLabelMap.keySet()) if(answerLabelMap.containsKey(la))
		{
			matchPosition.addAll(matchSingleLabel(new ArrayList<int[]>(correctLabelMap.get(la)), new ArrayList<int[]>(answerLabelMap.get(la)), sequenceSpecLength));
		}
		
		int[] correctIndexArray=indexArray(correctSpecLabelInterval, sequenceSpecLength);
		int[] answerIndexArray=indexArray(answerSpecLabelInterval, sequenceSpecLength);
		for(int i=0; i<sequenceSpecLength; ++i) if(correctIndexArray[i]==-1&&answerIndexArray[i]==-1) matchPosition.add(i);
		return matchPosition;
	}
	
	/**
	 * @param correctSpecLabelInterval int[](start, length, label)
	 * @param answerSpecLabelInterval int[](start, length, label)
	 * @return Total length of matched intervals.
	 */
	public static int computeMatchedLength(ArrayList<int[]> correctSpecLabelInterval, ArrayList<int[]> answerSpecLabelInterval, int sequenceSpecLength)
	{
		return compMatching(correctSpecLabelInterval, answerSpecLabelInterval, sequenceSpecLength).size();
	}
	
	/**
	 * @return Total length of matched intervals.
	 */
	public static int computeDistance(Sequence correctSequence, List<Note> outputNoteSequence, STFTParam stftParam, LabelList labelList)
	{
		Function<Note, int[]> conv=n->{
			int begin=stftParam.spectrogramPosition(n.getPosition());
			int length=stftParam.spectrogramPosition(n.end())-begin;
			int label=labelList.indexOf(n.getLabel());
			return new int[]{begin, length, label};
		};
		ArrayList<int[]> correctSpecLabelInterval=correctSequence.getNote().stream().map(conv).collect(CollectionUtils.arrayListCollector());
		ArrayList<int[]> outputSpecLabelInterval=outputNoteSequence.stream().map(conv).collect(CollectionUtils.arrayListCollector());
		int sequenceSpecLength=stftParam.spectrogramLength(correctSequence.getLength());
		return sequenceSpecLength-computeMatchedLength(correctSpecLabelInterval, outputSpecLabelInterval, sequenceSpecLength);
	}
	
	/**
	 * @param outputSequence Map of correct sequences to output intervals.
	 * @return Matching error.
	 */
	public static double computeError(Map<Sequence, ArrayList<Note>> outputSequence, STFTParam stftParam, LabelList labelList)
	{
		int specLength=0, distance=0;
		for(Sequence seq: outputSequence.keySet())
		{
			specLength+=stftParam.spectrogramLength(seq.getLength());
			distance+=computeDistance(seq, outputSequence.get(seq), stftParam, labelList);
		}
		return (double)distance/specLength;
	}
}
