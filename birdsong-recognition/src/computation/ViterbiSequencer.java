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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;

import utils.ArrayUtils;

public class ViterbiSequencer
{
	private StateTransition transition;
	private double[] currentScore, nextScore;
	private ArrayList<int[]> srcState;
	
	public ViterbiSequencer(StateTransition transition)
	{
		this.transition = transition;
		this.currentScore = new double[transition.numState()];
		this.nextScore = new double[transition.numState()];
		this.srcState = new ArrayList<>();
	}

	public ArrayList<Integer> compLabelSequence(ArrayList<double[]> observationProbability)
	{
		for(int t=srcState.size(); t<observationProbability.size()-1; ++t) srcState.add(new int[transition.numState()]);
		
		Arrays.fill(currentScore, Double.NEGATIVE_INFINITY);
		for(int d=0; d<transition.dstLabel(transition.headState()).length; ++d)
		{
			int dstState=transition.dstState(transition.headState())[d];
			int dstLabel=transition.dstLabel(transition.headState())[d];
			double transitionProb=transition.transitionProbability(transition.headState())[d];
			double observationProb=observationProbability.get(0)[dstLabel];
			currentScore[dstState]=Math.log(transitionProb)+Math.log(observationProb);
//			currentScore[dstState]=transitionProb+observationProb;
		}
		
		for(int t=0; t<observationProbability.size()-1; ++t)
		{
			Arrays.fill(nextScore, Double.NEGATIVE_INFINITY);
			for(int s=0; s<transition.numState(); ++s)
			{
				for(int d=0; d<transition.dstLabel(s).length; ++d)
				{
					int dstState=transition.dstState(s)[d];
					int dstLabel=transition.dstLabel(s)[d];
					double transitionProb=transition.transitionProbability(s)[d];
					double observationProb=observationProbability.get(t+1)[dstLabel];
					double ns=currentScore[s]+Math.log(transitionProb)+Math.log(observationProb);
//					double ns=currentScore[s]+transitionProb+observationProb;
					if(ns>=nextScore[dstState])
					{
						nextScore[dstState]=ns;
						srcState.get(t)[dstState]=s;
					}
				}
			}
			{
				double[] tmp=currentScore;
				currentScore=nextScore;
				nextScore=tmp;
			}
		}
		
		int currentState=-1;
		for(int s: transition.tailState()) if(currentState==-1||currentScore[s]>currentScore[currentState]) currentState=s;
		ArrayList<Integer> labelSequence=new ArrayList<Integer>(observationProbability.size());
		for(int t=observationProbability.size()-2; t>=0; --t)
		{
			int prevState=srcState.get(t)[currentState];
			int srcLabel=-1;
			for(int d=0; d<transition.dstState(prevState).length; ++d) if(transition.dstState(prevState)[d]==currentState)
			{
				srcLabel=transition.dstLabel(prevState)[d];
				break;
			}
			labelSequence.add(srcLabel);
			currentState=prevState;
		}
		{
			int prevState=transition.headState();
			int srcLabel=-1;
			for(int d=0; d<transition.dstState(prevState).length; ++d) if(transition.dstState(prevState)[d]==currentState)
			{
				srcLabel=transition.dstLabel(prevState)[d];
				break;
			}
			labelSequence.add(srcLabel);
			
		}
		Collections.reverse(labelSequence);
		
		return labelSequence;
	}
	
	public static interface StateTransition
	{
		int[] dstState(int srcState);
		int[] dstLabel(int srcState);
		double[] transitionProbability(int srcState);
		int headState();
		int numState();
		int numLabel();
		int[] tailState();
	}
	/**
	 * label={0, 1, ..., numLabel-1, headState}
	 */
	public static class FirstOrderTransition implements StateTransition
	{
		private int[] dstLabel;
		private ArrayList<double[]> transitionProbability;
		
		/**
		 * @param transitionProbability get(srcState)[dstLabel]
		 */
		public FirstOrderTransition(ArrayList<double[]> transitionProbability)
		{
			this.dstLabel = ArrayUtils.createSequence(0, transitionProbability.size()-1);
			this.transitionProbability = transitionProbability;
		}

		@Override
		public int[] dstState(int srcState) {return dstLabel;}

		@Override
		public int[] dstLabel(int srcState) {return dstLabel;}

		@Override
		public double[] transitionProbability(int srcState) {return transitionProbability.get(srcState);}

		@Override
		public int headState() {return numLabel();}

		@Override
		public int numState() {return numLabel();}

		@Override
		public int numLabel() {return dstLabel.length;}
		
		@Override
		public int[] tailState(){return dstLabel;}
	}
	public static class SecondOrderTransition implements StateTransition
	{
		private int[] dstLabel, tailState;
		private double[][][] transitionProbability;
		private int numState;
		private double[] evenTransitionProbability;
		private ArrayList<int[]> dstState;
		
		public SecondOrderTransition(double[][][] transitionProbability)
		{
			this.dstLabel = ArrayUtils.createSequence(0, transitionProbability.length);
			this.transitionProbability = transitionProbability;
			
			numState=(numLabel()+1)*numLabel()+1;
			evenTransitionProbability=ArrayUtils.createFilled(numLabel(), 1.0/numLabel());
			
			dstState=new ArrayList<>(numState());
			for(int label0=0; label0<numLabel()+1; ++label0) for(int label1=0; label1<numLabel(); ++label1)
			{
				int dstLabel0=label1;
				int[] ds=new int[numLabel()];
				dstState.add(ds);
				for(int dstLabel1=0; dstLabel1<numLabel(); ++dstLabel1)
				{
					ds[dstLabel1]=state(dstLabel0, dstLabel1);
				}
			}
			{
				int label1=numLabel();
				int dstLabel0=label1;
				int[] ds=new int[numLabel()];
				dstState.add(ds);
				for(int dstLabel1=0; dstLabel1<numLabel(); ++dstLabel1)
				{
					ds[dstLabel1]=state(dstLabel0, dstLabel1);
				}
			}
			
			tailState=ArrayUtils.createSequence(0, numState);
		}

		@Override
		public int[] dstState(int srcState) {return dstState.get(srcState);}

		@Override
		public int[] dstLabel(int srcState) {return dstLabel;}

		@Override
		public double[] transitionProbability(int srcState)
		{
			int label0=label0(srcState);
			if(label0==numLabel()) return evenTransitionProbability;  //label0==head
			int label1=label1(srcState);
			return transitionProbability[label0][label1];
		}

		@Override
		public int headState() {return numState-1;}

		@Override
		public int numState() {return numState;}

		@Override
		public int numLabel() {return dstLabel.length;}
		
		@Override
		public int[] tailState(){return tailState;}
		
		private int label0(int state)
		{
			if(state==headState()) return numLabel();
			return state/numLabel();
		}
		
		private int label1(int state)
		{
			if(state==headState()) return numLabel();
			return state%numLabel();
		}
		
		private int state(int label0, int label1)
		{
			return label0*numLabel()+label1;
		}
	}
	public static class SecondOrderLowerTransition implements StateTransition
	{
		private ArrayList<int[]> dstState, dstLabel;
		private ArrayList<double[]> transitionProbability;
		private int numUpperSoundLabel, numLowerSoundLabel, numFullLabel, numState;
		private ArrayList<Integer> stateLabel0, stateLabel1, stateLowerLabel;
		private int[] tailState;

		public SecondOrderLowerTransition(double[][][] upperTransitionProbability, int numUpperSoundLabel, int numLowerSoundLabel)
		{
			this.numUpperSoundLabel=numUpperSoundLabel;
			this.numLowerSoundLabel=numLowerSoundLabel;
			this.numState=(numUpperSoundLabel+1)*numUpperSoundLabel*(numLowerSoundLabel+1)+1;
			numFullLabel=numUpperSoundLabel*numLowerSoundLabel+1;
			stateLabel0=new ArrayList<Integer>(numState);
			stateLabel1=new ArrayList<Integer>(numState);
			stateLowerLabel=new ArrayList<Integer>(numState);
			dstState=new ArrayList<>(numState);
			dstLabel=new ArrayList<>(numState);
			transitionProbability=new ArrayList<>(numState);
			
			int state=0;
			for(int ul0=0; ul0<numUpperSoundLabel+1; ++ul0)
				for(int ul1=0; ul1<numUpperSoundLabel; ++ul1)
					for(int ll=0; ll<numLowerSoundLabel+1; ++ll)
			{
				if(state!=state(ul0, ul1, ll)) System.err.println(state+" != "+state(ul0, ul1, ll));
				
				stateLabel0.add(ul0);
				stateLabel1.add(ul1);
				stateLowerLabel.add(ll);
				int[] ds=null, dl=null;
				double[] tp=null;
				if(ll<numLowerSoundLabel-1)
				{
					ds=new int[]{state, state+1};
					dl=new int[]{fullLabel(ul1, ll), fullLabel(ul1, ll+1)};
					tp=new double[]{0.5, 0.5};
				}
				else if(ll==numLowerSoundLabel-1)  //last lower
				{
					//allow no silence
					int numDst=numUpperSoundLabel+2;
					ds=new int[numDst];
					dl=new int[numDst];
					tp=new double[numDst];
					for(int d=0; d<numUpperSoundLabel; ++d)
					{
						ds[d]=state(ul1, d, 0);
						dl[d]=fullLabel(d, 0);
						if(ul0<numUpperSoundLabel) tp[d]=upperTransitionProbability[ul0][ul1][d]*(numDst-2)/numDst;
						else tp[d]=1.0/numDst;
					}
					{
						int d=numUpperSoundLabel;
						ds[d]=state;
						dl[d]=fullLabel(ul1, ll);
						tp[d]=1.0/numDst;
					}
					{
						int d=numUpperSoundLabel+1;
						ds[d]=state+1;
						dl[d]=silenceFullLabel();
						tp[d]=1.0/numDst;
					}
					
					//forbid no silence
					/*ds=new int[]{state, state+1};
					dl=new int[]{fullLabel(ul1, ll), silenceFullLabel()};
					tp=new double[]{0.5, 0.5};*/
				}
				else if(ll==numLowerSoundLabel)  //silence
				{
					int numDst=numUpperSoundLabel+1;
					ds=new int[numDst];
					dl=new int[numDst];
					tp=new double[numDst];
					for(int d=0; d<numUpperSoundLabel; ++d)
					{
						ds[d]=state(ul1, d, 0);
						dl[d]=fullLabel(d, 0);
						if(ul0<numUpperSoundLabel) tp[d]=upperTransitionProbability[ul0][ul1][d]*(numDst-1)/numDst;
						else tp[d]=1.0/numDst;
					}
					{
						int d=numUpperSoundLabel;
						ds[d]=state;
						dl[d]=silenceFullLabel();
						tp[d]=1.0/numDst;
					}
				}
				dstState.add(ds);
				dstLabel.add(dl);
				transitionProbability.add(tp);
				++state;
			}
			//head state
			{
				int numDst=numUpperSoundLabel+1;
				int[] ds=new int[numDst];
				int[] dl=new int[numDst];
				double[] tp=ArrayUtils.createFilled(numDst, 1.0/numDst);
				for(int d=0; d<numUpperSoundLabel; ++d)
				{
					ds[d]=state(numUpperSoundLabel, d, 0);
					dl[d]=fullLabel(d, 0);
				}
				{
					int d=numUpperSoundLabel;
					ds[d]=state;
					dl[d]=silenceFullLabel();
				}
				dstState.add(ds);
				dstLabel.add(dl);
				transitionProbability.add(tp);
				
				stateLabel0.add(numUpperSoundLabel);
				stateLabel1.add(numUpperSoundLabel);
				stateLowerLabel.add(numLowerSoundLabel);
			}
			
			LinkedList<Integer> tailState=new LinkedList<>();
			for(int ul0=0; ul0<numUpperSoundLabel+1; ++ul0) for(int ul1=0; ul1<numUpperSoundLabel; ++ul1)
			{
				tailState.add(state(ul0, ul1, numLowerSoundLabel-1));
				tailState.add(state(ul0, ul1, numLowerSoundLabel));
			}
			tailState.add(headState());
			this.tailState=tailState.stream().mapToInt(x->x).toArray();
		}
		
		@Override
		public int[] dstState(int srcState){return dstState.get(srcState);}

		@Override
		public int[] dstLabel(int srcState) {return dstLabel.get(srcState);}

		@Override
		public double[] transitionProbability(int srcState)
		{
			return transitionProbability.get(srcState);
//			return ArrayUtils.create(dstState.get(srcState).length, 1d);
		}

		@Override
		public int headState() {return numState()-1;}

		@Override
		public int numLabel() {return numFullLabel;}
		
		@Override
		public int numState() {return numState;}
		
		@Override
		public int[] tailState(){return tailState;}
		
		private int silenceFullLabel(){return numLabel()-1;}
		
		public int fullLabel(int upperLabel, int lowerLabel)
		{
			if(upperLabel==numUpperSoundLabel||lowerLabel==numLowerSoundLabel) return silenceFullLabel();
			return upperLabel*numLowerSoundLabel+lowerLabel;
		}
		
		public int state(int label0, int label1, int lowerLabel)
		{
			if(label1==numUpperSoundLabel) return headState();
			return (label0*numUpperSoundLabel+label1)*(numLowerSoundLabel+1)+lowerLabel;
		}
		
		public int upperLabel(int fullLabel)
		{
			if(fullLabel==silenceFullLabel()) return numUpperSoundLabel;
			return fullLabel/numLowerSoundLabel;
		}
		
		public int lowerLabel(int fullLabel)
		{
			if(fullLabel==silenceFullLabel()) return numLowerSoundLabel;
			return fullLabel%numLowerSoundLabel;
		}
	}
}
