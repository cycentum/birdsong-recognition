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
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.math3.random.MersenneTwister;
import org.w3c.dom.Element;
import org.w3c.dom.Node;

import utils.Pair;
import utils.RandomUtils;
import utils.XmlUtils;

public class Sequence
{
	private String waveFileName;
	private int position, length;
	private ArrayList<Note> note;
	
	private Sequence(String waveFileName, int position, int length, ArrayList<Note> note)
	{
		this.waveFileName = waveFileName;
		this.position = position;
		this.length = length;
		this.note = note;
	}

	public String getWaveFileName() {
		return waveFileName;
	}

	public int getPosition() {
		return position;
	}

	public int getLength() {
		return length;
	}

	public ArrayList<Note> getNote() {
		return note;
	}
	
	public static ArrayList<Sequence> readXml(Path file) throws IOException
	{
		Node rootChild=XmlUtils.parse(file).getFirstChild().getFirstChild();
		int numSequence=XmlUtils.nodeInt(rootChild);
		ArrayList<Sequence> sequence=new ArrayList<Sequence>(numSequence);
		for(int s=0; s<numSequence; ++s)
		{
			rootChild=rootChild.getNextSibling();
			Node seqChild=rootChild.getFirstChild();
			String waveFileName=XmlUtils.nodeText(seqChild);
			seqChild=seqChild.getNextSibling();
			int position=XmlUtils.nodeInt(seqChild);
			seqChild=seqChild.getNextSibling();
			int length=XmlUtils.nodeInt(seqChild);
			seqChild=seqChild.getNextSibling();
			int numNote=XmlUtils.nodeInt(seqChild);
			ArrayList<Note> note=new ArrayList<>(numNote);
			sequence.add(new Sequence(waveFileName, position, length, note));
			for(int n=0; n<numNote; ++n)
			{
				seqChild=seqChild.getNextSibling();
				Node noteChild=seqChild.getFirstChild();
				int pos=XmlUtils.nodeInt(noteChild);
				noteChild=noteChild.getNextSibling();
				int len=XmlUtils.nodeInt(noteChild);
				noteChild=noteChild.getNextSibling();
				String label=XmlUtils.nodeText(noteChild);
				note.add(new Note(pos, len, label));
			}
		}
		return sequence;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + length;
		result = prime * result + position;
		result = prime * result + ((waveFileName == null) ? 0 : waveFileName.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Sequence other = (Sequence) obj;
		if (length != other.length)
			return false;
		if (position != other.position)
			return false;
		if (waveFileName == null) {
			if (other.waveFileName != null)
				return false;
		} else if (!waveFileName.equals(other.waveFileName))
			return false;
		return true;
	}
	
	public static void writeOutputSequence(HashMap<Sequence, ArrayList<Note>> outputSequence, Path file) throws IOException
	{
		Element rootEl=XmlUtils.rootElement("SequenceNoteInterval");
		XmlUtils.addChild(rootEl, "NumSequence", outputSequence.size());
		List<Sequence> sequenceList=outputSequence.keySet().stream().collect(Collectors.toList());
		sequenceList.sort((o1, o2)->{
			int c=o1.waveFileName.compareTo(o2.waveFileName);
			if(c!=0) return c;
			c=o1.position-o2.position;
			if(c!=0) return c;
			c=o1.length-o2.length;
			return c;
		});
		for(Sequence seq: outputSequence.keySet())
		{
			Element seqEl=XmlUtils.addChild(rootEl, "Sequence", null);
			XmlUtils.addChild(seqEl, "WaveFileName", seq.getWaveFileName());
			XmlUtils.addChild(seqEl, "Position", seq.getPosition());
			XmlUtils.addChild(seqEl, "Length", seq.getLength());
			XmlUtils.addChild(seqEl, "NumNote", outputSequence.get(seq).size());
			for(Note note: outputSequence.get(seq))
			{
				Element noteEl=XmlUtils.addChild(seqEl, "Note", null);
				XmlUtils.addChild(noteEl, "Position", note.getPosition());
				XmlUtils.addChild(noteEl, "Length", note.getLength());
				XmlUtils.addChild(noteEl, "Label", note.getLabel());
			}
		}
		XmlUtils.write(rootEl.getOwnerDocument(), file);
	}

	public int byteSize()
	{
		return Integer.BYTES*4+waveFileName.getBytes().length;
	}
	
	public void serialize(ByteBuffer buf)
	{
		byte[] b=waveFileName.getBytes();
		buf.putInt(b.length);
		buf.put(b);
		buf.putInt(position);
		buf.putInt(length);
	}
	
	public static Sequence deserialize(ByteBuffer buf, HashMap<Sequence, Sequence> sequenceMap)
	{
		int byteLen=buf.getInt();
		byte[] b=new byte[byteLen];
		buf.get(b);
		int position=buf.getInt();
		int length=buf.getInt();
		Sequence seq=sequenceMap.get(new Sequence(new String(b), position, length, null));
		if(seq==null) System.err.println("Sequence not contained in given set.");
		return seq;
	}
	
	public static class Note
	{
		private int position, length;
		private String label;
		
		public Note(int position, int length, String label)
		{
			this.position = position;
			this.length = length;
			this.label = label;
		}

		public int getPosition() {
			return position;
		}

		public int getLength() {
			return length;
		}

		public String getLabel() {
			return label;
		}
		
		public int end(){return position+length;}
	}
	
	public static class WavePosition
	{
		private Path waveFile;
		private int position, length;
		
		public WavePosition(Path waveFile, int position, int length)
		{
			this.waveFile = waveFile;
			this.position = position;
			this.length = length;
		}

		public Path getWaveFile() {
			return waveFile;
		}

		public void setWaveFile(Path waveFile) {
			this.waveFile = waveFile;
		}

		public int getPosition() {
			return position;
		}

		public void setPosition(int position) {
			this.position = position;
		}

		public int getLength() {
			return length;
		}

		public void setLength(int length) {
			this.length = length;
		}
		
		public int getEnd()
		{
			return position+length;
		}
	}
	
	public static HashMap<Sequence, WavePosition> wavePositionMap(Collection<Sequence> sequence, Path waveFileDir)
	{
		HashMap<Sequence, WavePosition> wavePositionMap=new HashMap<>(sequence.size()*4/3);
		for(Sequence seq: sequence)
		{
			WavePosition wp=new WavePosition(waveFileDir.resolve(seq.waveFileName), seq.getPosition(), seq.getLength());
			wavePositionMap.put(seq, wp);
		}
		return wavePositionMap;
	}
	
	public static ArrayList<ArrayList<Sequence>> divide(List<Sequence> sequence, int numGroup, LabelList labelList, MersenneTwister random)
	{
		ArrayList<Pair<Sequence, int[]>> sequenceLabelCount=sequence.stream()
			.map(seq->{
				int[] count=new int[labelList.size()];
				for(Note n: seq.getNote()) ++count[labelList.indexOf(n.getLabel())];
				return new Pair<>(seq, count);
			}).collect(Collectors.toCollection(ArrayList::new));
		
		int numLabel=labelList.size();
		ArrayList<Pair<Integer, Integer>> labelCountAll=new ArrayList<>();  //Pair(label, count)
		for(int li=0; li<numLabel; ++li) labelCountAll.add(new Pair<>(li, 0));
		for(Pair<Sequence, int[]> count: sequenceLabelCount)
		{
			for(int li=0; li<numLabel; ++li) labelCountAll.get(li).set1(labelCountAll.get(li).get1()+count.get1()[li]);
		}
		labelCountAll.sort((o1, o2)->o1.get1()-o2.get1());
		HashMap<Sequence, Integer> sequenceGroup=new HashMap<>(sequence.size()*4/3);
		HashSet<Sequence> finishedSequence=new HashSet<>(sequence.size()*4/3);
		int currentLabelIndex=0;
		ArrayList<Pair<Integer, int[]>> groupLabelCount=new ArrayList<>(numGroup);  //Pair(group, labelCount)
		for(int g=0; g<numGroup; ++g) groupLabelCount.add(new Pair<>(g, new int[numLabel]));
		while(sequenceGroup.size()<sequence.size())
		{
			int currentLabel=labelCountAll.get(currentLabelIndex).get0();
			sequenceLabelCount=RandomUtils.permutation(sequenceLabelCount, random);
			for(Pair<Sequence, int[]> lc: sequenceLabelCount) if(!finishedSequence.contains(lc.get0()))
			{
				if(lc.get1()[currentLabel]==0) continue;
				Pair<Integer, int[]> minGroup=Collections.min(groupLabelCount, (o1, o2)->o1.get1()[currentLabel]-o2.get1()[currentLabel]);
				sequenceGroup.put(lc.get0(), minGroup.get0());
				finishedSequence.add(lc.get0());
				for(int li=0; li<numLabel; ++li) minGroup.get1()[li]+=lc.get1()[li];
			}
			++currentLabelIndex;
		}
		
		ArrayList<ArrayList<Sequence>> group=new ArrayList<ArrayList<Sequence>>(numGroup);
		for(int g=0; g<numGroup; ++g) group.add(new ArrayList<>());
		for(Sequence seq: sequence) group.get(sequenceGroup.get(seq)).add(seq);
		return group;
	}
	
	public static class LabelList
	{
		private ArrayList<String> list;
		private HashMap<String, Integer> index;
		
		private LabelList(ArrayList<String> list)
		{
			this.list = list;
			index=new HashMap<>(list.size()*4/3);
			for(int li=0; li<list.size(); ++li) index.put(list.get(li), li);
		}
		
		public String get(int index){return list.get(index);}
		
		public int indexOf(String label){return index.get(label);}
		
		public int size(){return list.size();}

		public static LabelList create(Collection<Sequence> sequence)
		{
			ArrayList<String> labelList=sequence.stream()
					.flatMap(s->s.getNote().stream())
					.map(n->n.getLabel())
					.collect(Collectors.toSet())
					.stream()
					.collect(Collectors.toCollection(ArrayList::new));
			Collections.sort(labelList);
			return new LabelList(labelList);
		}
	}
}
