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
package utils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.ListIterator;

import org.apache.commons.math3.random.MersenneTwister;

import computation.STFTParam;
import computation.Sequence;
import cudnn.ActivationMode;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.FloatType;
import cudnn.IntType;
import cudnn.PoolingMode;
import cudnn.layer.ConvLayer;
import cudnn.layer.DataLayer;
import cudnn.layer.Layer;
import cudnn.layer.ParamLayer;
import cudnn.layer.PoolLayer;
import cudnn.layer.SeqSoftmaxConvLayer;
import cudnn.network.SeqNetwork;

public class DnnUtils
{
	public static enum LayerType
	{
		CONV, POOLING;
	}
	
	public static int nextSmallestInputSize(List<Pair<LayerType, Integer>> filterSize, int inputSize)
	{
		int scale=1, addition=0;
		ListIterator<Pair<LayerType, Integer>> iterator=filterSize.listIterator(filterSize.size());
		while(iterator.hasPrevious())
		{
			Pair<LayerType, Integer> fs=iterator.previous();
			if(fs.get0()==LayerType.CONV)
			{
				addition+=fs.get1()-1;
			}
			else if(fs.get0()==LayerType.POOLING)
			{
				addition*=fs.get1();
				scale*=fs.get1();
			}
		}
		if(inputSize<=addition) return addition+scale;
		return scale*MathUtils.ceil(inputSize-addition, scale)+addition;
	}
	
	public static int sumStride(List<Pair<LayerType, Integer>> filterSize)
	{
		int stride=1;
		for(Pair<LayerType, Integer> fs: filterSize)
		{
			if(fs.get0()==LayerType.POOLING) stride*=fs.get1();
		}
		return stride;
	}
	
	public static ArrayList<Pair<LayerType, Integer>> filterSizeList525242()
	{
		ArrayList<Pair<LayerType, Integer>> filterSizeList=new ArrayList<>();
		filterSizeList.add(new Pair<>(LayerType.CONV, 5));
		filterSizeList.add(new Pair<>(LayerType.POOLING, 2));
		filterSizeList.add(new Pair<>(LayerType.CONV, 5));
		filterSizeList.add(new Pair<>(LayerType.POOLING, 2));
		filterSizeList.add(new Pair<>(LayerType.CONV, 4));
		filterSizeList.add(new Pair<>(LayerType.POOLING, 2));
		return filterSizeList;
	}
	
	public static SeqNetwork netinnetNetwork(int dataLayerHeight, int inputWidth, int softMaxSize, int batchSize, Cudnn cudnn, CudaDriver driver, int localInputHeight, int finalInputHeight, ConvLayer.BackwardAlgorithm backwardAlgo) throws CudnnException, CudaException
	{
		int numConvChannel=16;
		int fullConnectionSize=240;
		DataLayer dataLayer=new DataLayer(1, dataLayerHeight, inputWidth);
		ConvLayer conv1=new ConvLayer(numConvChannel, 5, 5, ActivationMode.RELU, dataLayer, backwardAlgo);
		ConvLayer conv11=new ConvLayer(numConvChannel, 1, 1, ActivationMode.RELU, conv1, backwardAlgo);
		PoolLayer pool1=new PoolLayer(2, 2, PoolingMode.MAX, conv11);
		ConvLayer conv2=new ConvLayer(numConvChannel, 5, 5, ActivationMode.RELU, pool1, backwardAlgo);
		ConvLayer conv21=new ConvLayer(numConvChannel, 1, 1, ActivationMode.RELU, conv2, backwardAlgo);
		PoolLayer pool2=new PoolLayer(2, 2, PoolingMode.MAX, conv21);
		ConvLayer conv3=new ConvLayer(numConvChannel, 4, 4, ActivationMode.RELU, pool2, backwardAlgo);
		ConvLayer conv31=new ConvLayer(numConvChannel, 1, 1, ActivationMode.RELU, conv3, backwardAlgo);
		PoolLayer pool3=new PoolLayer(2, 2, PoolingMode.MAX, conv31);
		int fullFilterHeight=(((localInputHeight-conv1.getFilterHeight()+1)/pool1.getStride()-conv2.getFilterHeight()+1)/pool2.getStride()-conv3.getFilterHeight()+1)/pool3.getStride();
		ConvLayer full=new ConvLayer(fullConnectionSize, fullFilterHeight, 11, ActivationMode.RELU, pool3, backwardAlgo);
		ConvLayer globalFull=null;
		if(finalInputHeight>localInputHeight)
		{
			int globalFullFilterHeight=new STFTParam(localInputHeight, pool1.getStride()*pool2.getStride()*pool3.getStride()).spectrogramLength(finalInputHeight);
			globalFull=new ConvLayer(fullConnectionSize, globalFullFilterHeight, 1, ActivationMode.RELU, full, backwardAlgo);
		}
		SeqSoftmaxConvLayer smLayer=new SeqSoftmaxConvLayer(softMaxSize, 1, 1, 1, 1, finalInputHeight>localInputHeight?globalFull:full, backwardAlgo);
		
		ArrayList<Layer> layer=new ArrayList<>();
		layer.add(dataLayer);
		layer.add(conv1);
		layer.add(conv11);
		layer.add(pool1);
		layer.add(conv2);
		layer.add(conv21);
		layer.add(pool2);
		layer.add(conv3);
		layer.add(conv31);
		layer.add(pool3);
		layer.add(full);
		if(finalInputHeight>localInputHeight) layer.add(globalFull);
		layer.add(smLayer);
		
		SeqNetwork network=new SeqNetwork(FloatType.SINGLE, layer, batchSize, cudnn, IntType.BYTE, -1, 1);
		network.init(driver, cudnn);
		
		return network;
	}

	public static void initParam(List<Layer> layer, MersenneTwister random)
	{
		for(Layer la: layer) if(la instanceof ParamLayer)
		{
			int size=0;
			ConvLayer cl=(ConvLayer)la;
			size=cl.getLower().getNumChannel()*cl.getFilterHeight()*cl.getFilterHeight();
			double scale=Math.sqrt(2.0/size);
			for(int i=0; i<cl.getWeightF().length; ++i) cl.getWeightF()[i]=(float)(random.nextGaussian()*scale);
			for(int i=0; i<cl.getBiasF().length; ++i) cl.getBiasF()[i]=0;
		}
	}
	
	public static ArrayList<float[]> copyParamFromLayer(ArrayList<Layer> layer)
	{
		ArrayList<float[]> param=new ArrayList<float[]>();
		for(Layer la: layer) if(la instanceof ParamLayer)
		{
			ParamLayer pl=(ParamLayer)la;
			param.add(Arrays.copyOf(pl.getWeightF(), pl.getWeightSize()));
			param.add(Arrays.copyOf(pl.getBiasF(), pl.getBiasSize()));
		}
		return param;
	}
	
	public static void copyParamToLayer(ArrayList<float[]> src, ArrayList<Layer> dst)
	{
		{
			if((int)dst.stream().filter(la->la instanceof ParamLayer).count()*2!=src.size())
			{
				System.err.println("Parameters were not copied because of the inconsistent network structure.");
				return;
			}
			int paramIndex=0;
			for(Layer la: dst) if(la instanceof ParamLayer)
			{
				ParamLayer pl=(ParamLayer)la;
				if(pl.getWeightSize()!=src.get(paramIndex++).length||pl.getBiasSize()!=src.get(paramIndex++).length)
				{
					System.err.println("Parameters were not copied because of the inconsistent network structure.");
					return;
				}
			}
		}
		int paramIndex=0;
		for(Layer la: dst) if(la instanceof ParamLayer)
		{
			ParamLayer pl=(ParamLayer)la;
			System.arraycopy(src.get(paramIndex++), 0, pl.getWeightF(), 0, pl.getWeightSize());
			System.arraycopy(src.get(paramIndex++), 0, pl.getBiasF(), 0, pl.getBiasSize());
		}
	}
	
	public static void saveParam(ArrayList<float[]> param, Path file) throws IOException
	{
		int numFloat=param.stream().mapToInt(array->array.length).sum();
		ByteBuffer buf=ByteBuffer.allocate((param.size()+1)*Integer.BYTES+numFloat*Float.BYTES);
		buf.putInt(param.size());
		for(float[] p: param)
		{
			buf.putInt(p.length);
			for(float value: p) buf.putFloat(value);
		}
		Files.write(file, buf.array());
	}

	public static ArrayList<float[]> loadParam(Path file) throws IOException
	{
		ByteBuffer buf=ByteBuffer.wrap(Files.readAllBytes(file));
		int size=buf.getInt();
		ArrayList<float[]> param=new ArrayList<float[]>(size);
		for(int i=0; i<size; ++i)
		{
			int len=buf.getInt();
			float[] array=new float[len];
			param.add(array);
			for(int a=0; a<len; ++a) array[a]=buf.getFloat();
		}
		return param;
	}
	
	public static void saveOutput(HashMap<Sequence, float[]> output, Path file) throws IOException
	{
		int byteSize=Integer.BYTES;
		for(Sequence seq: output.keySet())
		{
			byteSize+=seq.byteSize();
			byteSize+=Integer.BYTES+output.get(seq).length*Float.BYTES;
		}
		ByteBuffer buf=ByteBuffer.allocate(byteSize);
		buf.putInt(output.size());
		for(Sequence seq: output.keySet())
		{
			seq.serialize(buf);
			buf.putInt(output.get(seq).length);
			for(float value: output.get(seq)) buf.putFloat(value);
		}
		Files.write(file, buf.array());
	}
	
	public static HashMap<Sequence, float[]> loadOutput(Path file, Collection<Sequence> sequence) throws IOException
	{
		HashMap<Sequence, Sequence> sequenceMap=new HashMap<>(sequence.size()*4/3);
		for(Sequence s: sequence) sequenceMap.put(s, s);
		
		ByteBuffer buf=ByteBuffer.wrap(Files.readAllBytes(file));
		int size=buf.getInt();
		HashMap<Sequence, float[]> output=new HashMap<Sequence, float[]>(size*4/3);
		for(int i=0; i<size; ++i)
		{
			Sequence seq=Sequence.deserialize(buf, sequenceMap);
			int len=buf.getInt();
			float[] value=new float[len];
			for(int v=0; v<len; ++v) value[v]=buf.getFloat();
			output.put(seq, value);
		}
		return output;
	}
}
