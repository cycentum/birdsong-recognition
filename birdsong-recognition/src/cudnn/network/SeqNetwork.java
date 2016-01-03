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
package cudnn.network;

import java.util.ArrayList;

import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.FloatType;
import cudnn.IntType;
import cudnn.Pointer;
import cudnn.layer.ConvLayer;
import cudnn.layer.DataLayer;
import cudnn.layer.Layer;
import cudnn.layer.NonDataLayer;
import cudnn.layer.ParamLayer;
import cudnn.layer.PoolLayer;
import cudnn.layer.SeqSoftmaxConvLayer;
import cudnn.learner.Model;
import jcuda.jcublas.JCublas;

public class SeqNetwork implements Model
{
	private FloatType floatType;
	private ArrayList<Layer> layer;
	private DataLayer dataLayer;
	private SeqSoftmaxConvLayer softMaxLayer;
	private int batchSize, workspaceSize;
	private Pointer workspaceDev;
	private ArrayList<Pointer> paramDev, gradDev;
	private ArrayList<Integer> paramSize;
	private Pointer dataDev, labelDev;
	private int labelShiftUpperH, labelShiftUpperW, labelShiftY, labelShiftX;
	private IntType labelType;
	private byte[] labelB;
	private int[] labelI;
	private ArrayList<float[]> dataF;
	private ArrayList<double[]> dataD;
	
	public SeqNetwork(FloatType floatType, ArrayList<Layer> layer, int batchSize, Cudnn cudnn, IntType labelType, int labelShiftUpperH, int labelShiftUpperW) throws CudnnException
	{
		this.layer=layer;
		this.floatType=floatType;
		this.batchSize=batchSize;
		this.labelType=labelType;
		
		for(Layer la: layer)
		{
			if(la instanceof DataLayer) dataLayer=(DataLayer)la;
			if(la instanceof SeqSoftmaxConvLayer) softMaxLayer=(SeqSoftmaxConvLayer)la;
			la.init(floatType, cudnn, batchSize);
			
			if(la instanceof ParamLayer) ((ParamLayer)la).newWeightBias(floatType);
		}
		this.labelShiftUpperH=labelShiftUpperH;
		this.labelShiftUpperW=labelShiftUpperW;
		if(this.labelShiftUpperH<=0)
		{
			this.labelShiftUpperH=1;
			Layer la=softMaxLayer;
			while(la instanceof NonDataLayer)
			{
				if(la instanceof PoolLayer) this.labelShiftUpperH*=((PoolLayer) la).getStride();
				la=((NonDataLayer) la).getLower();
			}
		}
		if(this.labelShiftUpperW<=0)
		{
			this.labelShiftUpperW=1;
			Layer la=softMaxLayer;
			while(la instanceof NonDataLayer)
			{
				if(la instanceof PoolLayer) this.labelShiftUpperW*=((PoolLayer) la).getStride();
				la=((NonDataLayer) la).getLower();
			}
		}
		softMaxLayer.setLabelShiftUpper(this.labelShiftUpperH, this.labelShiftUpperW);
	}
	
	public int getLabelShiftUpperH() {
		return labelShiftUpperH;
	}
	
	public int getLabelShiftUpperW() {
		return labelShiftUpperW;
	}
	
	/**
	 * @return sum, not average
	 */
	public double compError(CudaDriver driver) throws CudaException
	{
		return softMaxLayer.compError(driver, labelDev, labelType, floatType);
	}
	public void initError() throws CudaException
	{
		softMaxLayer.initError(floatType);
	}

	public void init(CudaDriver driver, Cudnn cudnn) throws CudnnException, CudaException
	{
		workspaceSize=0;
		for(Layer la: layer) if(la instanceof ConvLayer)
		{
			ConvLayer conv=(ConvLayer)la;
			int sizeInBytes=conv.compWorkspaceSize(cudnn);
			if(sizeInBytes>workspaceSize) workspaceSize=sizeInBytes;
		}
		
		if (workspaceSize > 0)
		{
			workspaceDev=Cuda.malloc(workspaceSize);
		}
		else workspaceDev=Pointer.createNull();

		for(Layer la: layer)
		{
			if(la instanceof ConvLayer) ((ConvLayer)la).setWodkspace(workspaceDev, workspaceSize);
		}
	}

	public void cudaMalloc() throws CudaException
	{
		for(Layer la: layer) la.cudaMalloc(floatType, batchSize);

		dataDev=Cuda.malloc(floatType.getBytes()*batchSize*dataLayer.sizeCHW());
		dataLayer.setDataDev(dataDev);
		labelDev=Cuda.malloc(labelType.getBytes()*batchSize*softMaxLayer.getLabelWidth()*softMaxLayer.getLabelHeight());
		
		paramDev=new ArrayList<>();
		gradDev=new ArrayList<>();
		paramSize=new ArrayList<>();
		for(Layer la: layer) if(la instanceof ParamLayer)
		{
			ParamLayer pl=(ParamLayer)la;
			paramDev.add(pl.getWeightDev());
			gradDev.add(pl.getGradWeightDev());
			paramSize.add(pl.getWeightSize());
			paramDev.add(pl.getBiasDev());
			gradDev.add(pl.getGradBiasDev());
			paramSize.add(pl.getBiasSize());
		}
	}
	
	public void copyParamToDev() throws CudaException
	{
		for(Layer la: layer) if(la instanceof ParamLayer) ((ParamLayer)la).copyParamToDev(floatType);		
	}
	
	public void copyParamFromDev() throws CudaException
	{
		for(Layer la: layer) if(la instanceof ParamLayer) ((ParamLayer)la).copyParamFromDev(floatType);
	}
	
	public void copyDataToDeviceHeightDisp() throws CudaException
	{
		if(floatType==FloatType.SINGLE)
		{
			int by=floatType.getBytes();
			for(int b=0; b<batchSize; ++b)
			{
				Cuda.memcpyAsyncHostToDevice(dataDev.withByteOffset(b*dataLayer.sizeHW()*by), Pointer.to(dataF.get(b)).withByteOffset(labelShiftY*dataLayer.getWidth()*by), dataLayer.sizeHW()*by);
			}
		}
		else
		{
			int by=floatType.getBytes();
			for(int b=0; b<batchSize; ++b)
			{
				Cuda.memcpyAsyncHostToDevice(dataDev.withByteOffset(b*dataLayer.sizeHW()*by), Pointer.to(dataD.get(b)).withByteOffset(labelShiftY*dataLayer.getWidth()*by), dataLayer.sizeHW()*by);
			}
		}
	}
	
	public void copyLabelToDevice(byte[] label) throws CudaException
	{
		if(label.length!=batchSize*softMaxLayer.getLabelHeight()*softMaxLayer.getLabelWidth()) System.err.println("label.length="+label.length+" != expected="+batchSize*softMaxLayer.getLabelHeight()*softMaxLayer.getLabelWidth());
		Cuda.memcpyAsyncHostToDevice(labelDev, label);
		this.labelB=label;
	}
	public void copyLabelToDevice(int[] label) throws CudaException
	{
		if(label.length!=batchSize*softMaxLayer.getLabelHeight()*softMaxLayer.getLabelWidth()) System.err.println("label.length="+label.length+" != expected="+batchSize*softMaxLayer.getLabelHeight()*softMaxLayer.getLabelWidth());
		Cuda.memcpyAsyncHostToDevice(labelDev, label);
		this.labelI=label;
	}
	
	public void destroy(Cudnn cudnn) throws CudaException, CudnnException
	{
//		Cuda.setDevice(m_gpuid);
		for(Layer la: layer) la.destroy(cudnn);
		if(workspaceDev.getJnaPointer()!=null) Cuda.free(workspaceDev);
		
		if(dataDev!=null) Cuda.free(dataDev);
		if(labelDev!=null) Cuda.free(labelDev);
	}

	public void compForward(CudaDriver driver, Cudnn cudnn) throws CudaException, CudnnException
	{
		for(Layer la: layer) if(la instanceof NonDataLayer)
		{
			NonDataLayer ndl=(NonDataLayer)la;
			ndl.compForward(floatType, cudnn, batchSize, driver);
		}
	}

	public void compBackward(CudaDriver driver, Cudnn cudnn) throws CudaException, CudnnException
	{
		{
			Cuda.memcpyAsyncDeviceToDevice(softMaxLayer.getDerActDev(), softMaxLayer.getValueDev(), floatType.getBytes()* softMaxLayer.sizeCHW()*batchSize);
	
			int blockWidth=128;
			softMaxLayer.backwardCost(driver, floatType, labelDev, batchSize, blockWidth, labelType);
			
			int size;
			if(labelType==IntType.BYTE) size=softMaxLayer.countValidLabelSize(labelB, batchSize);
			else size=softMaxLayer.countValidLabelSize(labelI, batchSize);
			
			if(floatType==FloatType.SINGLE)
			{
				float scalVal=(float)(1d/size);
				JCublas.cublasSscal(softMaxLayer.sizeCHW() * batchSize, scalVal, softMaxLayer.getDerActDev(), 1);
			}
			else
			{
				double scalVal=1d/size;
				JCublas.cublasDscal(softMaxLayer.sizeCHW() * batchSize, scalVal, softMaxLayer.getDerActDev(), 1);
			}
		}
		for(int li=layer.size()-1; li>=0; --li)
		{
			Layer la=layer.get(li);
			if(la instanceof NonDataLayer) ((NonDataLayer)la).compBackward(floatType, cudnn, batchSize, driver);
		}
	}

	public ArrayList<Pointer> getParamDev() {
		return paramDev;
	}

	public ArrayList<Pointer> getGradDev() {
		return gradDev;
	}

	public ArrayList<Integer> getParamSize() {
		return paramSize;
	}

	@Override
	public void compGradient(CudaDriver driver, Cudnn cudnn) throws CudaException, CudnnException
	{
		compForward(driver, cudnn);
		compBackward(driver, cudnn);
	}

	public void setDataF(ArrayList<float[]> data) {
		this.dataF = data;
	}
	
	public void setDataD(ArrayList<double[]> data) {
		this.dataD = data;
	}
	
	public void setLabelShift(int labelShiftY, int labelShiftX)
	{
		this.labelShiftY = labelShiftY;
		this.labelShiftX = labelShiftX;
		softMaxLayer.setLabelShift(labelShiftY, labelShiftX);
	}
	
	public int labelIndex(int batch, int y, int x, int singleY, int singleX)
	{
		return softMaxLayer.labelIndex(batch, y, x, singleY, singleX);
	}
	
	public ArrayList<Layer> getLayer() {
		return layer;
	}
	
	/**
	 * label is valid if and only if label>=0
	 * @return int[labelShiftUpperH*labelShiftUpperW]
	 */
	public int[] countValidLabelSize(byte[] label)
	{
		int[] validSize=new int[labelShiftUpperH*labelShiftUpperW];
		for(int shiftY=0; shiftY<labelShiftUpperH; ++shiftY) for(int shiftX=0; shiftX<labelShiftUpperW; ++shiftX)
		{
			setLabelShift(shiftY, shiftX);
			for(int b=0; b<batchSize; ++b)
				for(int y=0; y<softMaxLayer.getHeight(); ++y) for(int x=0; x<softMaxLayer.getWidth(); ++x)
					for(int sy=0; sy<softMaxLayer.getSingleHeight(); ++sy) for(int sx=0; sx<softMaxLayer.getSingleWidth(); ++sx)
						if(label[labelIndex(b, y, x, sy, sx)]>=0) ++validSize[shiftY*labelShiftUpperW+shiftX];
		}
		return validSize;
	}
}
