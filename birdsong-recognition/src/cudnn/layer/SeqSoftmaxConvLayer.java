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
package cudnn.layer;

import cudnn.ActivationMode;
import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.CudnnLibrary;
import cudnn.FloatType;
import cudnn.IntType;
import cudnn.Pointer;

public class SeqSoftmaxConvLayer extends ConvLayer implements OutputLayer
{
	private Pointer sizeDev, errorDev;
	private int labelShiftUpperH, labelShiftUpperW, labelShiftY, labelShiftX, labelWidth, labelHeight, batchSize;
	private float outputLowerForErrorF;
	private double outputLowerForErrorD;
	private float[] errorF;
	private double[] errorD;
	
	private int singleSize, singleHeight, singleWidth;
	private CudnnLibrary.cudnnTensorDescriptor_t softmaxTensorDesc;
	
	public SeqSoftmaxConvLayer(int singleSize, int singleHeight, int singleWidth, int filterHeight, int filterWidth, Layer lower, ConvLayer.BackwardAlgorithm backwardAlgo)
	{
		super(singleSize*singleHeight*singleWidth, filterHeight, filterWidth, ActivationMode.IDENT, lower, backwardAlgo);
		this.singleSize=singleSize;
		this.singleHeight=singleHeight;
		this.singleWidth=singleWidth;
	}

	@Override
	public void init(FloatType floatType, Cudnn cudnn, int batchSize) throws CudnnException
	{
		super.init(floatType, cudnn, batchSize);
		softmaxTensorDesc=cudnn.createTensorDescriptor();
		softmaxTensorDesc.setTensor4dDescriptor(cudnn, CudnnLibrary.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, floatType.getDataTypeValue(), batchSize, singleSize, singleHeight*getHeight(), singleWidth*getWidth());
	}

	@Override
	public void compForward(FloatType floatType, Cudnn cudnn, int batchSize, CudaDriver driver) throws CudnnException, CudaException
	{
		compForwardAct(floatType, cudnn);
		
		cudnn.softmaxForward(CudnnLibrary.cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE, CudnnLibrary.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_CHANNEL,
				Cudnn.const1(floatType), softmaxTensorDesc, getActDev().getJnaPointer(), Cudnn.const0(floatType), softmaxTensorDesc, getValueDev().getJnaPointer());
	}

	public int finalSizeHW(){return sizeHW()*singleHeight*singleWidth;}
	
	@Override
	public void cudaMalloc(FloatType floatType, int batchSize) throws CudaException
	{
		super.cudaMalloc(floatType, batchSize);
		sizeDev=Cuda.malloc(11*Integer.BYTES);
		this.batchSize=batchSize;
	}
	
	public void initError(FloatType floatType) throws CudaException
	{
		errorDev=Cuda.malloc(batchSize*finalSizeHW()*floatType.getBytes());
		if(floatType==FloatType.SINGLE)
		{
			errorF=new float[batchSize*finalSizeHW()];
		}
		else
		{
			errorD=new double[batchSize*finalSizeHW()];
		}
	}
	
	@Override
	public void destroy(Cudnn cudnn) throws CudaException, CudnnException
	{
		super.destroy(cudnn);
		if(errorDev!=null) errorDev.free();
		if(sizeDev!=null) sizeDev.free();
	}
	
	@Override
	public void copyParamToDev(FloatType floatType) throws CudaException
	{
		super.copyParamToDev(floatType);
		if(sizeDev!=null)
		{
			int[] size=new int[11];
			size[0]=batchSize;
			size[1]=getNumChannel();
			size[2]=getHeight();
			size[3]=getWidth();
			size[4]=getSingleSize();
			size[5]=singleHeight;
			size[6]=singleWidth;
			size[7]=labelHeight;
			size[8]=labelWidth;
			size[9]=labelShiftUpperH;
			size[10]=labelShiftUpperW;
			Cuda.memcpyAsyncHostToDevice(sizeDev, size);
		}
	}
	
	/**
	 * @return sum, not average
	 */
	public double compError(CudaDriver driver, Pointer labelDev, IntType labelType, FloatType floatType) throws CudaException
	{
		jcuda.Pointer pLower;
		if(floatType==FloatType.SINGLE) pLower=jcuda.Pointer.to(new float[]{outputLowerForErrorF});
		else pLower=jcuda.Pointer.to(new double[]{outputLowerForErrorD});
		
		int blockWidth=128;
		driver.call("SeqSoftmaxConvError"+(floatType==FloatType.SINGLE?"Float":"Double")+(labelType==IntType.BYTE?"Char":""), (int)Math.ceil((double)batchSize*finalSizeHW()/blockWidth), 1, 1, blockWidth, 1, 1, 0, jcuda.Pointer.to(
				jcuda.Pointer.to(errorDev),
				jcuda.Pointer.to(getValueDev()),
				jcuda.Pointer.to(labelDev),
				jcuda.Pointer.to(sizeDev),
				jcuda.Pointer.to(new int[]{labelShiftY}),
				jcuda.Pointer.to(new int[]{labelShiftX}),
				pLower
				));
		
		double sum=0;
		if(floatType==FloatType.SINGLE)
		{
			Cuda.memcpyDeviceToHost(Pointer.to(errorF), errorDev, errorF.length*floatType.getBytes());
			for(float e: errorF) sum+=e;
		}
		else
		{
			Cuda.memcpyDeviceToHost(Pointer.to(errorD), errorDev, errorD.length*floatType.getBytes());
			for(double e: errorD) sum+=e;
		}
		return sum;
	}
	
	public void backwardCost(CudaDriver driver, FloatType floatType, Pointer labelDev, int batchSize, int blockWidth, IntType labelType) throws CudaException
	{
		driver.call("SeqSoftmaxConvBackward"+(floatType==FloatType.SINGLE?"Float":"Double")+(labelType==IntType.BYTE?"Char":""), (int)Math.ceil((double)batchSize*finalSizeHW()/blockWidth), 1, 1, blockWidth, 1, 1, 0, jcuda.Pointer.to(
				jcuda.Pointer.to(labelDev),
				jcuda.Pointer.to(getDerActDev()),
				jcuda.Pointer.to(sizeDev),
				jcuda.Pointer.to(new int[]{labelShiftY}),
				jcuda.Pointer.to(new int[]{labelShiftX})
				));
	}

	public int getLabelHeight(){return labelHeight;}
	
	public int getLabelWidth(){return labelWidth;}

	public void setLabelShiftUpper(int labelShiftUpperH, int labelShiftUpperW) {
		this.labelShiftUpperH = labelShiftUpperH;
		this.labelHeight=getHeight()*labelShiftUpperH-1+singleHeight;
		this.labelShiftUpperW = labelShiftUpperW;
		this.labelWidth=getWidth()*labelShiftUpperW-1+singleWidth;
	}
	
	public void setLabelShift(int labelShiftY, int labelShiftX) {
		this.labelShiftY = labelShiftY;
		this.labelShiftX = labelShiftX;
	}

	public int countValidLabelSize(byte[] label, int batchSize)
	{
		int size=0;
		for(int b=0; b<batchSize; ++b)
			for(int y=0; y<getHeight(); ++y) for(int x=0; x<getWidth(); ++x)
				for(int sy=0; sy<singleHeight; ++sy) for(int sx=0; sx<singleWidth; ++sx)	
		{
			byte la=label[labelIndex(b, y, x, sy, sx)];
			if(la>=0) ++size;
		}
		return size;
	}
	public int countValidLabelSize(int[] label, int batchSize)
	{
		int size=0;
		for(int b=0; b<batchSize; ++b)
			for(int y=0; y<getHeight(); ++y) for(int x=0; x<getWidth(); ++x)
				for(int sy=0; sy<singleHeight; ++sy) for(int sx=0; sx<singleWidth; ++sx)	
		{
			int la=label[labelIndex(b, y, x, sy, sx)];
			if(la>=0) ++size;
		}
		return size;
	}
	
	public int labelIndex(int batch, int y, int x, int singleY, int singleX)
	{
		return batch*labelWidth*labelHeight+(y*labelShiftUpperH+labelShiftY+singleY)*getLabelWidth()+x*labelShiftUpperW+labelShiftX+singleX;
	}

	public void setOutputLowerForError(double outputLowerForError, FloatType floatType) {
		if(floatType==FloatType.SINGLE)
		{
			this.outputLowerForErrorF = (float)outputLowerForError;
		}
		else
		{
			this.outputLowerForErrorD = outputLowerForError;
		}
	}

	public int getSingleHeight() {
		return singleHeight;
	}

	public int getSingleWidth() {
		return singleWidth;
	}

	public int getSingleSize() {
		return singleSize;
	}
}
