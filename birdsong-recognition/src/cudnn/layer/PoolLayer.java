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

import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.CudnnLibrary;
import cudnn.FloatType;
import cudnn.Pointer;
import cudnn.PoolingMode;

/**
 * A class for a pooling layer.
 * {@link #init(FloatType, Cudnn, int)} and {@link #cudaMalloc(FloatType, int)} must be called before computation.
 * {@link #destroy(Cudnn)} must be called before termination of the program.
 * **Dev means a pointer in a GPU.
 * @author koumura
 *
 */
public class PoolLayer implements NonDataLayer
{
	private int numChannel, height, width;
	private int filterHeight, filterWidth;
	private int stride;
	private Layer lower;

	private CudnnLibrary.cudnnTensorDescriptor_t tensorDesc;
	private CudnnLibrary.cudnnPoolingDescriptor_t poolDesc;
	
	private Pointer valueDev;
	private Pointer derValueDev;
	
	private PoolingMode poolingMode;
	
	public PoolLayer(int filterHeight, int filterWidth, int stride, PoolingMode poolingMode, Layer lower)
	{
		this.lower=lower;
		this.numChannel=lower.getNumChannel();
		this.height=(lower.getHeight()-(filterHeight-stride))/stride;
		this.width=(lower.getWidth()-(filterWidth-stride))/stride;
		this.filterHeight=filterHeight;
		this.filterWidth=filterWidth;
		this.stride=stride;
		this.lower=lower;
		this.poolingMode=poolingMode;
	}
	public PoolLayer(int filterSize, int stride, PoolingMode poolingMode, Layer lower)
	{
		this(filterSize, filterSize, stride, poolingMode, lower);
	}

	public void destroy(Cudnn cudnn) throws CudnnException, CudaException
	{
		getTensorDesc().destroy(cudnn);
		poolDesc.destroy(cudnn);
		
//		if(getValueDev()!=null) Cuda.free(getValueDev());
//		if(derValueDev!=null) Cuda.free(derValueDev);
		if(valueDev!=null) valueDev.free();
		if(derValueDev!=null) derValueDev.free();
	}

	public void cudaMalloc(FloatType floatType, int batchSize) throws CudaException
	{
		valueDev=Cuda.malloc(floatType.getBytes() * batchSize * sizeCHW());
		derValueDev=Cuda.malloc(  floatType.getBytes() * batchSize * sizeCHW());
	}
	
	public void cudaMallocCombLower(FloatType floatType, int batchSize, Pointer valueDev, Pointer derValueDev) throws CudaException
	{
		this.valueDev=valueDev;
		this.derValueDev=derValueDev;
	}
	
	public void init(FloatType floatType, Cudnn cudnn, int batchSize) throws CudnnException
	{
		tensorDesc=cudnn.createTensorDescriptor();
		cudnn.initTensorDesc(floatType, this, batchSize);
		
		poolDesc=cudnn.createPoolingDescriptor();
		poolDesc.setPooling2dDescriptor(cudnn, poolingMode.getValue(),filterHeight, filterWidth,
				0, 0,
				getStride(), getStride());
	}
	
	public void compForward(FloatType floatType, Cudnn cudnn, int batchSize, CudaDriver driver) throws CudnnException, CudaException
	{
		cudnn.poolingForward(poolDesc, Cudnn.const1(floatType), lower.getTensorDesc(), lower.getValueDev().getJnaPointer(), Cudnn.const0(floatType), getTensorDesc(), getValueDev().getJnaPointer());		
	}
	
	public void compBackward(FloatType floatType, Cudnn cudnn, int batchSize, CudaDriver driver) throws CudnnException
	{
		if(lower instanceof NonDataLayer)
		{
			cudnn.poolingBackward(poolDesc, Cudnn.const1(floatType),
					getTensorDesc(), valueDev.getJnaPointer(), getTensorDesc(), derValueDev.getJnaPointer(),
					lower.getTensorDesc(), lower.getValueDev().getJnaPointer(), Cudnn.const0(floatType), lower.getTensorDesc(), ((NonDataLayer) lower).getDerValueDev().getJnaPointer());
		}
	}
	
	public Pointer derValueDev(){return derValueDev;}
	
	public int getStride() {
		return stride;
	}

	public int getNumChannel() {
		return numChannel;
	}

	public int getHeight() {
		return height;
	}

	public int getWidth() {
		return width;
	}

	public CudnnLibrary.cudnnTensorDescriptor_t getTensorDesc() {
		return tensorDesc;
	}

	public void setTensorDesc(CudnnLibrary.cudnnTensorDescriptor_t tensorDesc) {
		this.tensorDesc = tensorDesc;
	}

	public Pointer getValueDev() {
		return valueDev;
	}

	@Override
	public Pointer getDerValueDev() {return derValueDev;}

	@Override
	public Layer getLower() {return lower;}
}
