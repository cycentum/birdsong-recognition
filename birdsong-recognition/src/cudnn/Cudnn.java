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
package cudnn;


import java.nio.file.Path;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByReference;
import com.sun.jna.ptr.DoubleByReference;
import com.sun.jna.ptr.FloatByReference;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

import cudnn.CudnnLibrary.cudnnConvolutionDescriptor_t;
import cudnn.CudnnLibrary.cudnnFilterDescriptor_t;
import cudnn.CudnnLibrary.cudnnTensorDescriptor_t;
import cudnn.layer.Layer;

/**
 * A class to handle the cudnn library.
 * {@link #destroty()} must be called before termination.
 * @author koumura
 *
 */
public class Cudnn
{
	CudnnLibrary.Instance library;
	CudnnLibrary.cudnnHandle_t handle;

	public static final FloatByReference FLOAT_1=new FloatByReference(1);
	public static final FloatByReference FLOAT_0=new FloatByReference(0);
	public static final DoubleByReference DOUBLE_1=new DoubleByReference(1);
	public static final DoubleByReference DOUBLE_0=new DoubleByReference(0);
	
	public static ByReference const1(FloatType floatType)
	{
		if(floatType==FloatType.SINGLE) return FLOAT_1;
		else return DOUBLE_1;
	}
	
	public static ByReference const0(FloatType floatType)
	{
		if(floatType==FloatType.SINGLE) return FLOAT_0;
		else return DOUBLE_0;
	}
	
	public Cudnn(Path libraryPath) throws CudnnException
	{
		library=new CudnnLibrary.Instance(libraryPath);
		PointerByReference handle=new PointerByReference();
		checkError(library.get().cudnnCreate(handle));
		this.handle=new CudnnLibrary.cudnnHandle_t(handle.getValue());
	}
	
	public void destroty() throws CudnnException
	{
		checkError(library.get().cudnnDestroy(handle));
	}
	
	public void initTensorDesc(FloatType floatType, Layer layer, int batchSize) throws CudnnException
	{
		layer.getTensorDesc().setTensor4dDescriptor(this, CudnnLibrary.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW, floatType.getDataTypeValue(), batchSize, layer.getNumChannel(), layer.getHeight(), layer.getWidth());
	}
	
	void checkError(int status) throws CudnnException
	{
		if(status!=CudnnLibrary.cudnnStatus_t.CUDNN_STATUS_SUCCESS)
		{
			throw new CudnnException(library.get().cudnnGetErrorString(status));
		}
	}
	
	public CudnnLibrary.cudnnTensorDescriptor_t createTensorDescriptor() throws CudnnException
	{
		PointerByReference desc=new PointerByReference();
		checkError(library.get().cudnnCreateTensorDescriptor(desc));
		return new CudnnLibrary.cudnnTensorDescriptor_t(desc.getValue());
	}
	
	public CudnnLibrary.cudnnFilterDescriptor_t createFilterDescriptor() throws CudnnException
	{
		PointerByReference desc=new PointerByReference();
		checkError(library.get().cudnnCreateFilterDescriptor(desc));
		return new CudnnLibrary.cudnnFilterDescriptor_t(desc.getValue());
	}
	
	public CudnnLibrary.cudnnConvolutionDescriptor_t createConvolutionDescriptor() throws CudnnException
	{
		PointerByReference desc=new PointerByReference();
		checkError(library.get().cudnnCreateConvolutionDescriptor(desc));
		return new CudnnLibrary.cudnnConvolutionDescriptor_t(desc.getValue());
	}

	public CudnnLibrary.cudnnPoolingDescriptor_t createPoolingDescriptor() throws CudnnException
	{
		PointerByReference desc=new PointerByReference();
		checkError(library.get().cudnnCreatePoolingDescriptor(desc));
		return new CudnnLibrary.cudnnPoolingDescriptor_t(desc.getValue());
	}

	/**
	 * Helper function to return the dimensions of the output tensor given a convolution descriptor<br>
	 * Original signature : <code>cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, int*, int*, int*, int*)</code><br>
	 * <i>native declaration : line 322</i>
	 * @throws CudnnException 
	 */
	public void getConvolution2dForwardOutputDim(CudnnLibrary.cudnnConvolutionDescriptor_t convDesc, CudnnLibrary.cudnnTensorDescriptor_t inputTensorDesc, CudnnLibrary.cudnnFilterDescriptor_t filterDesc, IntByReference n, IntByReference c, IntByReference h, IntByReference w) throws CudnnException
	{
		checkError(library.get().cudnnGetConvolution2dForwardOutputDim(convDesc, inputTensorDesc, filterDesc, n, c, h, w));
	}
	
	/**
	 * Original signature : <code>cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdPreference_t, size_t, cudnnConvolutionFwdAlgo_t*)</code><br>
	 * <i>native declaration : line 379</i>
	 * @throws CudnnException 
	 */
	public int getConvolutionForwardAlgorithm(CudnnLibrary.cudnnTensorDescriptor_t srcDesc, CudnnLibrary.cudnnFilterDescriptor_t filterDesc, CudnnLibrary.cudnnConvolutionDescriptor_t convDesc, CudnnLibrary.cudnnTensorDescriptor_t destDesc, int preference, int memoryLimitInbytes) throws CudnnException
	{
		IntByReference algo=new IntByReference();
		checkError(library.get().cudnnGetConvolutionForwardAlgorithm(this.handle, srcDesc, filterDesc, convDesc, destDesc, preference, memoryLimitInbytes, algo));
		return algo.getValue();
	}
	
	/**
	 * Helper function to return the minimum size of the workspace to be passed to the convolution given an algo<br>
	 * Original signature : <code>cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t, const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t, size_t*)</code><br>
	 * <i>native declaration : line 394</i>
	 * @throws CudnnException 
	 */
	public int getConvolutionForwardWorkspaceSize(CudnnLibrary.cudnnTensorDescriptor_t srcDesc, CudnnLibrary.cudnnFilterDescriptor_t filterDesc, CudnnLibrary.cudnnConvolutionDescriptor_t convDesc, CudnnLibrary.cudnnTensorDescriptor_t destDesc, int algo) throws CudnnException
	{
		IntByReference sizeInBytes=new IntByReference();
		checkError(library.get().cudnnGetConvolutionForwardWorkspaceSize(handle, srcDesc, filterDesc, convDesc, destDesc, algo, sizeInBytes));
		return sizeInBytes.getValue();
	}
	
	/**
	 * Function to perform the forward multiconvolution<br>
	 * Original signature : <code>cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnFilterDescriptor_t, const void*, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void*, size_t, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 407</i>
	 * @throws CudnnException 
	 */
	public void convolutionForward(ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, CudnnLibrary.cudnnFilterDescriptor_t filterDesc, Pointer filterData, CudnnLibrary.cudnnConvolutionDescriptor_t convDesc, int algo, Pointer workSpace, int workSpaceSizeInBytes, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData) throws CudnnException
	{
		checkError(library.get().cudnnConvolutionForward(handle, alpha, srcDesc, srcData, filterDesc, filterData, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, destDesc, destData));
	}
	
	/**
	 * Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc<br>
	 * Original signature : <code>cudnnStatus_t cudnnAddTensor(cudnnHandle_t, cudnnAddMode_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 229</i>
	 * @throws CudnnException 
	 */
	public void addTensor(int mode, ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t biasDesc, Pointer biasData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t srcDestDesc, Pointer srcDestData) throws CudnnException
	{
		checkError(library.get().cudnnAddTensor(handle, mode, alpha, biasDesc, biasData, beta, srcDestDesc, srcDestData));
	}

	/**
	 * Function to perform forward pooling<br>
	 * Original signature : <code>cudnnStatus_t cudnnPoolingForward(cudnnHandle_t, const cudnnPoolingDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 581</i>
	 * @throws CudnnException 
	 */
	public void poolingForward(CudnnLibrary.cudnnPoolingDescriptor_t poolingDesc, ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData) throws CudnnException
	{
		checkError(library.get().cudnnPoolingForward(handle, poolingDesc, alpha, srcDesc, srcData, beta, destDesc, destData));
	}

	/**
	 * Function to perform forward activation<br>
	 * Original signature : <code>cudnnStatus_t cudnnActivationForward(cudnnHandle_t, cudnnActivationMode_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 619</i>
	 * @throws CudnnException 
	 */
	public void activationForward(int mode, ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData) throws CudnnException
	{
		checkError(library.get().cudnnActivationForward(handle, mode, alpha, srcDesc, srcData, beta, destDesc, destData));
	}

	/**
	 * Function to perform forward softmax<br>
	 * Original signature : <code>cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t, cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 487</i>
	 * @throws CudnnException 
	 */
	public void softmaxForward(int algorithm, int mode, ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData) throws CudnnException
	{
		checkError(library.get().cudnnSoftmaxForward(handle, algorithm, mode, alpha, srcDesc, srcData, beta, destDesc, destData));
	}
	
	/**
	 * Function to perform backward activation<br>
	 * Original signature : <code>cudnnStatus_t cudnnActivationBackward(cudnnHandle_t, cudnnActivationMode_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 630</i>
	 * @throws CudnnException 
	 */
	public void activationBackward(int mode, ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, CudnnLibrary.cudnnTensorDescriptor_t srcDiffDesc, Pointer srcDiffData, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDiffDesc, Pointer destDiffData) throws CudnnException
	{
		checkError(library.get().cudnnActivationBackward(handle, mode, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData));
	}
	
	/**
	 * Function to perform backward pooling<br>
	 * Original signature : <code>cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t, const cudnnPoolingDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 592</i>
	 * @throws CudnnException 
	 */
	public void poolingBackward(CudnnLibrary.cudnnPoolingDescriptor_t poolingDesc, ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, CudnnLibrary.cudnnTensorDescriptor_t srcDiffDesc, Pointer srcDiffData, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDiffDesc, Pointer destDiffData) throws CudnnException
	{
		checkError(library.get().cudnnPoolingBackward(handle, poolingDesc, alpha, srcDesc, srcData, srcDiffDesc, srcDiffData, destDesc, destData, beta, destDiffDesc, destDiffData));
	}
	
	/**
	 * Functions to perform the backward multiconvolution<br>
	 * Original signature : <code>cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 423</i>
	 * @throws CudnnException 
	 */
	public void convolutionBackwardBias(ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t destDesc, Pointer destData) throws CudnnException
	{
		checkError(library.get().cudnnConvolutionBackwardBias(handle, alpha, srcDesc, srcData, beta, destDesc, destData));
	}

	/**
	 * Original signature : <code>cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnConvolutionDescriptor_t, const void*, const cudnnFilterDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 434</i>
	 * @throws CudnnException 
	 */
	public void convolutionBackwardFilter(ByReference alpha, CudnnLibrary.cudnnTensorDescriptor_t srcDesc, Pointer srcData, CudnnLibrary.cudnnTensorDescriptor_t diffDesc, Pointer diffData, CudnnLibrary.cudnnConvolutionDescriptor_t convDesc, ByReference beta, CudnnLibrary.cudnnFilterDescriptor_t gradDesc, Pointer gradData) throws CudnnException
	{
		checkError(library.get().cudnnConvolutionBackwardFilter(handle, alpha, srcDesc, srcData, diffDesc, diffData, convDesc, beta, gradDesc, gradData));
	}
	
	/**
	 * Original signature : <code>cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t, const void*, const cudnnFilterDescriptor_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnConvolutionDescriptor_t, const void*, const cudnnTensorDescriptor_t, void*)</code><br>
	 * <i>native declaration : line 447</i>
	 * @throws CudnnException 
	 */
	public void convolutionBackwardData(ByReference alpha, CudnnLibrary.cudnnFilterDescriptor_t filterDesc, Pointer filterData, CudnnLibrary.cudnnTensorDescriptor_t diffDesc, Pointer diffData, CudnnLibrary.cudnnConvolutionDescriptor_t convDesc, ByReference beta, CudnnLibrary.cudnnTensorDescriptor_t gradDesc, Pointer gradData) throws CudnnException
	{
		checkError(library.get().cudnnConvolutionBackwardData(handle, alpha, filterDesc, filterData, diffDesc, diffData, convDesc, beta, gradDesc, gradData));
	}

	public CudnnLibrary.Instance getLibrary() {
		return library;
	}
	
	
//	/**
//	 * Tensor Bias addition : srcDest = alpha * bias + beta * srcDestDesc<br>
//	 * Original signature : <code>cudnnStatus_t cudnnAddTensor(cudnnHandle_t, cudnnAddMode_t, const void*, const cudnnTensorDescriptor_t, const void*, const void*, cudnnTensorDescriptor_t, void*)</code><br>
//	 * <i>native declaration : line 229</i>
//	 */
//	public void addTensor(CudnnLibrary.cudnnHandle_t handle, int mode, Pointer alpha, CudnnLibrary.cudnnTensorDescriptor_t biasDesc, Pointer biasData, Pointer beta, CudnnLibrary.cudnnTensorDescriptor_t srcDestDesc, Pointer srcDestData)
//	{
//		int cudnnAddTensor(CudnnLibrary.cudnnHandle_t handle, int mode, Pointer alpha, CudnnLibrary.cudnnTensorDescriptor_t biasDesc, Pointer biasData, Pointer beta, CudnnLibrary.cudnnTensorDescriptor_t srcDestDesc, Pointer srcDestData);
//	}
	
	//v3
	public int getConvolutionBackwardFilterAlgorithm(
			cudnnTensorDescriptor_t srcDesc,
			cudnnTensorDescriptor_t diffDesc,
			cudnnConvolutionDescriptor_t convDesc,
			cudnnFilterDescriptor_t gradDesc,
			int preference,
			int memoryLimitInbytes) throws CudnnException
	{
		IntByReference algo=new IntByReference();
		checkError(library.get().cudnnGetConvolutionBackwardFilterAlgorithm(
				handle,
				srcDesc,
				diffDesc,
				convDesc,
				gradDesc,
				preference,
				memoryLimitInbytes,
				algo));
		return algo.getValue();
	}
	
	public void convolutionBackwardFilter(
			ByReference alpha,
			cudnnTensorDescriptor_t srcDesc,
			Pointer srcData,
			cudnnTensorDescriptor_t diffDesc,
			Pointer diffData,
			cudnnConvolutionDescriptor_t convDesc,
			int algo,
			Pointer workSpace,
			int workSpaceSizeInBytes,
			ByReference beta,
			cudnnFilterDescriptor_t gradDesc,
			Pointer gradData) throws CudnnException
	{
		checkError(library.get().cudnnConvolutionBackwardFilter_v3(
				handle,
				alpha,
				srcDesc,
				srcData,
				diffDesc,
				diffData,
				convDesc,
				algo,
				workSpace,
				workSpaceSizeInBytes,
				beta,
				gradDesc,
				gradData));
	}
	
	public void convolutionBackwardData(
			ByReference alpha,
			cudnnFilterDescriptor_t filterDesc,
			Pointer filterData,
			cudnnTensorDescriptor_t diffDesc,
			Pointer diffData,
			cudnnConvolutionDescriptor_t convDesc,
			int algo,
			Pointer workSpace,
			int workSpaceSizeInBytes,
			ByReference beta,
			cudnnTensorDescriptor_t gradDesc,
			Pointer gradData) throws CudnnException
	{
		checkError(library.get().cudnnConvolutionBackwardData_v3(
				handle,
				alpha,
				filterDesc,
				filterData,
				diffDesc,
				diffData,
				convDesc,
				algo,
				workSpace,
				workSpaceSizeInBytes,
				beta,
				gradDesc,
				gradData));
	}
	
	public int convolutionBackwardFilterWorkspaceSize(
			cudnnTensorDescriptor_t srcDesc,
			cudnnTensorDescriptor_t diffDesc,
			cudnnConvolutionDescriptor_t convDesc,
			cudnnFilterDescriptor_t gradDesc,
			int algo) throws CudnnException
	{
		IntByReference size=new IntByReference();
		checkError(library.get().cudnnGetConvolutionBackwardFilterWorkspaceSize(
				handle,
				srcDesc,
				diffDesc,
				convDesc,
				gradDesc,
				algo,
				size));
		return size.getValue();
	}
	
	public int convolutionBackwardDataWorkspaceSize(
			cudnnFilterDescriptor_t filterDesc,
			cudnnTensorDescriptor_t diffDesc,
			cudnnConvolutionDescriptor_t convDesc,
			cudnnTensorDescriptor_t gradDesc,
			int algo) throws CudnnException
	{
		IntByReference size=new IntByReference();
		checkError(library.get().cudnnGetConvolutionBackwardDataWorkspaceSize(
				handle,
				filterDesc,
				diffDesc,
				convDesc,
				gradDesc,
				algo,
				size));
		return size.getValue();
	}
}
