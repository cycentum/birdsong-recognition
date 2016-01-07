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

import com.sun.jna.ptr.IntByReference;

import cudnn.ActivationMode;
import cudnn.Cuda;
import cudnn.CudaDriver;
import cudnn.CudaException;
import cudnn.Cudnn;
import cudnn.CudnnException;
import cudnn.CudnnLibrary;
import cudnn.CudnnLibrary.cudnnConvolutionFwdPreference_t;
import cudnn.FloatType;
import cudnn.Pointer;

/**
 * A class for a convolutional layer.
 * {@link #init(FloatType, Cudnn, int)} and {@link #cudaMalloc(FloatType, int)} must be called before computation.
 * {@link #destroy(Cudnn)} must be called before termination of the program.
 * **Dev means a pointer in a GPU.
 * @author koumura
 *
 */
public class ConvLayer implements ParamLayer, NonDataLayer
{
	private Layer lower;
	private int numChannel, filterWidth, filterHeight, width, height;

	private float[] weightF, biasF;
	private double[] weightD, biasD;
	private int weightSize, biasSize;
	
	private CudnnLibrary.cudnnTensorDescriptor_t tensorDesc;
	private CudnnLibrary.cudnnTensorDescriptor_t biasTensorDesc;
	private CudnnLibrary.cudnnFilterDescriptor_t filterDesc;
	private CudnnLibrary.cudnnConvolutionDescriptor_t convDesc;
	private IntByReference forwardAlgo;
	private int backwardFilterAlgo, backwardDataAlgo;
	
	private Pointer weightDev, biasDev, actDev, gradWeightDev, gradBiasDev, derActDev;
	
	private ActivationMode activationMode;

	private Pointer workspaceDev;
	private int workspaceSize;
	
	private BackwardAlgorithm backwardAlgo;
	
	public ConvLayer(int numChannel, int filterHeight, int filterWidth, ActivationMode activationMode, Layer lower, BackwardAlgorithm backwardAlgo)
	{
		this.lower=lower;
		
		weightSize=lower.getNumChannel()* filterWidth* filterHeight* numChannel;
		biasSize=numChannel;
		this.numChannel=numChannel;
		this.width=lower.getWidth()-filterWidth+1;
		this.height=lower.getHeight()-filterHeight+1;
		this.activationMode=activationMode;
		this.filterWidth=filterWidth;
		this.filterHeight=filterHeight;
		
		this.backwardAlgo=backwardAlgo;
	}

	public void destroy(Cudnn cudnn) throws CudnnException, CudaException
	{
		getTensorDesc().destroy(cudnn);
		biasTensorDesc.destroy(cudnn);
		filterDesc.destroy(cudnn);
		convDesc.destroy(cudnn);

		if(getWeightDev()!=null) Cuda.free(getWeightDev());
		if(getBiasDev()!=null) Cuda.free(getBiasDev());
		if(actDev!=null) Cuda.free(actDev);
//		/*act!=value*/if(getValueDev()!=null) Cuda.free(getValueDev());
		if(getGradWeightDev()!=null) Cuda.free(getGradWeightDev());
		if(getGradBiasDev()!=null) Cuda.free(getGradBiasDev());
		if(derActDev!=null) Cuda.free(derActDev);
//		/*act!=value*/if(derValueDev!=null) Cuda.free(derValueDev);
	}
	
	public void cudaMalloc(FloatType floatType, int batchSize) throws CudaException
	{
		actDev=Cuda.malloc(floatType.getBytes()* batchSize * getNumChannel() * getHeight()                  * getWidth());
//		/*act!=value*/valueDev=Cuda.malloc(floatType.getBytes() * batchSize * getNumChannel() * getHeight()                  * getWidth());
//		derActDev=Cuda.malloc(floatType.getBytes()*lower.sizeBCHW());
//		/*act!=value*/derValueDev=Cuda.malloc(  floatType.getBytes() * lower.sizeBCHW());
		derActDev=Cuda.malloc(floatType.getBytes()*sizeCHW()*batchSize);
//		/*act!=value*/derValueDev=Cuda.malloc(  floatType.getBytes() * sizeBCHW());
		weightDev=Cuda.malloc(    floatType.getBytes() * weightSize);
		biasDev=Cuda.malloc(    floatType.getBytes() * biasSize);
		gradWeightDev=Cuda.malloc(    floatType.getBytes() * weightSize);
		gradBiasDev=Cuda.malloc(    floatType.getBytes() * biasSize);
	}
	public void setWodkspace(Pointer workspaceDev, int workspaceSize)
	{
		this.workspaceDev=workspaceDev;
		this.workspaceSize=workspaceSize;
	}
	
	protected void compForwardAct(FloatType floatType, Cudnn cudnn) throws CudnnException, CudaException
	{
		cudnn.convolutionForward(Cudnn.const1(floatType), lower.getTensorDesc(), lower.getValueDev().getJnaPointer(), filterDesc, weightDev.getJnaPointer(), convDesc, forwardAlgo.getValue(), workspaceDev.getJnaPointer(), workspaceSize, Cudnn.const0(floatType), getTensorDesc(), actDev.getJnaPointer());
		cudnn.addTensor(CudnnLibrary.cudnnAddMode_t.CUDNN_ADD_SAME_C, Cudnn.const1(floatType), biasTensorDesc, getBiasDev().getJnaPointer(), Cudnn.const1(floatType), getTensorDesc(), actDev.getJnaPointer());
	}
	
	public void compForward(FloatType floatType, Cudnn cudnn, int batchSize, CudaDriver driver) throws CudnnException, CudaException
	{
		compForwardAct(floatType, cudnn);
		if(activationMode!=ActivationMode.IDENT)
			cudnn.activationForward(activationMode.getValue(), Cudnn.const1(floatType), getTensorDesc(), actDev.getJnaPointer(), Cudnn.const0(floatType), getTensorDesc(), actDev.getJnaPointer());
	}
	
	public void compBackward(FloatType floatType, Cudnn cudnn, int batchSize, CudaDriver driver) throws CudnnException, CudaException
	{
		if(activationMode!=ActivationMode.IDENT)
		{
			cudnn.activationBackward(activationMode.getValue(), Cudnn.const1(floatType),
					getTensorDesc(), getValueDev().getJnaPointer(), getTensorDesc(), derActDev.getJnaPointer(),
					getTensorDesc(), actDev.getJnaPointer(), Cudnn.const0(floatType), getTensorDesc(), derActDev.getJnaPointer());
		}
		
		// valueDev layer
		cudnn.convolutionBackwardBias(Cudnn.const1(floatType), getTensorDesc(),
				derActDev.getJnaPointer(), Cudnn.const0(floatType), biasTensorDesc, getGradBiasDev().getJnaPointer());

		cudnn.convolutionBackwardFilter(Cudnn.const1(floatType), lower.getTensorDesc(), 
				lower.getValueDev().getJnaPointer(), getTensorDesc(), derActDev.getJnaPointer(), convDesc,
				backwardFilterAlgo, workspaceDev.getJnaPointer(), workspaceSize,
				Cudnn.const0(floatType), filterDesc, gradWeightDev.getJnaPointer());
		
		if(lower instanceof NonDataLayer)
		{
			cudnn.convolutionBackwardData(Cudnn.const1(floatType), filterDesc, 
					weightDev.getJnaPointer(), getTensorDesc(), derActDev.getJnaPointer(), convDesc,
					backwardDataAlgo, workspaceDev.getJnaPointer(), workspaceSize,
					Cudnn.const0(floatType), lower.getTensorDesc(), ((NonDataLayer) lower).getDerValueDev().getJnaPointer());
		}
	}
	
	public void init(FloatType floatType, Cudnn cudnn, int batchSize) throws CudnnException
	{
		tensorDesc=cudnn.createTensorDescriptor();
		cudnn.initTensorDesc(floatType, this, batchSize);
		biasTensorDesc=cudnn.createTensorDescriptor();
		
		convDesc=cudnn.createConvolutionDescriptor();
		biasTensorDesc.setTensor4dDescriptor(cudnn, CudnnLibrary.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW,
				floatType.getDataTypeValue(),
				1, getNumChannel(),
				1, 1);
		
		filterDesc=cudnn.createFilterDescriptor();
		filterDesc.setFilter4dDescriptor(cudnn, floatType.getDataTypeValue(),
				getNumChannel(),
				lower.getNumChannel(), 
				filterHeight,
				filterWidth);

		convDesc.setConvolution2dDescriptor(cudnn, 0, 0,
				1, 1,
				1, 1,
				CudnnLibrary.cudnnConvolutionMode_t.CUDNN_CROSS_CORRELATION);

		int algoValue=cudnn.getConvolutionForwardAlgorithm(lower.getTensorDesc(), filterDesc, convDesc, tensorDesc, cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0);
		forwardAlgo=new IntByReference(algoValue);
		
		if(backwardAlgo==BackwardAlgorithm.FAST_NON_DETERMINISTIC)
		{
			backwardFilterAlgo=CudnnLibrary.cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
			backwardDataAlgo=CudnnLibrary.cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		}
		else
		{
			backwardFilterAlgo=CudnnLibrary.cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
			backwardDataAlgo=CudnnLibrary.cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
		}
	}
	
	public float[] getWeightF() {
		return weightF;
	}

	public float[] getBiasF() {
		return biasF;
	}

	public int getNumChannel() {
		return numChannel;
	}

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}

	public Pointer getWeightDev() {
		return weightDev;
	}

	public Pointer getBiasDev() {
		return biasDev;
	}

	public CudnnLibrary.cudnnTensorDescriptor_t getTensorDesc() {
		return tensorDesc;
	}

	public int getWeightSize() {
		return weightSize;
	}

	public void setWeightSize(int weightSize) {
		this.weightSize = weightSize;
	}

	public int getBiasSize() {
		return biasSize;
	}

	public void setBiasSize(int biasSize) {
		this.biasSize = biasSize;
	}

	public void setWeightF(float[] weightF) {
		this.weightF = weightF;
	}

	public void setBiasF(float[] biasF) {
		this.biasF = biasF;
	}

	public void setWeightD(double[] weightD) {
		this.weightD = weightD;
	}

	public void setBiasD(double[] biasD) {
		this.biasD = biasD;
	}

	@Override
	public Layer getLower() {
		return lower;
	}

	@Override
	public Pointer getValueDev() {
//		/*act!=value*/return valueDev;
		return actDev;
	}

	public Pointer getGradWeightDev() {
		return gradWeightDev;
	}

	public Pointer getGradBiasDev() {
		return gradBiasDev;
	}

	@Override
	public Pointer getDerValueDev()
	{
//		/*act!=value*/return derValueDev;
		return derActDev;
	}
	
	public double[] getWeightD(){return weightD;}
	
	public double[] getBiasD(){return biasD;}

	protected Pointer getActDev() {
		return actDev;
	}

	public Pointer getDerActDev() {
		return derActDev;
	}
	
	public int getFilterHeight(){return filterHeight;}
	
	public int getFilterWidth(){return filterWidth;}

	public int compWorkspaceSize(Cudnn cudnn) throws CudnnException
	{
		int sizeInBytes=Integer.max(Integer.max(
			cudnn.getConvolutionForwardWorkspaceSize(lower.getTensorDesc(), filterDesc, convDesc, tensorDesc, forwardAlgo.getValue()),
			cudnn.convolutionBackwardDataWorkspaceSize(filterDesc, tensorDesc, convDesc, lower.getTensorDesc(), backwardDataAlgo)),
			cudnn.convolutionBackwardFilterWorkspaceSize(lower.getTensorDesc(), tensorDesc, convDesc, filterDesc, backwardFilterAlgo));
		return sizeInBytes;
	}

	public static enum BackwardAlgorithm
	{
		FAST_NON_DETERMINISTIC, SLOW_DETERMINISTIC;
	}
}
