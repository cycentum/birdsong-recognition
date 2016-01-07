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

import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.Pointer;

/**
 * A class for cuda functions.
 * @author koumura
 *
 */
public class Cuda
{
	public static void setDevice(int device) throws CudaException
	{
		checkError(JCuda.cudaSetDevice(device));
	}
	
	public static void checkError(int status) throws CudaException
	{
		if(status!=cudaError.cudaSuccess)
		{
			throw new CudaException(cudaError.stringFor(status));
		}
	}
	
	/**
	 * @param size in bytes
	 */
	public static cudnn.Pointer malloc(long size) throws CudaException
	{
		Pointer deviceData=new Pointer();
		checkError(JCuda.cudaMalloc(deviceData, size));
		return cudnn.Pointer.fromJcuda(deviceData);
	}
	
	public static void memcpyAsyncHostToDevice(Pointer deviceData, Pointer hostData, long size) throws CudaException
	{
		checkError(JCuda.cudaMemcpyAsync(deviceData, hostData, size, cudaMemcpyKind.cudaMemcpyHostToDevice, null));
	}
	public static void memcpyAsyncHostToDevice(Pointer deviceData, int[] hostData) throws CudaException
	{
		memcpyAsyncHostToDevice(deviceData, Pointer.to(hostData), Integer.BYTES*hostData.length);
	}
	public static void memcpyAsyncHostToDevice(Pointer deviceData, byte[] hostData) throws CudaException
	{
		memcpyAsyncHostToDevice(deviceData, Pointer.to(hostData), hostData.length);
	}
	public static void memcpyAsyncHostToDevice(Pointer deviceData, float[] hostData) throws CudaException
	{
		memcpyAsyncHostToDevice(deviceData, Pointer.to(hostData), Float.BYTES*hostData.length);
	}
	public static void memcpyAsyncHostToDevice(Pointer deviceData, double[] hostData) throws CudaException
	{
		memcpyAsyncHostToDevice(deviceData, Pointer.to(hostData), Double.BYTES*hostData.length);
	}
	
	public static void memcpyAsyncDeviceToHost(Pointer hostData, Pointer deviceData, long size) throws CudaException
	{
		checkError(JCuda.cudaMemcpyAsync(hostData, deviceData, size, cudaMemcpyKind.cudaMemcpyDeviceToHost, null));
	}
	
	public static void memcpyAsyncDeviceToDevice(Pointer dst, Pointer src, long size) throws CudaException
	{
		checkError(JCuda.cudaMemcpyAsync(dst, src, size, cudaMemcpyKind.cudaMemcpyDeviceToDevice, null));
	}

	public static void memcpyDeviceToHost(Pointer hostData, Pointer deviceData, long size) throws CudaException
	{
		checkError(JCuda.cudaMemcpy(hostData, deviceData, size, cudaMemcpyKind.cudaMemcpyDeviceToHost));
	}
	
	public static int[] memcpyDeviceToIntArray(Pointer deviceData, int size) throws CudaException
	{
		int[] host=new int[size];
		memcpyDeviceToHost(Pointer.to(host), deviceData, Integer.BYTES*size);
		return host;
	}
	
	public static byte[] memcpyDeviceToByteArray(Pointer deviceData, int size) throws CudaException
	{
		byte[] host=new byte[size];
		memcpyDeviceToHost(Pointer.to(host), deviceData, Byte.BYTES*size);
		return host;
	}
	
	public static float[] memcpyDeviceToFloatArray(Pointer deviceData, int size) throws CudaException
	{
		float[] host=new float[size];
		memcpyDeviceToHost(Pointer.to(host), deviceData, Float.BYTES*size);
		return host;
	}
	
	public static double[] memcpyDeviceToDoubleArray(Pointer deviceData, int size) throws CudaException
	{
		double[] host=new double[size];
		memcpyDeviceToHost(Pointer.to(host), deviceData, Double.BYTES*size);
		return host;
	}
	
	public static void memcpyDeviceToDevice(Pointer dst, Pointer src, long size) throws CudaException
	{
		checkError(JCuda.cudaMemcpy(dst, src, size, cudaMemcpyKind.cudaMemcpyDeviceToDevice));
	}
	
	public static void memcpyHostToDevice(Pointer deviceData, Pointer hostData, long size) throws CudaException
	{
		checkError(JCuda.cudaMemcpy(deviceData, hostData, size, cudaMemcpyKind.cudaMemcpyHostToDevice));
	}
	public static void memcpyHostToDevice(Pointer deviceData, int[] hostData) throws CudaException
	{
		memcpyHostToDevice(deviceData, Pointer.to(hostData), Integer.BYTES*hostData.length);
	}
	public static void memcpyHostToDevice(Pointer deviceData, byte[] hostData) throws CudaException
	{
		memcpyHostToDevice(deviceData, Pointer.to(hostData), Byte.BYTES*hostData.length);
	}
	public static void memcpyHostToDevice(Pointer deviceData, float[] hostData) throws CudaException
	{
		memcpyHostToDevice(deviceData, Pointer.to(hostData), Float.BYTES*hostData.length);
	}
	public static void memcpyHostToDevice(Pointer deviceData, double[] hostData) throws CudaException
	{
		memcpyHostToDevice(deviceData, Pointer.to(hostData), Double.BYTES*hostData.length);
	}
	
	public static void free(Pointer deviceData) throws CudaException
	{
		checkError(JCuda.cudaFree(deviceData));
	}
	
	public static void deviceSynchronize() throws CudaException
	{
		checkError(JCuda.cudaDeviceSynchronize());
	}
	
	public static void deviceReset() throws CudaException
	{
		checkError(JCuda.cudaDeviceReset());
	}
}
