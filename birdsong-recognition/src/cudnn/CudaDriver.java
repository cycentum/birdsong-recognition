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

import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import java.nio.file.Path;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class CudaDriver
{
	private CUmodule module;
	private CUcontext context;

	public CudaDriver() throws CudaException
	{
		JCudaDriver.cuInit(0);
		CUdevice device = new CUdevice();
		checkError(JCudaDriver.cuDeviceGet(device, 0));
		context = new CUcontext();
		checkError(JCudaDriver.cuCtxCreate(context, 0, device));
		module = new CUmodule();
	}
	public CudaDriver(Path path) throws CudaException
	{
		this();
		load(path.toAbsolutePath().toString());
	}
	
	public void destroy() throws CudaException
	{
		checkError(JCudaDriver.cuCtxDestroy(context));
	}

	private static void checkError(int status) throws CudaException
	{
		Cuda.checkError(status);
	}

	public void load(String ptxPath) throws CudaException
	{
		checkError(JCudaDriver.cuModuleLoad(module, ptxPath));
	}

	/*public static CUdeviceptr malloc(int size) throws CudaException
	{
		CUdeviceptr pointer= new CUdeviceptr();
		checkError(JCudaDriver.cuMemAlloc(pointer, size));
		return pointer;
	}

	public static void free(CUdeviceptr pointer) throws CudaException
	{
		checkError(JCudaDriver.cuMemFree(pointer));
	}*/

	public CUfunction getFunction(String name) throws CudaException
	{
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, name);
        return function;
	}
	
	public static void call(CUfunction function, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, Pointer kernelParams) throws CudaException
	{
		checkError(cuLaunchKernel(function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, null, kernelParams, null));
	}
	public void call(String function, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, Pointer kernelParams) throws CudaException
	{
		call(getFunction(function), gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, kernelParams);
	}
	
	public static void ctxSynchronize() throws CudaException
	{
		checkError(cuCtxSynchronize());
	}
	
	/*public static void memcpyDeviceToHost(Pointer dstHost, CUdeviceptr srcDevice, int size) throws CudaException
	{
		checkError(JCudaDriver.cuMemcpyDtoH(dstHost, srcDevice, size));
	}
	
	public static void memcpyAsyncHostToDevice(CUdeviceptr dstDevice, Pointer srcHost, int size) throws CudaException
	{
		checkError(JCudaDriver.cuMemcpyHtoDAsync(dstDevice, srcHost, size, null));
	}*/
}
