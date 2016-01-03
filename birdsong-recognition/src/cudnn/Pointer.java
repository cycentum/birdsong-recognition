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
package cudnn;

import java.util.LinkedList;

import com.sun.jna.PointerUtils;

public class Pointer extends jcuda.Pointer
{
	private com.sun.jna.Pointer jnaPointer;
	private LinkedList<Pointer> child;
	private Pointer parent;
	
	private Pointer(jcuda.Pointer jcudaPointer, com.sun.jna.Pointer jnaPointer)
	{
		super(jcudaPointer);
		this.jnaPointer = jnaPointer;
	}
	private Pointer(jcuda.Pointer jcudaPointer)
	{
		super(jcudaPointer);
		this.jnaPointer=PointerUtils.fromAddress(address());
	}
	private Pointer()
	{
		super();
		this.jnaPointer=PointerUtils.fromAddress(address());
	}

	private long address(){return getNativePointer()+getByteOffset();}
	
	public com.sun.jna.Pointer getJnaPointer() {
		return jnaPointer;
	}
	
	public Pointer withByteOffset(long offset)
	{
		Pointer ch=fromJcuda(super.withByteOffset(offset));
		if(child==null) child=new LinkedList<>();
		child.add(ch);
		ch.parent=this;
		return ch;
	}
	
	public void free() throws CudaException
	{
		if(parent==null) Cuda.free(this);
	}
	
	public static Pointer fromJcuda(jcuda.Pointer jcudaPointer)
	{
		if(jcudaPointer==null) return createNull();
		return new Pointer(jcudaPointer);
	}
	
	public static Pointer createNull()
	{
		return new Pointer();
	}
}
