/*
 * Copyright (C) 2016 Takuya KOUMURA
 * https://github.com/takuya-koumura/birdsong-recognition
 *
 * This file is part of birdsong-recognition.
 * 
 * Birdsong-recognition is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Birdsong-recognition is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with birdsong-recognition.  If not, see <http://www.gnu.org/licenses/>.
 */
package utils;

import java.util.Arrays;

public class ArrayUtils
{
	public static int[] createSequence(int start, int end)
	{
		int[] array=new int[end-start];
		for(int i=0; i<array.length; ++i) array[i]=i+start;
		return array;
	}
	
	public static double[] createFilled(int size, double value)
	{
		double[] array=new double[size];
		Arrays.fill(array, value);
		return array;
	}
	public static int[] createFilled(int size, int value)
	{
		int [] array=new int[size];
		Arrays.fill(array, value);
		return array;
	}
	
	public static int last(int[] array){return array[array.length-1];}
	
	public static int sum01(int[] array){return array[0]+array[1];}
}
