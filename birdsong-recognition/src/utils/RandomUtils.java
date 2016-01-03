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

import java.util.ArrayList;

import org.apache.commons.math3.random.MersenneTwister;

public class RandomUtils
{
	public static int[] permutation(int size, MersenneTwister random)
	{
		int[] index=new int[size];
		ArrayList<Integer> remaining=new ArrayList<>();
		for(int i=0; i<size; ++i) remaining.add(i);
		for(int i=0; i<size; ++i)
		{
			int next=random.nextInt(remaining.size());
			index[i]=remaining.get(next);
			remaining.remove(next);
		}
		return index;
	}
	public static <T> ArrayList<T> permutation(ArrayList<T> list, MersenneTwister random)
	{
		int[] index=permutation(list.size(), random);
		ArrayList<T> permutation=new ArrayList<T>(list.size());
		for(int i: index) permutation.add(list.get(i));
		return permutation;
	}
}
