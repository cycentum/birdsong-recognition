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

import java.util.Comparator;

public class Pair<T0, T1>
{
	private T0 value0;
	private T1 value1;
	
	public Pair(T0 value0, T1 value1) {
		this.value0 = value0;
		this.value1 = value1;
	}

	public T0 get0() {
		return value0;
	}

	public void set0(T0 value0) {
		this.value0 = value0;
	}

	public T1 get1() {
		return value1;
	}

	public void set1(T1 value1) {
		this.value1 = value1;
	}
	
	public static <T0 extends Comparable<T0>, T1> Comparator<Pair<T0, T1>> comparator0()
	{
		return (o1, o2)->o1.get0().compareTo(o2.get0());
	}

	public static <T0, T1 extends Comparable<T1>> Comparator<Pair<T0, T1>> comparator1()
	{
		return (o1, o2)->o1.get1().compareTo(o2.get1());
	}
}
