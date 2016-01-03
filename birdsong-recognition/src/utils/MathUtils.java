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

import no.uib.cipr.matrix.NotConvergedException;
import no.uib.cipr.matrix.SymmPackEVD;
import no.uib.cipr.matrix.UpperSymmPackMatrix;

public class MathUtils
{
	public static int ceil(int numerator, int denominator)
	{
		if(numerator%denominator==0) return numerator/denominator;
		return numerator/denominator+1;
	}
	public static long ceil(long numerator, int denominator)
	{
		if(numerator%denominator==0) return numerator/denominator;
		return numerator/denominator+1;
	}
	
	private static double square(double x){return x*x;}
	
	public static double[] dpss(int fftLength, int nw) throws NotConvergedException
	{
		double w=(double)nw/fftLength;
		UpperSymmPackMatrix matrix=new UpperSymmPackMatrix(fftLength);
		for(int i=0; i<fftLength; ++i)
		{
			double value=square((double)(fftLength-1-2*i)/2)*Math.cos(2*Math.PI*w);
			matrix.set(i, i, value);
		}
		for(int i=1; i<fftLength; ++i)
		{
			double value=(double)i*(fftLength-i)/2;
			matrix.set(i, i-1, value);
			matrix.set(i-1, i, value);
		}
		
		int dimension=matrix.numRows();
		SymmPackEVD evd=SymmPackEVD.factorize(matrix);
		int rowIndex=0;
		for(int i=0; i<dimension; ++i) if(evd.getEigenvalues()[i]>evd.getEigenvalues()[rowIndex]) rowIndex=i;
		double[] eigenVector=new double[fftLength];
		for(int j=0; j<dimension; ++j) eigenVector[j]=evd.getEigenvectors().get(j, rowIndex);
		return eigenVector;
	}

	public static int subIntervalBegin(int size, int numSub, int subIndex)
	{
		return subIndex*size/numSub;
	}
	public static long subIntervalBegin(long size, int numSub, int subIndex)
	{
		return subIndex*size/numSub;
	}
	
	public static double sum(double... value)
	{
		double sum=0;
		for(double v: value) sum+=v;
		return sum;
	}
	public static int sum(int... value)
	{
		int sum=0;
		for(int v: value) sum+=v;
		return sum;
	}
	
	public static void divideBySum(double[] value)
	{
		double sum=sum(value);
		for(int i=0; i<value.length; ++i) value[i]/=sum;
	}
	public static double[] divideBySum(int[] value)
	{
		int sum=sum(value);
		double[] freq=new double[value.length];
		for(int i=0; i<value.length; ++i) freq[i]=(double)value[i]/sum;
		return freq;
	}
}
