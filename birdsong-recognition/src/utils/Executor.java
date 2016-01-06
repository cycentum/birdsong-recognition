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
package utils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class Executor
{
	private ExecutorService executor;
	private int numThread;
	
	public Executor(int numThread)
	{
		this.numThread = numThread;
		this.executor = Executors.newFixedThreadPool(numThread);
	}

	public ExecutorService get() {
		return executor;
	}

	public int getNumThread() {
		return numThread;
	}
}
