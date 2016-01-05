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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedList;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.Node;
import org.w3c.dom.bootstrap.DOMImplementationRegistry;
import org.w3c.dom.ls.DOMImplementationLS;
import org.w3c.dom.ls.LSInput;
import org.w3c.dom.ls.LSOutput;
import org.w3c.dom.ls.LSParser;
import org.w3c.dom.ls.LSSerializer;

public class XmlUtils
{
	private static final DocumentBuilder DOCUMENT_BUILDER;
	private static final DOMImplementationRegistry REGISTRY;

	static
	{
		DocumentBuilder documentBuilder=null;
		try
		{
			documentBuilder=DocumentBuilderFactory.newInstance().newDocumentBuilder();
		}
		catch (ParserConfigurationException e) {e.printStackTrace();}
		DOCUMENT_BUILDER=documentBuilder;
		
		DOMImplementationRegistry registry = null;
		try
		{
			registry = DOMImplementationRegistry.newInstance();
		}
		catch (ClassNotFoundException | InstantiationException | IllegalAccessException | ClassCastException e) {e.printStackTrace();}
		REGISTRY=registry;
	}
	
	public static Element rootElement(String nodeName)
	{
		Document document = DOCUMENT_BUILDER.newDocument();
		Element root= document.createElement(nodeName);
		document.appendChild(root);
		return root;
	}
	
	public static void write(Document document, OutputStream stream)
	{
		DOMImplementationLS impl = (DOMImplementationLS)REGISTRY.getDOMImplementation("LS");
		LSSerializer writer = impl.createLSSerializer();
		LSOutput output=impl.createLSOutput();
		output.setByteStream(stream);
		writer.write(document, output);
	}
	public static void write(Document document, Path file) throws IOException
	{
		BufferedOutputStream stream=new BufferedOutputStream(Files.newOutputStream(file));
		write(document, stream);
		stream.close();
	}
	
	public static void removeBlankText(Node node, LinkedList<Pair<Node, Node>> removed)
	{
		for(int c=0; c<node.getChildNodes().getLength(); ++c)
		{
			Node child=node.getChildNodes().item(c);
			if(child.getNodeType()==Node.TEXT_NODE&&child.getTextContent().trim().length()==0) removed.add(new Pair<>(node, child));
			else removeBlankText(child, removed);
		}
	}
	
	public static Document parse(InputStream stream)
	{
		DOMImplementationLS impl = (DOMImplementationLS)REGISTRY.getDOMImplementation("LS");
		LSParser builder = impl.createLSParser(DOMImplementationLS.MODE_SYNCHRONOUS, null);
		LSInput input=impl.createLSInput();
		input.setByteStream(stream);
		Document document = builder.parse(input);

		LinkedList<Pair<Node, Node>> removed=new LinkedList<>();
		removeBlankText(document, removed);
		for(Pair<Node, Node> r: removed) r.get0().removeChild(r.get1());
		
		return document;
	}
	public static Document parse(Path file) throws IOException
	{
		BufferedInputStream stream=new BufferedInputStream(Files.newInputStream(file));
		Document document=parse(stream);
		stream.close();
		return document;
	}
	
	public static <T> Element addChild(Node parent, String name, T content)
	{
		Element child=parent.getOwnerDocument().createElement(name);
		if(content!=null) child.setTextContent(content.toString());
		parent.appendChild(child);
		return child;
	}
	
	public static boolean nodeBoolean(Node node){return Boolean.parseBoolean(node/*.getFirstChild()*/.getTextContent());}
	public static byte nodeByte(Node node){return Byte.parseByte(node/*.getFirstChild()*/.getTextContent());}
	public static int nodeInt(Node node){return Integer.parseInt(node/*.getFirstChild()*/.getTextContent());}
	public static short nodeShort(Node node){return Short.parseShort(node/*.getFirstChild()*/.getTextContent());}
	public static float nodeFloat(Node node){return Float.parseFloat(node/*.getFirstChild()*/.getTextContent());}
	public static double nodeDouble(Node node){return Double.parseDouble(node/*.getFirstChild()*/.getTextContent());}
	public static String nodeText(Node node){return node/*.getFirstChild()*/.getTextContent();}
}
