/*
 *  Created by David Chiu
 *  Dec. 28th, 2018
 *
 */
package com.davidchiu.ncnncam;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static com.davidchiu.ncnncam.Detection.COLORS;

public class Ncnn
{
    public Vector<String> mLabel = new Vector<>();
    private int imageWidth;
    private int imageHeight;

//    public native boolean init(byte[] param, byte[] bin, byte[] words, AssetManager mgr);
    public native boolean init(AssetManager mgr);

    public native float[] nativeDetect(Bitmap bitmap);

    static {
        System.loadLibrary("ncnn_jni");
    }

    public void setImageSize(int width, int height) {
        imageHeight = height;
        imageWidth = width;
    }

    public boolean initNcnn(Context context, String paramFile, String weightsFile, String labels) throws IOException
    {
        return init(context.getAssets());
    }

    //
    // This needs to be customized based on underling models: yolo, ssd etc.
    // Now only yolo is supported
    //
    public List<Detection> detect(Bitmap image) {
        float[] result = nativeDetect(image);
        List<Detection> list=null;
        if (result == null || result[0] <= -1) return list;

        list = parseNcnnResults(result);

        return list;
    }

    List<Detection> parseNcnnResults(float[] ncnnArray) {
        ArrayList<Detection> list = new ArrayList<>();
        if (ncnnArray == null || ncnnArray.length<=0 || ncnnArray[0]<=0) {
            return  null;
        }

        int nItems = (int)((ncnnArray.length-1)/ncnnArray[0]);
        int nParams = (int)ncnnArray[0];
        for (int i = 0; i < nItems; i++) {
            String id = "" + (int)(ncnnArray[i*nParams+1]);
            RectF location = new RectF(ncnnArray[i*nParams+3]*imageWidth, ncnnArray[i*nParams+4]*imageHeight, ncnnArray[i*nParams+5]*imageWidth, ncnnArray[i*nParams+6]*imageHeight);
//            RectF location = new RectF(ncnnArray[i*nParams+4]*imageHeight,ncnnArray[i*nParams+3]*imageWidth,  ncnnArray[i*nParams+6]*imageHeight, ncnnArray[i*nParams+5]*imageWidth);
//            RectF location = new RectF(ncnnArray[i*nParams+3]*imageHeight, ncnnArray[i*nParams+4]*imageWidth, ncnnArray[i*nParams+5]*imageHeight, ncnnArray[i*nParams+6]*imageWidth);
//            LOGGER.i("BOX RAW: %.2f %.2f", ncnnArray[i*nParams+3], ncnnArray[i*nParams+4]);
            Detection recognition = new Detection();
            recognition.id = (int)ncnnArray[i*nParams + 1];
//            recognition.title = mLabel.get(recognition.id);
            if (id.equals("1")) {
                recognition.title = "Son";
            }
            else {
                recognition.title = "new";
            }
//            recognition.title = id;
            recognition.detectionConfidence = ncnnArray[i*nParams+2];
            recognition.location = location;
            recognition.color = COLORS[0];
            list.add(recognition);
            Log.i("ncnn", "detect: " + ncnnArray[i*nParams+1] + " " +  ncnnArray[i*nParams+2] + " " +
                   ncnnArray[i*nParams+3]  + " " + ncnnArray[i*nParams+4] + " " +  ncnnArray[i*nParams+5] + " " + ncnnArray[i*nParams+6] +
                    " width: " + imageWidth + " height: " + imageHeight);
        }
        return list;
    }

}
