/*
 *  Created by David Chiu
 *  Dec. 28th, 2018
 *
 */
package com.davidchiu.ncnncam;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.Environment;
import android.util.Log;
import android.content.res.AssetManager;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.io.FileWriter;
import java.io.File;
import java.nio.channels.FileChannel;
import java.nio.ByteBuffer;
import java.io.RandomAccessFile;
import android.content.Context;

import static com.davidchiu.ncnncam.Detection.COLORS;

public class Ncnn
{
    private static final Logger LOGGER = new Logger();
    public Vector<String> mLabel = new Vector<>();
    private int imageWidth;
    private int imageHeight;
    static ArrayList<String> acc_names = new ArrayList<>();

//    public native boolean init(byte[] param, byte[] bin, byte[] words, AssetManager mgr);
    public native boolean init(AssetManager mgr, int[] ids, float[] features);

    public native float[] nativeDetect(Bitmap bitmap);

    public native float[] embed(AssetManager mgr, String img_path);

    public native boolean load(AssetManager mgr, int[] ids, float[] features);

    static {
        System.loadLibrary("ncnn_jni");
    }

    public void setImageSize(int width, int height) {
        imageHeight = height;
        imageWidth = width;
    }

    public boolean initNcnn(Context context, String root_dir_path) throws IOException
    {
//        if (acc_names.size() > 0) {
//            return true;
//        }
//        String test_dir_path = "/storage/emulated/0/Download/Son";
        int feature_size = 128;
        File dir = new File(root_dir_path);
        File[] subdirs = dir.listFiles();
        ArrayList<float[]> features = new ArrayList<>();
        List<Integer> ids = new ArrayList<>();
        for (int i = 0; i < subdirs.length; i++)
        {
            File[] files = subdirs[i].listFiles();
            for (int j = 0; j < files.length; j++) {
                String file_path = root_dir_path + "/" + subdirs[i].getName() + "/" + files[j].getName();
                Log.d("Files", "FileName: " + file_path);
                float[] readback=new float[feature_size];
                try{
                    // read file
                    RandomAccessFile rFile = new RandomAccessFile(file_path, "rw");
                    FileChannel inChannel = rFile.getChannel();
                    ByteBuffer buf_in = ByteBuffer.allocate(feature_size*4);
                    buf_in.clear();
                    inChannel.read(buf_in);
                    buf_in.rewind();
                    buf_in.asFloatBuffer().get(readback);
                    inChannel.close();

                    // add to java list
                    features.add(readback);
                    ids.add(i);
                    acc_names.add(getName(root_dir_path + "/" + subdirs[i].getName()));
                }
                catch (IOException ex) {
                    System.err.println(ex.getMessage());
                }
            }
        }

        // parse to c++
//        int[] _ids = ids.stream().mapToInt(i->i).toArray();
        int[] _ids = new int[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            _ids[i] = ids.get(i);
        }
        float[] _features = new float[features.size() * feature_size];
        for (int i = 0; i < features.size(); i++) {
            for (int j = 0; j < feature_size; j++) {
                _features[i * feature_size + j] = features.get(i)[j];
            }
        }

        return init(context.getAssets(), _ids, _features);
    }

    public int getEmbed(Context context, ArrayList<String> img_paths, String new_dir_path, int id) {
        ArrayList<float[]> embed_results = new ArrayList<>();
        LOGGER.i("IMG_PATH size %s", img_paths.size());
        int feature_size = 128;
        for (int i = 0; i < img_paths.size(); i++) {
            float[] e_res = embed(context.getAssets(), img_paths.get(i));
            if (e_res.length == feature_size) {
                embed_results.add(e_res);
            }
            LOGGER.i("IMG_PATH %s", img_paths.get(i));
//            LOGGER.i("Embed size %d %d", embed_results.size(), embed_results.get(embed_results.size()-1).length);
        }

        //WRITE to file
        for (int i = 0; i < embed_results.size(); i++) {
            String new_face_path = new_dir_path + "/" + i + ".bin";
            float[] embed_vector = embed_results.get(i);
            try {
                File newFaceFile = new File(new_face_path);
                newFaceFile.createNewFile();
                RandomAccessFile aFile = new RandomAccessFile(new_face_path, "rw");
//                LOGGER.i("Face file %s", new_face_path);
                FileChannel outChannel = aFile.getChannel();

                //one float 4 bytes
//                LOGGER.i("Embed vector length: %d", embed_vector.length);
                ByteBuffer buf = ByteBuffer.allocate(4 * embed_vector.length);
                buf.clear();
                buf.asFloatBuffer().put(embed_vector);
                outChannel.write(buf);
                outChannel.close();

            }
            catch (IOException ex) {
                System.err.println(ex.getMessage());
            }
        }

        int[] _ids = new int[embed_results.size()];
        for (int i = 0; i < embed_results.size(); i++) {
            _ids[i] = id;
        }
        float[] _features = new float[embed_results.size() * feature_size];
        for (int i = 0; i < embed_results.size(); i++) {
            for (int j = 0; j < feature_size; j++) {
                _features[i * feature_size + j] = embed_results.get(i)[j];
            }
        }
        if (load(context.getAssets(), _ids, _features)) {
            // get new name
            String cur_name = getName(new_dir_path);
            acc_names.add(cur_name);
            return 0;
        }
        else {
            return 1;
        }
    }

    public String getName(String full_path) {
        String[] parts = full_path.split("/");
        String last_part = parts[parts.length -1];
        String[] name_parts = last_part.split("_");
        return name_parts[name_parts.length - 1];
    }

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
//        List<String> names = new ArrayList<>();
//        names.add("Son");
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
            if (recognition.id == -1) {
                recognition.title = "new";
            }
            else {
                Log.i("ALL_NAME", "" + acc_names.size() + " " + recognition.id);
                Log.i("ALL_ACC_NAME", "" + acc_names);
                recognition.title = acc_names.get(recognition.id);
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
