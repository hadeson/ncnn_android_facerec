/*
 *  Created by David Chiu
 *  Dec. 28th, 2018 based on tensorflow's original file in android sample
 *
 */


/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.davidchiu.ncnncam;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.text.TextUtils;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.EditText;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import android.content.Intent;
import android.provider.MediaStore;
import java.io.File;
import android.os.Bundle;
import java.io.FileOutputStream;
import java.util.Date;
import java.text.SimpleDateFormat;
import android.os.Environment;
import java.io.IOException;
import android.net.Uri;
import android.support.v4.content.FileProvider;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/coco_labels_list.txt";
  
  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;

  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Ncnn detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  //private MultiBoxTracker tracker;

  private byte[] luminanceCopy;


//  public static final int NCNN_YOLO_WIDTH = 320;
//  public static final int NCNN_YOLO_HEIGHT = 240;
    public static final int NCNN_YOLO_WIDTH = 640;
    public static final int NCNN_YOLO_HEIGHT = 480;
  public int frameWidth ;
  public int frameHeight;
  private Matrix frameToCanvasMatrix;
  public List<Detection> detections;
  private final Paint boxPaint = new Paint();
  private  BorderedText borderedText=null;
  public int cropWidth;          //width and height for image input to model
  public int cropHeight;
  static int total_acc_num = 0;
  static String cur_new_dir_path;

    public void addFace(View view) {
        EditText text = (EditText)findViewById(R.id.editText);
        String edit_text_val = text.getText().toString();
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        String root_dir_path = storageDir.getAbsolutePath() + "/face_embed";
        File rootDir = new File(root_dir_path);
        if (!rootDir.isDirectory()) {
            rootDir.mkdir();
        }
        total_acc_num = rootDir.listFiles().length;
        LOGGER.d("Total acc: " + total_acc_num);
        if (dispatchTakePictureIntent()) {
            LOGGER.d("Add Face Button, img: " + currentPhotoPath);
            LOGGER.d("Edit text val: " + edit_text_val);
            if (!edit_text_val.equals("")) {
                String new_dir_path = root_dir_path+"/"+total_acc_num+"_"+edit_text_val;
                File newDir = new File(new_dir_path);
                if (!newDir.isDirectory()) {
                    newDir.mkdir();
//                    currentPhotoPath.clear();
//                    currentPhotoPath.add("/storage/emulated/0/Android/data/com.davidchiu.ncnncam/files/Pictures/JPEG_20200216_170739_8576944262220727347.jpg");
//                    currentPhotoPath.add("/storage/emulated/0/Dcim/Camera/IMG_20200212_171659.jpg");

                    cur_new_dir_path = new_dir_path;
//                    detector.getEmbed(this, currentPhotoPath, new_dir_path, total_acc_num);

//                    total_acc_num += 1;
                }
                else {
                    currentPhotoPath.clear();
                }
            }
        }
        else {
            LOGGER.d("Add Face Button, FAILED");
        }
    }

    static final int REQUEST_TAKE_PHOTO = 1;
    static final int total_pic_cap = 1;

    private boolean dispatchTakePictureIntent() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        // Ensure that there's a camera activity to handle the intent
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            // Create the File where the photo should go
//            File photoFile = null;
            ArrayList<File> photoFile = new ArrayList<>();
            try {
                photoFile = createImageFile(total_pic_cap);
            } catch (IOException ex) {
                // Error occurred while creating the File
            }
            // Continue only if the File was successfully created
            if (photoFile != null) {
                for (int i = 0; i < total_pic_cap; i++) {
                    Uri photoURI = FileProvider.getUriForFile(this,
//                            "com.example.android.fileprovider",
                            "com.davidchiu.ncnncam.fileprovider",
                            photoFile.get(i));
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
                    startActivityForResult(takePictureIntent, REQUEST_TAKE_PHOTO);
                }
                return true;
            }
            else return false;
        }
        else return false;
    }

    static ArrayList<String> currentPhotoPath = new ArrayList<>();
    private ArrayList<File> createImageFile(int img_num) throws IOException {
        // Create an image file name

        currentPhotoPath.clear();
        ArrayList<File> result = new ArrayList<>();
        for (int i = 0; i < img_num; i++) {
            String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
            String imageFileName = "JPEG_" + timeStamp + "_";
            File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
            File image = File.createTempFile(
                    imageFileName,  /* prefix */
                    ".jpg",         /* suffix */
                    storageDir      /* directory */
            );
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath.add(image.getAbsolutePath());
            result.add(image);
        }

        return result;
    }

//    @Override
//    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
//        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
//            Bundle extras = data.getExtras();
//            Bitmap imageBitmap = (Bitmap) extras.get("data");
//            try (FileOutputStream out = new FileOutputStream(filename)) {
////                bmp.compress(Bitmap.CompressFormat.PNG, 100, out); // bmp is your Bitmap instance
//                // PNG is a lossless format, the compression factor (100) is ignored
//            } catch (IOException e) {
//                e.printStackTrace();
//            }
//        }
//    }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
      final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
      borderedText = new BorderedText(textSizePx);
      borderedText.setTypeface(Typeface.MONOSPACE);

      boxPaint.setColor(Color.RED);
      boxPaint.setStyle(Style.STROKE);
      boxPaint.setStrokeWidth(12.0f);
      boxPaint.setStrokeCap(Paint.Cap.ROUND);
      boxPaint.setStrokeJoin(Paint.Join.ROUND);
      boxPaint.setStrokeMiter(100);

      //tracker = new MultiBoxTracker(this);

      cropWidth = NCNN_YOLO_WIDTH;     //was: 128
      cropHeight = NCNN_YOLO_HEIGHT;    //was: 96


      previewWidth = size.getWidth();
      previewHeight = size.getHeight();
      frameHeight = previewHeight;
      frameWidth = previewWidth;


      try {
        detector = new Ncnn();
        detector.setImageSize(size.getWidth(), size.getHeight());
//        String[] test_img_path = {
//            "/storage/emulated/0/Dcim/Camera/IMG_20200130_152037.jpg",
//            "/storage/emulated/0/Dcim/Camera/IMG_20200212_171702.jpg",
//            "/storage/emulated/0/Dcim/Camera/IMG_20200212_171659.jpg",
//            "/storage/emulated/0/Dcim/Camera/IMG_20200212_171656.jpg",
//            "/storage/emulated/0/Dcim/Camera/IMG_20200212_171654.jpg",
//            "/storage/emulated/0/Dcim/Camera/IMG_20200212_171652.jpg",
//            "/storage/emulated/0/Dcim/Camera/IMG_20200212_171648.jpg",
//        };
//        detector.getEmbed(this, test_img_path);
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        String root_dir_path = storageDir.getAbsolutePath() + "/face_embed";
        File rootDir = new File(root_dir_path);
        if (!rootDir.isDirectory()) {
          rootDir.mkdir();
        }
        detector.initNcnn(this, root_dir_path);
        if (currentPhotoPath.size() > 0) {
            detector.getEmbed(this, currentPhotoPath, cur_new_dir_path, total_acc_num);
            total_acc_num += 1;
            currentPhotoPath.clear();
        }
        //cropSize = TF_OD_API_INPUT_SIZE;
      } catch (final Exception e) {
        //LOGGER.e("Exception initializing classifier!", e);
        Toast toast =
            Toast.makeText(
                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
        toast.show();
        finish();
      }

    sensorOrientation = rotation - getScreenOrientation();
//    sensorOrientation = 0;
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

    if (false) {
        croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Config.ARGB_8888);
        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropWidth, cropHeight,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);
    } else {
        prepare(sensorOrientation);  //90 means vertical screen while 0 means horizontal
    }
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new OverlayView.DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            draw(canvas);
            if (isDebug()) {
              //tracker.drawDebug(canvas);
            }
          }
        });

    addCallback(
        new OverlayView.DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }
            final Bitmap copy = cropCopyBitmap;
            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);
            canvas.drawBitmap(copy, matrix, new Paint());

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString =""; //detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Inference time: " + lastProcessingTimeMs + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

    boolean loadTestImage=false;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
      trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
        Log.i("yolov2ncnn", " detect drop frame: " + timestamp);
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    if (loadTestImage) {
        rgbFrameBitmap = loadTestImage(null);
        if (rgbFrameBitmap == null) {
            readyForNextImage();
            computingDetection = false;
            return;
        }
    } else {
        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    }

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            //final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            detections = detector.detect(croppedBitmap);
//            LOGGER.i("detection id " + detections.get(0).id);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
            Log.i("yolov2", " detect : " + lastProcessingTimeMs);
            if (detections != null) {
                Log.i("detect: ", " objects: " + detections.size());
            }
            if(true) {
             cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
             final Canvas canvas = new Canvas(cropCopyBitmap);
             final Paint paint = new Paint();
             paint.setColor(Color.RED);
             paint.setStyle(Style.STROKE);
             paint.setStrokeWidth(2.0f);

             requestRender();
            }
            computingDetection = false;
          }
        });
  }

    public synchronized void draw(final Canvas canvas) {
        final boolean rotated = sensorOrientation % 180 == 90;
        final float multiplier =
                Math.min(canvas.getHeight() / (float) (rotated ? frameWidth : frameHeight),
                        canvas.getWidth() / (float) (rotated ? frameHeight : frameWidth));
        Log.i("DRAW", "multiplier" + multiplier + " " + frameHeight + " " + frameWidth + " " + canvas.getHeight() + " " + canvas.getWidth() + " " + rotated);
        frameToCanvasMatrix =
                ImageUtils.getTransformationMatrix(
                        frameWidth,
                        frameHeight,
                        (int) (multiplier * (rotated ? frameHeight : frameWidth)),
                        (int) (multiplier * (rotated ? frameWidth : frameHeight)),
//                        1080,
//                        1908
//                        sensorOrientation,
                        0,
                        false);
        if (detections == null) return;
        else {
            Log.i("DRAW", "" + detections.size() );
        }
        for (final Detection recognition : detections) {
            final RectF trackedPos = new RectF(recognition.location);
            LOGGER.i("BOX: %.2f %.2f %.2f %.2f", recognition.location.top, recognition.location.left, recognition.location.bottom, recognition.location.right);

            Log.i("DRAW", "Before map rect" + trackedPos.left + " " + trackedPos.top + " " + trackedPos.right + " " + trackedPos.bottom );
            frameToCanvasMatrix.mapRect(trackedPos);
            Log.i("DRAW", "After map rect" + trackedPos.left + " " + trackedPos.top + " " + trackedPos.right + " " + trackedPos.bottom );
            boxPaint.setColor(recognition.color);

            final float cornerSize = Math.min(trackedPos.width(), trackedPos.height()) / 8.0f;
            canvas.drawRoundRect(trackedPos, cornerSize, cornerSize, boxPaint);

            final String labelString =
                    !TextUtils.isEmpty(recognition.title)
                            ? String.format("%s %.2f", recognition.title, recognition.detectionConfidence)
                            : String.format("%.2f", recognition.detectionConfidence);
            borderedText.drawText(canvas, trackedPos.left + cornerSize, trackedPos.bottom, labelString);
        }
    }

    public void prepare(int sensorOrientation) {
        if (sensorOrientation == 0 || sensorOrientation == 180) {
                croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Bitmap.Config.ARGB_8888);
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                cropWidth, cropHeight,
                                sensorOrientation, MAINTAIN_ASPECT); // from [previewWidth, previewHeight] to [cropW, cropH]

        } else {
                croppedBitmap = Bitmap.createBitmap(cropHeight, cropWidth, Bitmap.Config.ARGB_8888);
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                cropHeight, cropWidth,
                                sensorOrientation, MAINTAIN_ASPECT); // from [previewWidth, previewHeight] to [cropW, cropH]
        }

            cropToFrameTransform = new Matrix();
            frameToCropTransform.invert(cropToFrameTransform);
    }


  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    //detector.enableStatLogging(debug);
  }


    /**
     * This method is for loading test images
     */
    protected int _testcount = -1;
    protected int _testtotal = 1;
    protected String _imagePath = "/sdcard";
    protected String _imagename = "test";
    protected String _imageextension = "jpg";

    protected Bitmap loadTestImage(String path) {
        if (path != null) {
            _imagePath = path;
        }
        if (_imagePath == null || _imageextension == null || _imagename == null) {
            throw new RuntimeException("path/file name/extension not set");
        }
        _testcount++;
        if (_testcount >= _testtotal) {
            _testcount = 0;
        }
        String _file_path_name = _imagePath + "/" + _imagename + _testcount + "." + _imageextension;
        Log.i("loadTestImage", _file_path_name);
        return BitmapFactory.decodeFile(_file_path_name);
    }

}
