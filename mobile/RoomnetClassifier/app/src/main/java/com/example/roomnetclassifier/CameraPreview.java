package com.example.roomnetclassifier;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;


public class VisionModel {
    public MappedByteBuffer buffered_model;
    private Activity parent_activity;
    public String model_fpath;

    public VisionModel(Activity activity, String modelPath) {
        parent_activity = activity;
        model_fpath = modelPath;
        buffered_model = loadModelFile();
    }

    private MappedByteBuffer loadModelFile() {
        AssetFileDescriptor fileDescriptor = parent_activity.getAssets().openFd(model_fpath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        MappedByteBuffer buffer_out = fileChannel.map(FileChannel.MapMode.READ_ONLY,
                startOffset, declaredLength);
        return buffer_out;
    }
}

public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private Camera.PreviewCallback cb;

    public CameraPreview(Context context, Camera camera) {
        super(context);
        mCamera = camera;

        mHolder = getHolder();
        mHolder.addCallback(this);

        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS); // deprecated in new Android
        cb = new Camera.PreviewCallback() {
            @Override
            public void onPreviewFrame(byte[] bytes, Camera camera) {
                int k = 0;
            }
        };
    }

    public void surfaceCreated(SurfaceHolder holder) {
        try {
            mCamera.setPreviewDisplay(holder);
            mCamera.startPreview();
        }
        catch (IOException e) {
            Log.d("tag ", e.getMessage());
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // Release Camera here
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        if (mHolder.getSurface() == null) {
            return;
        }

        // stopping preview before making changes
        try {
            mCamera.stopPreview();
        }
        catch (Exception e) {
            // tried to stop a non-existent preview
        }

        // Do image modification here; change size, rotate, etc

        // Restart preview
        try {
            mCamera.setPreviewDisplay(holder);
            mCamera.setPreviewCallback(cb);
            mCamera.startPreview();
        }
        catch (Exception e) {
            Log.d("tag ", e.getMessage());
        }
    }
}
