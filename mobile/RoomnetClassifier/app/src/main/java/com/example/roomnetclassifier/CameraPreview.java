package com.example.roomnetclassifier;

import android.content.Context;

import android.content.res.AssetFileDescriptor;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;

public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private Camera.PreviewCallback cb;
    private VisionModel visionModel;

    public CameraPreview(Context context, Camera camera, AssetFileDescriptor fd) {
        super(context);
        mCamera = camera;

        mHolder = getHolder();
        mHolder.addCallback(this);
        try {
            visionModel = new VisionModel(fd);
        }
        catch (IOException e){
            int k = 0;
        }

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
