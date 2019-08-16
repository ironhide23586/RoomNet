package com.example.roomnetclassifier;

import android.content.Context;

import android.content.res.AssetFileDescriptor;
import android.graphics.ImageFormat;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public class CameraPreview extends SurfaceView
        implements SurfaceHolder.Callback {
    private SurfaceHolder mHolder;
    private Camera mCamera;
    private VisionModel visionModel;

    private int[] rgbBytes = null;
    private byte[][] yuvBytes = new byte[3][];
    public int previewHeight;
    public int previewWidth;
    public boolean isBusy;
    private Camera.Parameters camParams;
    private Camera.Size previewSize;
    private List<Integer> supportedPreviewFormats;
    private int currentPreviewFormat;

    private Camera.PreviewCallback imageListener = new Camera.PreviewCallback() {
        @Override
        public void onPreviewFrame(byte[] bytes, Camera camera) {
            isBusy = true;

            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream('im_cam.bytes'));
            bos.write(fileBytes);
            bos.flush();
            bos.close();

            isBusy = false;
        }
    };

    public CameraPreview(Context context, Camera camera, AssetFileDescriptor fd) {
        super(context);
        mCamera = camera;
        mHolder = getHolder();
        mHolder.addCallback(this);
        try {
            visionModel = new VisionModel(fd);
        } catch (IOException e) {
            int k = 0;
        }

        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS); // deprecated in new Android

        if (rgbBytes == null) {
            camParams = camera.getParameters();
            previewSize = camParams.getPreviewSize();
            supportedPreviewFormats = camParams.getSupportedPreviewFormats();
            camParams.setPreviewFormat(ImageFormat.YV12);
            currentPreviewFormat = camParams.getPreviewFormat();

            previewHeight = previewSize.height;
            previewWidth = previewSize.width;
            rgbBytes = new int[previewWidth * previewHeight];
        }

        mCamera.setPreviewCallback(imageListener);
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
            mCamera.setPreviewCallback(imageListener);
            mCamera.startPreview();
        }
        catch (Exception e) {
            Log.d("tag ", e.getMessage());
        }
    }
}
