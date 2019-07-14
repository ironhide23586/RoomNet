package com.example.roomnetclassifier;

import android.content.Context;
import android.hardware.Camera;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.io.IOException;

public class CameraPreview extends SurfaceView implements SurfaceHolder.Callback {
    private SurfaceHolder mHolder;
    private Camera mCamera;

    public CameraPreview(Context context, Camera camera) {
        super(context);
        mCamera = camera;

        mHolder = getHolder();
        mHolder.addCallback(this);

        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS); // deprecated in new Android
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
        }
        catch (Exception e) {
            Log.d("tag ", e.getMessage());
        }
    }
}
