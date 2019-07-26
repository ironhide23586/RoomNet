package com.example.roomnetclassifier;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.hardware.Camera;
import android.os.Bundle;
import android.widget.FrameLayout;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;

public class CameraActivity extends Activity {

    private Camera mCamera;
    private CameraPreview mPreview;
    private static final int CAMERA_REQUEST_CODE = 100;
//    private VisionModel visionModel;
    private AssetFileDescriptor tfliteModelFd;

    @Override
    public void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[] {Manifest.permission.CAMERA}, CAMERA_REQUEST_CODE);
        }

        mCamera = getCameraInstance();
        try {
            tfliteModelFd = this.getAssets().openFd("roomnet.tflite");
        }
        catch (IOException e) {
            int k = 0;
        }

        mPreview = new CameraPreview(this, mCamera, tfliteModelFd);
        FrameLayout preview = (FrameLayout) findViewById(R.id.camera_preview);
        preview.addView(mPreview);

//        String modelPath = "roomnet.tflite";
//        try {
//            visionModel = new VisionModel(this, modelPath);
//        }
//        catch (IOException e) {
//            int k = 0;
//        }
//        int k = 0;
    }


    private boolean checkCameraHardware(Context context) {
        if (context.getPackageManager().hasSystemFeature(PackageManager.FEATURE_CAMERA)) {

            return true;
        }
        else {
            return false;
        }
    }

    public static Camera getCameraInstance() {
        Camera c = null;
        try {
            c = Camera.open();
        }
        catch (Exception e) {
            // Camera unavailable
        }
        return c;
    }
}
